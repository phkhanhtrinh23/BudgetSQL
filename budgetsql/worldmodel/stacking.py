"""Calibrating candidate values before they are allocated.

An allocation that picks among noisy predictions favours whatever it
over-estimates, and a rollout compounds the errors of the stages it composes.
Each candidate therefore gets two predictions --- the stage-level rollout and a
head trained on complete workflows --- and a small stack maps the pair onto the
realised outcome.  The stacks are cross-fitted twice over databases: the stack
that plans a database is fitted on other databases' rows, and each of those rows
was scored by heads that saw neither the planned database nor the row's own.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import PlannerConfig
from ..features import QuestionMeta, phi
from ..grammar import STOP, Grammar, State, Workflow, parse_chain
from ..planner.execute import first_block
from ..trace.corpus import TraceCorpus
from .model import StageWorldModel

TIER_ORDER = ("S", "M", "L")


def workflow_descriptor(wf: Workflow) -> np.ndarray:
    """A fixed-size description of a whole workflow for the direct head."""
    return _descriptor(list(wf.stages()), wf.depth(), wf.max_llm_calls())


def state_descriptor(state: State) -> np.ndarray:
    """The same description for the stages a question has already run, so the
    answer it holds can be scored by the same head."""
    stages = [st for st, _ in state.history]
    calls = sum(st.llm_calls for st in stages)
    return _descriptor(stages, len(stages), calls)


def _descriptor(stages: list, depth: int, calls: int) -> np.ndarray:
    kinds = [s.kind for s in stages]
    tiers = [s.tier for s in stages if s.tier]
    v = [
        len(stages), depth, calls,
        float("repair" in kinds), float("escalate" in kinds),
        float("verify" in kinds), float("context" in kinds),
        max([s.samples for s in stages], default=0),
    ]
    v += [float(sum(1 for t in tiers if t == tier)) for tier in TIER_ORDER]
    v += [float(max([TIER_ORDER.index(t) for t in tiers], default=-1))]
    return np.array(v, dtype=float)


class DirectHead:
    """Predicts a whole workflow's outcome from the question's state and the
    workflow's description, without rolling its stages out."""

    def __init__(self, min_rows: int = 40):
        self.min_rows = min_rows
        self.model: object | None = None
        self.prior = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DirectHead":
        self.prior = float(np.mean(y)) if len(y) else 0.5
        if len(y) >= self.min_rows and len(np.unique(y)) == 2:
            self.model = make_pipeline(StandardScaler(),
                                       LogisticRegression(max_iter=1000, C=1.0))
            self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), self.prior)
        return self.model.predict_proba(X)[:, 1]


@dataclass
class ValueStack:
    """Maps the pair of predictions onto a calibrated value and gain."""

    correctness: object | None = None
    gain: object | None = None
    fallback_weight: float = 0.5

    def fit_correctness(self, pair: np.ndarray, y: np.ndarray) -> None:
        if len(y) >= 30 and len(np.unique(y)) == 2:
            self.correctness = make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=1000))
            self.correctness.fit(pair, y)

    def fit_gain(self, pair: np.ndarray, g: np.ndarray) -> None:
        if len(g) >= 30 and np.std(g) > 0:
            self.gain = LinearRegression()
            self.gain.fit(pair, g)

    def value(self, pair: np.ndarray) -> np.ndarray:
        if self.correctness is None:
            return np.clip(pair @ np.array([self.fallback_weight,
                                            1.0 - self.fallback_weight]), 0.0, 1.0)
        return self.correctness.predict_proba(pair)[:, 1]

    def gain_of(self, pair: np.ndarray) -> np.ndarray:
        if self.gain is None:
            return pair @ np.array([self.fallback_weight, 1.0 - self.fallback_weight])
        return self.gain.predict(pair)


@dataclass
class StackedValues:
    """The values the allocation sees: a calibrated value for stop and a
    calibrated gain for every continuation."""

    model: StageWorldModel
    direct: DirectHead
    stack: ValueStack
    cfg: PlannerConfig = field(default_factory=PlannerConfig)

    def with_cfg(self, cfg: PlannerConfig) -> "StackedValues":
        """The same calibration read under another configuration."""
        return StackedValues(self.model.clone(cfg), self.direct, self.stack, cfg)

    def _features(self, metas, states, wfs) -> np.ndarray:
        """A row per candidate: the question's state and what the candidate is.
        Stop is described by the stages the question has already run."""
        return np.stack([
            np.concatenate([phi(m, s),
                            state_descriptor(s) if w.is_stop
                            else workflow_descriptor(w)])
            for m, s, w in zip(metas, states, wfs)])



    def values(self, metas: list[QuestionMeta], states: list[State],
               wfs: list[Workflow], rolls: list, s_held: list[float]) -> np.ndarray:
        """Stacked value of each candidate: stop keeps the held answer's
        calibrated correctness, a continuation adds its calibrated gain."""
        held = np.asarray(s_held, dtype=float)
        rolled = np.array([h if w.is_stop else r.p
                           for w, r, h in zip(wfs, rolls, held)])
        if not self.cfg.use_stacking and self.cfg.use_rollouts:
            return np.clip(rolled, 0.0, 1.0)

        # two predictions per candidate, and the same two for the answer held
        d_candidate = self.direct.predict(self._features(metas, states, wfs))
        d_held = self.direct.predict(
            self._features(metas, states, [STOP] * len(wfs)))
        if not self.cfg.use_rollouts:
            # score each composition as a whole, without rolling its stages out
            rolled, held = d_candidate, d_held
            if not self.cfg.use_stacking:
                return np.clip(rolled, 0.0, 1.0)

        # the calibrated correctness of the answer held, and the calibrated gain
        # a continuation would add to it
        held_value = self.stack.value(np.stack([held, d_held], axis=1))
        gain = self.stack.gain_of(np.stack([rolled - held,
                                            d_candidate - d_held], axis=1))
        out = held_value + np.clip(gain, -1.0, 1.0)
        for i, (wf, state) in enumerate(zip(wfs, states)):
            if not state.has_answer:
                # nothing is held yet, so the rollout is the whole value
                out[i] = rolled[i]
            elif wf.is_stop:
                out[i] = held_value[i]
        return np.clip(out, 0.0, 1.0)


class CrossFitStacker:
    """Fits the direct head and the stacks for one held-out budget group.

    The rows are continuations: an opening workflow is replayed on a training
    question, the state it leaves is the state a continuation would start from,
    and the realised gain over keeping the held answer is what the gain stack
    learns to predict from the two predictions.
    """

    def __init__(self, cfg: PlannerConfig | None = None, inner_folds: int = 4,
                 workflows: dict[str, Workflow] | None = None,
                 grammar: Grammar | None = None,
                 openers: tuple[str, ...] = ("S", "M", "L"),
                 continuations: tuple[str, ...] = ("rep:L", "rep:M",
                                                   "esc:M", "esc:L"),
                 thin: bool = False):
        self.cfg = cfg or PlannerConfig()
        self.inner_folds = inner_folds
        self.grammar = grammar or Grammar()
        self.workflows = workflows or {}
        self.openers = openers
        self.continuations = continuations
        self.thin = thin

    def _keeps(self, meta, oname: str, spec: str) -> bool:
        """Whether a thinned corpus traces this composition on this question.

        A thinned corpus records one composition per question, as a deployment
        that cannot replay everything would, so no question has the outcome of
        a whole workflow other than its own.
        """
        if not self.thin:
            return True
        combos = [(o, c) for o in self.openers
                  for c in ("stop",) + tuple(self.continuations)]
        pick = combos[zlib.crc32(meta.qid.encode()) % len(combos)]
        return pick == (oname, spec)

    def fit(self, corpus: TraceCorpus, holdout: str | list[str]) -> StackedValues:
        held = {holdout} if isinstance(holdout, str) else set(holdout)
        train = [d for d in corpus.dbs if d not in held]
        model = StageWorldModel(self.cfg).fit(corpus, train)

        folds = [train[i::self.inner_folds] for i in range(self.inner_folds)]
        folds = [f for f in folds if f]
        rows: list[_Row] = []
        for fold in folds:
            inner_train = [d for d in train if d not in fold]
            if not inner_train:
                continue
            inner = StageWorldModel(self.cfg).fit(corpus, inner_train)
            rows.extend(self._score(corpus, inner, fold))

        direct = DirectHead()
        stack = ValueStack()
        if rows:
            X = np.vstack([r.features for r in rows])
            y = np.array([r.realised for r in rows])
            direct.fit(X, y)
            d_candidate = direct.predict(X)
            d_held = direct.predict(np.vstack([r.state_features for r in rows]))
            is_stop = np.array([r.is_stop for r in rows], dtype=bool)
            rolled = np.array([r.rolled for r in rows])
            held = np.array([r.held for r in rows])
            if is_stop.any():
                stack.fit_correctness(
                    np.stack([rolled[is_stop], d_candidate[is_stop]], axis=1),
                    y[is_stop])
            cont = ~is_stop
            if cont.any():
                stack.fit_gain(
                    np.stack([rolled[cont] - held[cont],
                              d_candidate[cont] - d_held[cont]], axis=1),
                    y[cont] - np.array([r.stop_value for r in rows])[cont])
        return StackedValues(model, direct, stack, self.cfg)

    def _score(self, corpus: TraceCorpus, inner: StageWorldModel,
               dbs: list[str]) -> list["_Row"]:
        metas = [m for d in dbs for m in corpus.questions_of(d)]
        if not metas:
            return []
        chains = {spec: parse_chain(spec) for spec in self.continuations}
        rows: list[_Row] = []
        for oname in self.openers:
            opener = self.workflows.get(oname)
            if opener is None:
                continue
            opening = first_block(opener)
            states, stops = [], []
            for meta in metas:
                state, ok, _, _ = corpus.run(meta, State(), opening)
                states.append(state)
                stops.append(int(ok))
            helds = inner.s_of_many(list(zip(metas, states)))
            state_rows = {id(state): np.concatenate([phi(meta, state),
                                                      state_descriptor(state)])
                          for meta, state in zip(metas, states)}
            for meta, state, stop, held in zip(metas, states, stops, helds):
                if not self._keeps(meta, oname, "stop"):
                    continue
                rows.append(_Row(
                    features=state_rows[id(state)],
                    state_features=state_rows[id(state)],
                    rolled=held, held=held, realised=float(stop),
                    stop_value=float(stop), is_stop=True))
            for spec, chain in chains.items():
                eligible = [(m, s, st, h) for m, s, st, h
                            in zip(metas, states, stops, helds)
                            if self.grammar.is_wellformed(chain, s)
                            and self._keeps(m, oname, spec)]
                if not eligible:
                    continue
                rolls = inner.rollout_many([(m, s, chain) for m, s, _, _ in eligible],
                                           [h for _, _, _, h in eligible])
                for (meta, state, stop, held), roll in zip(eligible, rolls):
                    _, ok, _, _ = corpus.run(meta, state, chain, held=bool(stop))
                    rows.append(_Row(
                        features=np.concatenate([phi(meta, state),
                                                 workflow_descriptor(chain)]),
                        state_features=state_rows[id(state)],
                        rolled=roll.p, held=held, realised=float(int(ok)),
                        stop_value=float(stop), is_stop=False))
        return rows


@dataclass
class _Row:
    """One calibration row: what was predicted, and what happened."""

    features: np.ndarray
    state_features: np.ndarray
    rolled: float
    held: float
    realised: float
    stop_value: float
    is_stop: bool
