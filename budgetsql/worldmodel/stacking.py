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
from ..grammar import STOP, Grammar, State, Workflow, parse_chain, single_call_first
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


N_DESCRIPTOR = len(_descriptor([], 0, 0))


def descriptor_box(grammar: Grammar, state: State) -> tuple[np.ndarray, np.ndarray]:
    """Bounds on ``workflow_descriptor(w)`` over every composition ``w`` the
    grammar admits from ``state``, stop excluded: at most as many stages as the
    largest decision tree the grammar allows, at most its depth and its calls
    left, and every flag, count and tier free within those limits."""
    memo: dict = {}

    def sig(st: State) -> tuple:
        return (st.has_answer, st.executed, st.llm_calls, st.last_status,
                None if st.answer_stage is None else st.answer_stage.tier,
                None if st.answer_stage is None else st.answer_stage.samples,
                any(x.kind == "verify" for x, _ in st.history), len(st.history) == 0,
                st.ctx)

    def walk(st: State) -> tuple[int, int, int]:
        """(largest tree, deepest branch, most samples) from ``st``."""
        key = sig(st)
        if key in memo:
            return memo[key]
        nodes = depth = samples = 0
        for a in grammar.actions(st):
            n_a, d_a, s_a = 1, 1, a.samples
            for o in a.observations():
                n, d, sm = walk(st.append(a, o))
                n_a += n
                d_a = max(d_a, 1 + d)
                s_a = max(s_a, sm)
            nodes, depth, samples = max(nodes, n_a), max(depth, d_a), max(samples, s_a)
        memo[key] = (nodes, depth, samples)
        return memo[key]

    nodes, depth, samples = walk(state)
    calls = max(0, grammar.max_llm_calls - state.llm_calls)
    lo = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1], dtype=float)
    hi = np.array([nodes, depth, calls, 1, 1, 1, float(not state.history), samples,
                   nodes, nodes, nodes, len(TIER_ORDER) - 1], dtype=float)
    return lo, np.maximum(lo, hi)


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

    def range_over(self, state_part: np.ndarray, lo: np.ndarray,
                   hi: np.ndarray) -> tuple[float, float]:
        """The smallest and largest prediction for one state over every
        workflow descriptor in the box ``[lo, hi]``.  The head is logistic in
        its scaled inputs, so each extreme sits at a corner of the box."""
        if self.model is None:
            return self.prior, self.prior
        scaler, logit = self.model[0], self.model[-1]
        w = np.ravel(logit.coef_) / np.asarray(scaler.scale_, dtype=float)
        b = float(np.ravel(logit.intercept_)[0]) - float(w @ scaler.mean_)
        k = len(state_part)
        base = b + float(w[:k] @ state_part)
        wd = w[k:]
        top = base + float(np.sum(np.where(wd > 0, wd * hi, wd * lo)))
        bot = base + float(np.sum(np.where(wd > 0, wd * lo, wd * hi)))
        sig = lambda z: float(1.0 / (1.0 + np.exp(-z)))
        return sig(bot), sig(top)


@dataclass
class ValueStack:
    """Maps the pair of predictions onto a calibrated value and gain."""

    correctness: object | None = None
    gain: object | None = None
    #: the stack of a whole composition planned from the empty state, which
    #: values the opening plan
    opening: object | None = None
    fallback_weight: float = 0.5

    def fit_opening(self, pair: np.ndarray, y: np.ndarray) -> None:
        if len(y) >= 30 and len(np.unique(y)) == 2:
            self.opening = make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=1000))
            self.opening.fit(pair, y)

    def opening_value(self, pair: np.ndarray) -> np.ndarray:
        if self.opening is None:
            return np.clip(pair[:, 0], 0.0, 1.0)
        return self.opening.predict_proba(pair)[:, 1]

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

    def _phi(self, meta, state) -> np.ndarray:
        """``phi`` with the world model's input ablations, which the direct
        head and the stacks honour too."""
        return phi(meta, state, use_question=self.cfg.use_question_features,
                   use_execution=self.cfg.use_execution_features)

    def _features(self, metas, states, wfs) -> np.ndarray:
        """A row per candidate: the question's state and what the candidate is.
        Stop is described by the stages the question has already run."""
        return np.stack([
            np.concatenate([self._phi(m, s),
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
        empty = np.array([not st.has_answer for st in states])
        if empty.any():
            # nothing is held yet: a whole composition, valued by the stack
            # fitted on held-out opening rows
            opened = self.stack.opening_value(
                np.stack([rolled[empty], d_candidate[empty]], axis=1))
        k = 0
        for i, (wf, state) in enumerate(zip(wfs, states)):
            if not state.has_answer:
                out[i] = opened[k]
                k += 1
            elif wf.is_stop:
                out[i] = held_value[i]
        return np.clip(out, 0.0, 1.0)


    def bound_forms(self, metas: list[QuestionMeta], states: list[State],
                    s_held: list[float], grammar: Grammar | None = None
                    ) -> list[tuple[float, float, float, float]]:
        """For each question, ``(stop, const, scale, bonus)`` such that every
        continuation w the grammar admits satisfies

            p-tilde(w) <= const + scale * p-hat(w) + bonus,

        with ``stop`` the stacked value of keeping the held answer.  The stacks
        are linear in the rollout, so ``scale * p-hat(w) - lambda' c-hat(w)`` is
        maximised over every composition exactly by backward induction; the
        direct head reads a whole workflow, and ``bonus`` bounds its share by
        its largest value over every workflow description the grammar allows
        from the state (``descriptor_box``).  A Psi computed from this bound is
        at least the exact one, so a failed test still proves that no turn
        can pay.  A continuation whose clipped value falls to zero is
        dominated by stop, which keeps the bound valid."""
        grammar = grammar or Grammar()
        held = np.asarray(s_held, dtype=float)
        n = len(metas)
        if not self.cfg.use_stacking and self.cfg.use_rollouts:
            return [(float(np.clip(h, 0.0, 1.0)), 0.0, 1.0, 0.0) for h in held]
        X_held = self._features(metas, states, [STOP] * n)
        d_held = self.direct.predict(X_held)
        k = X_held.shape[1] - N_DESCRIPTOR
        ranges = []
        for i, state in enumerate(states):
            lo, hi = descriptor_box(grammar, state)
            ranges.append(self.direct.range_over(X_held[i, :k], lo, hi))
        if not self.cfg.use_stacking:
            # flat model, no stack: a continuation's value is the direct head
            return [(float(np.clip(d, 0.0, 1.0)), 0.0, 0.0, rng[1])
                    for d, rng in zip(d_held, ranges)]
        held_in = held if self.cfg.use_rollouts else d_held
        held_value = self.stack.value(np.stack([held_in, d_held], axis=1))
        if self.stack.gain is None:
            c0 = 0.0
            c1 = self.stack.fallback_weight
            c2 = 1.0 - self.stack.fallback_weight
        else:
            c0 = float(self.stack.gain.intercept_)
            c1, c2 = (float(c) for c in np.ravel(self.stack.gain.coef_)[:2])
        out = []
        for i, state in enumerate(states):
            d_lo, d_hi = ranges[i]
            if not state.has_answer:
                # nothing is held yet, so the rollout is the whole value
                out.append((0.0, 0.0, 1.0, 0.0) if self.cfg.use_rollouts
                           else (0.0, 0.0, 0.0, d_hi))
                continue
            stop = float(np.clip(held_value[i], 0.0, 1.0))
            if self.cfg.use_rollouts:
                const = stop + c0 - c1 * held[i] - c2 * d_held[i]
                out.append((stop, const, c1, max(c2 * d_lo, c2 * d_hi)))
            else:
                c = c1 + c2
                const = stop + c0 - c * d_held[i]
                out.append((stop, const, 0.0, max(c * d_lo, c * d_hi)))
        return out


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

    def _phi(self, meta, state) -> np.ndarray:
        return phi(meta, state, use_question=self.cfg.use_question_features,
                   use_execution=self.cfg.use_execution_features)

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
        fold_of: list[int] = []
        for fi, fold in enumerate(folds):
            inner_train = [d for d in train if d not in fold]
            if not inner_train:
                continue
            inner = StageWorldModel(self.cfg).fit(corpus, inner_train)
            got = self._score(corpus, inner, fold)
            rows.extend(got)
            fold_of.extend([fi] * len(got))

        direct = DirectHead()
        stack = ValueStack()
        if rows:
            X = np.vstack([r.features for r in rows])
            S = np.vstack([r.state_features for r in rows])
            y = np.array([r.realised for r in rows])
            # the direct head is cross-fitted like the rollouts: each row is
            # scored by a head trained without that row's databases
            fold_of = np.asarray(fold_of)
            d_candidate = np.zeros(len(rows))
            d_held = np.zeros(len(rows))
            for fi in np.unique(fold_of):
                test = fold_of == fi
                inner_head = DirectHead().fit(X[~test], y[~test])
                d_candidate[test] = inner_head.predict(X[test])
                d_held[test] = inner_head.predict(S[test])
            direct.fit(X, y)
            is_stop = np.array([r.is_stop for r in rows], dtype=bool)
            is_open = np.array([r.is_open for r in rows], dtype=bool)
            rolled = np.array([r.rolled for r in rows])
            held = np.array([r.held for r in rows])
            if not self.cfg.use_rollouts:
                # the flat model scores whole workflows: its stacks are fitted
                # on the direct head's predictions, which is what they read
                rolled, held = d_candidate.copy(), d_held.copy()
            if is_stop.any():
                stack.fit_correctness(
                    np.stack([rolled[is_stop], d_candidate[is_stop]], axis=1),
                    y[is_stop])
            if is_open.any():
                stack.fit_opening(np.stack([rolled[is_open], d_candidate[is_open]], axis=1),
                                  y[is_open])
            cont = ~is_stop & ~is_open
            if cont.any():
                stack.fit_gain(
                    np.stack([rolled[cont] - held[cont],
                              d_candidate[cont] - d_held[cont]], axis=1),
                    y[cont] - np.array([r.stop_value for r in rows])[cont])
        return StackedValues(model, direct, stack, self.cfg)

    def _score_opening(self, corpus: TraceCorpus, inner: StageWorldModel,
                       metas: list) -> list["_Row"]:
        """Rows for the opening plan: each composition the opening solve may
        choose, scored from the empty state and realised whole, under Select."""
        rows: list[_Row] = []
        opening = [wf for wf in self.workflows.values() if single_call_first(wf)]
        if not opening:
            return rows
        empty = State()
        items = [(m, empty, wf) for wf in opening for m in metas]
        rolls = inner.rollout_many(items, [0.0] * len(items))
        for (meta, _, wf), roll in zip(items, rolls):
            _, ok, _, _ = corpus.run(meta, empty, wf, select=selector(inner, meta, 0.0))
            x = np.concatenate([self._phi(meta, empty), workflow_descriptor(wf)])
            rows.append(_Row(features=x, state_features=x, rolled=roll.p, held=0.0,
                             realised=float(int(ok)), stop_value=0.0, is_stop=False,
                             is_open=True))
        return rows

    def _score(self, corpus: TraceCorpus, inner: StageWorldModel,
               dbs: list[str]) -> list["_Row"]:
        metas = [m for d in dbs for m in corpus.questions_of(d)]
        if not metas:
            return []
        chains = {spec: parse_chain(spec) for spec in self.continuations}
        rows: list[_Row] = self._score_opening(corpus, inner, metas)
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
            state_rows = {id(state): np.concatenate([self._phi(meta, state),
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
                    # realised as the planner would run it: Select decides
                    # whether each new answer is kept
                    _, ok, _, _ = corpus.run(meta, state, chain, held=bool(stop),
                                             select=selector(inner, meta, held))
                    rows.append(_Row(
                        features=np.concatenate([self._phi(meta, state),
                                                 workflow_descriptor(chain)]),
                        state_features=state_rows[id(state)],
                        rolled=roll.p, held=held, realised=float(int(ok)),
                        stop_value=float(stop), is_stop=False))
        return rows


def selector(model: StageWorldModel, meta: QuestionMeta, s_held: float):
    """Select as ``TraceCorpus.run`` asks for it: the decision on each replacing
    stage's new answer, tracking the held answer's predicted correctness along
    the branch.  With Select off, every new answer is kept."""
    s = [float(s_held)]

    def select(state: State, stage, out) -> bool:
        res = model.predict(meta, state, stage).resolve(s[0])
        try:
            j = res.index(out.obs)
        except ValueError:
            return True
        keep = bool(res.keep_new[j]) if model.cfg.use_select else True
        s[0] = res.s_given(j, keep)
        return keep

    def track(state: State, stage, out) -> None:
        res = model.predict(meta, state, stage).resolve(s[0])
        try:
            s[0] = res.s_given(res.index(out.obs), True)
        except ValueError:
            pass

    select.observe = track
    return select


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
    is_open: bool = False
