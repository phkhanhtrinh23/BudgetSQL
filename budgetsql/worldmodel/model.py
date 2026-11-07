"""The stage-level world model: what a stage would observe, what it would do to
the held answer, and what it would cost --- all without calling an LLM.

Predictions are made per stage, so any composition the agent proposes can be
evaluated by rolling its decision tree out, including compositions no training
question ran as a whole.  Rollouts are batched: a planner asks for many
questions at once, the model groups the rows it needs by head and evaluates each
head once.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import D, DIMS, PlannerConfig
from ..features import (QuestionMeta, differential_mask, phi,
                        stage_descriptor)
from ..grammar import (OBSERVE_ONLY, REPLACING, Stage, State,
                       Workflow)
from ..trace.corpus import TraceCorpus, TrainingRow
from .heads import FIXREG_LABELS, StageHeads, head_key, make_heads

EPS = 1e-9


@dataclass
class StagePrediction:
    """A stage's predicted observation distribution, effect on the held answer
    and cost.  The observation distribution is stored conditioned on the held
    answer's correctness, so resolving it against a prior keeps the arithmetic
    consistent."""

    stage_kind: str
    labels: tuple[str, ...]
    p_given_correct: np.ndarray
    p_given_wrong: np.ndarray
    s_new: float = 0.0
    fix: np.ndarray | None = None
    reg: np.ndarray | None = None
    mu: np.ndarray = field(default_factory=lambda: np.zeros(D))
    var: np.ndarray = field(default_factory=lambda: np.zeros(D))
    uncertain: bool = False

    def resolve(self, s_held: float) -> "ResolvedStage":
        s = float(np.clip(s_held, 0.0, 1.0))
        if self.stage_kind == "context":
            return ResolvedStage(self.labels, np.array([1.0]), np.array([s]),
                                 np.array([0.0]), np.array([0.0]), np.array([False]))
        if self.stage_kind == "generate":
            p = np.array([1.0])
            return ResolvedStage(self.labels, p, np.array([self.s_new]),
                                 np.zeros(1), np.zeros(1), np.array([True]))
        p = s * self.p_given_correct + (1.0 - s) * self.p_given_wrong
        p = np.maximum(p, EPS)
        p = p / p.sum()
        s_old = np.clip(s * self.p_given_correct / p, 0.0, 1.0)
        if self.stage_kind in OBSERVE_ONLY:
            return ResolvedStage(self.labels, p, s_old, np.zeros_like(p),
                                 np.zeros_like(p), np.zeros_like(p, dtype=bool))
        f = np.clip(self.fix if self.fix is not None else np.zeros_like(p), 0.0, 1.0)
        r = np.clip(self.reg if self.reg is not None else np.zeros_like(p), 0.0, 1.0)
        f = np.minimum(f, 1.0 - s_old)          # a fix needs a wrong held answer
        r = np.minimum(r, s_old)                # a regression needs a right one
        keep_new = f > r
        s_after = np.clip(s_old + np.maximum(f - r, 0.0), 0.0, 1.0)
        return ResolvedStage(self.labels, p, s_after, f, r, keep_new)


@dataclass
class ResolvedStage:
    labels: tuple[str, ...]
    p_obs: np.ndarray
    s_after: np.ndarray
    fix: np.ndarray
    reg: np.ndarray
    keep_new: np.ndarray

    def index(self, obs: str) -> int:
        return self.labels.index(obs)


@dataclass
class Rollout:
    """What a continuation is predicted to buy and to cost from one state."""

    p: float
    cost: np.ndarray
    var: np.ndarray
    worst_case: np.ndarray
    uncertain: bool = False
    spread: float = 0.0

    @property
    def as_dict(self) -> dict:
        return {"p": self.p, **{f"cost_{k}": float(v) for k, v in zip(DIMS, self.cost)}}


class StageWorldModel:
    """Trained on a trace corpus, leaving out the databases it will plan."""

    def __init__(self, cfg: PlannerConfig | None = None):
        self.cfg = cfg or PlannerConfig()
        self.heads: dict[str, StageHeads] = {}
        self.train_dbs: list[str] = []
        self._cache: dict[tuple, StagePrediction] = {}
        self._kappa: dict[tuple[str, str, int], float] = {}
        self._cost_obs: dict[tuple[str, str, int], tuple[float, float]] = {}
        self._diff_mask = differential_mask()

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def fit(self, corpus: TraceCorpus, train_dbs: list[str] | None = None) -> "StageWorldModel":
        self.train_dbs = sorted(train_dbs if train_dbs is not None else corpus.dbs)
        rows = corpus.training_rows(self.train_dbs)
        self._fit_rows(rows)
        return self

    def _fit_rows(self, rows: list[TrainingRow]) -> None:
        by_key: dict[str, list[TrainingRow]] = {}
        for row in rows:
            by_key.setdefault(head_key(row.stage), []).append(row)
        c = self.cfg
        for key, group in by_key.items():
            stage = group[0].stage
            labels = stage.observations()
            heads = make_heads(labels, stage.kind, min_support=c.min_support,
                               n_estimators=c.n_estimators, num_leaves=c.num_leaves,
                               learning_rate=c.learning_rate,
                               log_scale=c.log_scale_costs)
            X = np.stack([self._row_features(r) for r in group])
            y_obs = np.array([labels.index(r.outcome.obs) for r in group])
            held = np.array([1 if r.held_before else 0 for r in group])
            costs = np.stack([r.outcome.cost for r in group])

            if heads.needs_conditional:
                for mask, head in ((held == 1, heads.obs_given_correct),
                                   (held == 0, heads.obs_given_wrong)):
                    head.fit(X[mask], y_obs[mask]) if mask.any() else head.fit(X[:0], y_obs[:0])
            else:
                heads.obs_marginal.fit(X, y_obs)

            if stage.kind in REPLACING:
                lab = np.zeros(len(group), dtype=int)
                for i, r in enumerate(group):
                    if r.outcome.is_fix:
                        lab[i] = FIXREG_LABELS.index("fix")
                    elif r.outcome.is_regression:
                        lab[i] = FIXREG_LABELS.index("regression")
                Xo = np.hstack([X, _onehot(y_obs, len(labels))])
                if c.use_differential_features:
                    heads.fixreg.fit(Xo, lab)
                else:
                    heads.fixreg.fit(self._drop_differential(Xo), lab)
            if stage.kind == "generate":
                y = np.array([1 if r.outcome.answer_correct else 0 for r in group])
                heads.success.fit(X, y)

            for k in range(D):
                heads.cost[k].fit(X, costs[:, k])
            heads.mean_cost = costs.mean(axis=0)
            heads.n_rows = len(group)
            self.heads[key] = heads
        self._cache.clear()

    def _row_features(self, row: TrainingRow) -> np.ndarray:
        return self._features(row.meta, row.state, row.stage)

    def _features(self, meta: QuestionMeta, state: State, stage: Stage) -> np.ndarray:
        x = phi(meta, state,
                use_question=self.cfg.use_question_features,
                use_execution=self.cfg.use_execution_features)
        return np.concatenate([x, stage_descriptor(stage)])

    def _drop_differential(self, X: np.ndarray) -> np.ndarray:
        keep = np.concatenate([self._diff_mask,
                               np.ones(X.shape[1] - len(self._diff_mask), dtype=bool)])
        return X[:, keep]

    def clone(self, cfg: PlannerConfig | None = None) -> "StageWorldModel":
        """A second view of the same fitted heads with its own online state.

        Two arms of an experiment must not share cost corrections, and two
        planners alive at once must not share a configuration.
        """
        out = StageWorldModel(cfg or self.cfg)
        out.heads = self.heads
        out.train_dbs = self.train_dbs
        out._cache = self._cache          # predictions are pre-correction
        return out

    # ------------------------------------------------------------------ #
    # prediction
    # ------------------------------------------------------------------ #
    def predict(self, meta: QuestionMeta, state: State, stage: Stage) -> StagePrediction:
        key = (meta.qid, state.key, stage.id)
        got = self._cache.get(key)
        if got is None:
            self.predict_batch([(meta, state, stage)])
            got = self._cache[key]
        return self._scaled(got, meta, stage)

    def predict_batch(self, items: list[tuple[QuestionMeta, State, Stage]]) -> None:
        """Fill the cache for many (state, stage) pairs with one pass per head."""
        todo: dict[str, list[tuple[tuple, QuestionMeta, State, Stage]]] = {}
        for meta, state, stage in items:
            key = (meta.qid, state.key, stage.id)
            if key in self._cache:
                continue
            todo.setdefault(head_key(stage), []).append((key, meta, state, stage))
        for hk, group in todo.items():
            heads = self.heads.get(hk)
            stage0 = group[0][3]
            labels = stage0.observations()
            n = len(group)
            if heads is None:                      # a stage type the corpus never ran
                flat = np.full((n, len(labels)), 1.0 / len(labels))
                for i, (key, meta, state, stage) in enumerate(group):
                    self._cache[key] = StagePrediction(
                        stage.kind, labels, flat[i], flat[i], 0.5,
                        np.zeros(len(labels)), np.zeros(len(labels)),
                        np.zeros(D), np.zeros(D), True)
                continue
            X = np.stack([self._features(m, s, a) for _, m, s, a in group])
            if heads.needs_conditional:
                pc = heads.obs_given_correct.predict(X)
                pw = heads.obs_given_wrong.predict(X)
            else:
                pm = heads.obs_marginal.predict(X)
                pc = pw = pm
            s_new = heads.success.predict(X) if heads.success is not None else np.zeros(n)
            mu = np.stack([heads.cost[k].predict(X) for k in range(D)], axis=1)
            var = np.array([h.resid_var for h in heads.cost])
            fix = reg = None
            if stage0.kind in REPLACING and heads.fixreg is not None:
                fix = np.zeros((n, len(labels)))
                reg = np.zeros((n, len(labels)))
                for j in range(len(labels)):
                    Xo = np.hstack([X, _onehot(np.full(n, j), len(labels))])
                    if not self.cfg.use_differential_features:
                        Xo = self._drop_differential(Xo)
                    p = heads.fixreg.predict(Xo)
                    fix[:, j] = p[:, FIXREG_LABELS.index("fix")]
                    reg[:, j] = p[:, FIXREG_LABELS.index("regression")]
            unc = heads.uncertain()
            for i, (key, meta, state, stage) in enumerate(group):
                self._cache[key] = StagePrediction(
                    stage.kind, labels, pc[i], pw[i], float(s_new[i]),
                    None if fix is None else fix[i], None if reg is None else reg[i],
                    np.maximum(mu[i], 0.0), var, unc)

    def _scaled(self, pred: StagePrediction, meta: QuestionMeta,
                stage: Stage) -> StagePrediction:
        """Apply the online cost corrections of this database and stage type."""
        if not self.cfg.online_cost_updates:
            return pred
        scale = np.array([self._kappa.get((meta.db, stage.kind, k), 1.0) for k in range(D)])
        if np.allclose(scale, 1.0):
            return pred
        out = StagePrediction(pred.stage_kind, pred.labels, pred.p_given_correct,
                              pred.p_given_wrong, pred.s_new, pred.fix, pred.reg,
                              pred.mu * scale, pred.var * scale ** 2, pred.uncertain)
        return out

    # ------------------------------------------------------------------ #
    # the held answer
    # ------------------------------------------------------------------ #
    def s_of_state(self, meta: QuestionMeta, state: State) -> float:
        """Predicted correctness of the answer held in a state, obtained by
        replaying the state's own observations through the same arithmetic the
        rollouts use."""
        s = 0.0
        walk = State()
        for stage, obs in state.history:
            pred = self.predict(meta, walk, stage)
            res = pred.resolve(s)
            try:
                j = res.index(obs)
            except ValueError:
                j = 0
            s = float(res.s_after[j])
            walk = walk.append(stage, obs)
        return s

    def s_of_many(self, items: list[tuple[QuestionMeta, State]]) -> list[float]:
        """The same for many states at once, one batch of head evaluations per
        depth rather than one per stage."""
        s = [0.0] * len(items)
        walk = [State() for _ in items]
        depth = max((len(st) for _, st in items), default=0)
        for d in range(depth):
            live = [i for i, (_, st) in enumerate(items) if len(st) > d]
            if not live:
                break
            batch = [(items[i][0], walk[i], items[i][1].history[d][0]) for i in live]
            self.predict_batch(batch)
            for i in live:
                meta, state = items[i]
                stage, obs = state.history[d]
                res = self.predict(meta, walk[i], stage).resolve(s[i])
                try:
                    j = res.index(obs)
                except ValueError:
                    j = 0
                s[i] = float(res.s_after[j])
                walk[i] = walk[i].append(stage, obs)
        return s

    def select(self, meta: QuestionMeta, state: State, stage: Stage, obs: str,
               s_held: float) -> bool:
        """Keep the new answer only where a fix is likelier than a regression."""
        res = self.predict(meta, state, stage).resolve(s_held)
        try:
            j = res.index(obs)
        except ValueError:
            return True
        return bool(res.keep_new[j])

    # ------------------------------------------------------------------ #
    # rollouts
    # ------------------------------------------------------------------ #
    def rollout(self, meta: QuestionMeta, state: State, wf: Workflow,
                s_held: float | None = None) -> Rollout:
        return self.rollout_many([(meta, state, wf)], [s_held] if s_held is not None else None)[0]

    def rollout_many(self, items: list[tuple[QuestionMeta, State, Workflow]],
                     s_held: list[float] | None = None) -> list[Rollout]:
        """Roll out many continuations, batching every head evaluation."""
        needed: list[tuple[QuestionMeta, State, Stage]] = []
        for meta, state, wf in items:
            _collect(meta, state, wf, needed)
        if needed:
            self.predict_batch(needed)
        if s_held is None:
            s_held = self.s_of_many([(m, st) for m, st, _ in items])
        return [self._eval(meta, state, wf, s_held[i])
                for i, (meta, state, wf) in enumerate(items)]

    def _eval(self, meta: QuestionMeta, state: State, wf: Workflow,
              s_held: float) -> Rollout:
        if wf.is_stop:
            return Rollout(float(np.clip(s_held, 0.0, 1.0)), np.zeros(D), np.zeros(D),
                           np.zeros(D))
        stage = wf.stage
        pred = self.predict(meta, state, stage)
        res = pred.resolve(s_held)
        p = 0.0
        cost = pred.mu.astype(float).copy()
        second = np.zeros(D)
        worst = np.zeros(D)
        var = pred.var.astype(float).copy()
        unc = pred.uncertain
        branch_p: list[float] = []
        for j, obs in enumerate(res.labels):
            w = float(res.p_obs[j])
            if w <= 0.0:
                continue
            sub = self._eval(meta, state.append(stage, obs), wf.child(obs),
                             float(res.s_after[j]))
            p += w * sub.p
            cost += w * sub.cost
            second += w * (sub.cost ** 2 + sub.var)
            worst = np.maximum(worst, sub.worst_case)
            unc = unc or sub.uncertain
            branch_p.append(w * sub.p)
        var = var + np.maximum(second - (cost - pred.mu) ** 2, 0.0)
        worst = _worst_case(stage) + worst
        spread = float(np.std(branch_p)) if branch_p else 0.0
        return Rollout(float(np.clip(p, 0.0, 1.0)), cost, var, worst, unc, spread)

    # ------------------------------------------------------------------ #
    # online cost updates
    # ------------------------------------------------------------------ #
    def observe_cost(self, db: str, stage: Stage, predicted: np.ndarray,
                     realized: np.ndarray, completed: bool = True) -> None:
        """A completed call reveals its exact cost.  Cancelled calls are
        censored: what they were charged is not the cost of a completed call."""
        if not completed or not self.cfg.online_cost_updates:
            return
        heads = self.heads.get(f"{stage.kind}/{len(stage.observations())}")
        base = heads.mean_cost if heads is not None else np.zeros(D)
        nu = self.cfg.prior_calls
        for k in range(D):
            key = (db, stage.kind, k)
            s_real, s_pred = self._cost_obs.get(key, (0.0, 0.0))
            s_real += float(realized[k])
            s_pred += float(predicted[k])
            self._cost_obs[key] = (s_real, s_pred)
            prior = nu * float(base[k])
            denom = s_pred + prior
            self._kappa[key] = (s_real + prior) / denom if denom > EPS else 1.0

    def kappa(self, db: str, stage_kind: str) -> np.ndarray:
        return np.array([self._kappa.get((db, stage_kind, k), 1.0) for k in range(D)])

    def reset_online(self) -> None:
        self._kappa.clear()
        self._cost_obs.clear()


def _collect(meta: QuestionMeta, state: State, wf: Workflow,
             out: list[tuple[QuestionMeta, State, Stage]]) -> None:
    if wf.is_stop:
        return
    out.append((meta, state, wf.stage))
    for obs in wf.stage.observations():
        _collect(meta, state.append(wf.stage, obs), wf.child(obs), out)


def _worst_case(stage: Stage) -> np.ndarray:
    """A conservative bound on one stage's cost, used to order the meter's
    reservations when no measured worst case is at hand."""
    if stage.llm_calls == 0:
        return np.zeros(D)
    from ..config import PRICES
    price = PRICES.get(stage.tier)
    if price is None:
        return np.zeros(D)
    return np.array([price.out * stage.cap * 1e-6 * stage.samples,
                     float(stage.cap * stage.samples),
                     10.0 * stage.samples])


def _onehot(idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((len(idx), n))
    out[np.arange(len(idx)), idx] = 1.0
    return out
