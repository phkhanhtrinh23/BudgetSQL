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
from ..costs import stage_worst_case
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
            one = np.array([s])
            return ResolvedStage(self.labels, np.array([1.0]), one,
                                 np.array([0.0]), np.array([0.0]), np.array([False]),
                                 one, one)
        if self.stage_kind == "generate":
            p = np.array([1.0])
            new = np.array([self.s_new])
            return ResolvedStage(self.labels, p, new, np.zeros(1), np.zeros(1),
                                 np.array([True]), new, new)
        p = s * self.p_given_correct + (1.0 - s) * self.p_given_wrong
        p = np.maximum(p, EPS)
        p = p / p.sum()
        s_old = np.clip(s * self.p_given_correct / p, 0.0, 1.0)
        if self.stage_kind in OBSERVE_ONLY:
            return ResolvedStage(self.labels, p, s_old, np.zeros_like(p),
                                 np.zeros_like(p), np.zeros_like(p, dtype=bool),
                                 s_old, s_old)
        f = np.clip(self.fix if self.fix is not None else np.zeros_like(p), 0.0, 1.0)
        r = np.clip(self.reg if self.reg is not None else np.zeros_like(p), 0.0, 1.0)
        f = np.minimum(f, 1.0 - s_old)
        r = np.minimum(r, s_old)
        keep_new = f > r
        s_kept = np.clip(s_old + f - r, 0.0, 1.0)
        s_after = np.where(keep_new, s_kept, s_old)
        return ResolvedStage(self.labels, p, s_after, f, r, keep_new, s_kept, s_old)


@dataclass
class ResolvedStage:
    labels: tuple[str, ...]
    p_obs: np.ndarray
    s_after: np.ndarray
    fix: np.ndarray
    reg: np.ndarray
    keep_new: np.ndarray
    s_kept: np.ndarray = None
    s_rejected: np.ndarray = None

    def index(self, obs: str) -> int:
        return self.labels.index(obs)

    def s_given(self, j: int, keep: bool) -> float:
        """The held answer's correctness after observation ``j`` when the new
        answer is kept or rejected, whatever Select would have decided."""
        return float((self.s_kept if keep else self.s_rejected)[j])


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
        self._logit_shift: dict[str, float] = {}
        #: training means of a stage's observables given its observation,
        #: imputed into the states a rollout imagines
        self._imputed: dict[tuple, _Values] = {}
        self._diff_mask = differential_mask()

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def fit(self, corpus: TraceCorpus, train_dbs: list[str] | None = None) -> "StageWorldModel":
        self.train_dbs = sorted(train_dbs if train_dbs is not None else corpus.dbs)
        rows = corpus.training_rows(self.train_dbs)
        self._fit_rows(rows)
        return self

    def _fit_imputation(self, rows: list[TrainingRow]) -> None:
        acc: dict[tuple, list] = {}
        for r in rows:
            o = r.outcome
            v = (float(o.n_distinct), float(o.completion_tokens),
                 1.0 if o.equal_to_held else 0.0)
            for key in ((r.stage.id, o.obs), (r.stage.kind, o.obs)):
                acc.setdefault(key, []).append(v)
        self._imputed = {k: _Values(*np.mean(v, axis=0)) for k, v in acc.items()}

    def imputed(self, stage: Stage, obs: str) -> "_Values":
        """The observables a rollout imputes to a stage that observed ``obs``."""
        return (self._imputed.get((stage.id, obs))
                or self._imputed.get((stage.kind, obs))
                or _Values(float(max(1, stage.samples)) if stage.kind == "generate" else 1.0,
                           0.0, 0.0))

    def child(self, state: State, stage: Stage, obs: str, keep: bool = True,
              meta: QuestionMeta | None = None) -> State:
        """An imagined successor state, with imputed observables."""
        return state.append(stage, obs, keep=keep, values=self.imputed(stage, obs))

    def _fit_rows(self, rows: list[TrainingRow]) -> None:
        by_key: dict[str, list[TrainingRow]] = {}
        for row in rows:
            by_key.setdefault(head_key(row.stage), []).append(row)
        c = self.cfg
        self._fit_imputation(rows)
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
        out._imputed = self._imputed
        out._cache = self._cache
        return out

    # ------------------------------------------------------------------ #
    # prediction
    # ------------------------------------------------------------------ #
    def predict(self, meta: QuestionMeta, state: State, stage: Stage) -> StagePrediction:
        key = _cache_key(meta, state, stage)
        got = self._cache.get(key)
        if got is None:
            self.predict_batch([(meta, state, stage)])
            got = self._cache[key]
        return self._scaled(got, meta, stage)

    def predict_raw(self, meta: QuestionMeta, state: State, stage: Stage) -> StagePrediction:
        """The heads' prediction before this batch's online corrections: the
        uncorrected mu-hat that the cost ratio kappa compares realised costs
        against."""
        key = _cache_key(meta, state, stage)
        got = self._cache.get(key)
        if got is None:
            self.predict_batch([(meta, state, stage)])
            got = self._cache[key]
        return got

    def predict_batch(self, items: list[tuple[QuestionMeta, State, Stage]]) -> None:
        """Fill the cache for many (state, stage) pairs with one pass per head."""
        todo: dict[str, list[tuple[tuple, QuestionMeta, State, Stage]]] = {}
        for meta, state, stage in items:
            key = _cache_key(meta, state, stage)
            if key in self._cache:
                continue
            todo.setdefault(head_key(stage), []).append((key, meta, state, stage))
        for hk, group in todo.items():
            heads = self.heads.get(hk)
            stage0 = group[0][3]
            labels = stage0.observations()
            n = len(group)
            if heads is None:
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
        """Apply this database's online cost corrections and, where a batch has
        been recalibrated, the shift of its correctness predictions."""
        shift = self._logit_shift.get(meta.db, 0.0)
        scale = np.ones(D)
        if self.cfg.online_cost_updates:
            scale = np.array([self._kappa.get((meta.db, head_key(stage), k), 1.0)
                              for k in range(D)])
        if np.allclose(scale, 1.0) and shift == 0.0:
            return pred
        s_new = pred.s_new if shift == 0.0 else _shifted(pred.s_new, shift)
        return StagePrediction(pred.stage_kind, pred.labels, pred.p_given_correct,
                               pred.p_given_wrong, s_new, pred.fix, pred.reg,
                               pred.mu * scale, pred.var * scale ** 2,
                               pred.uncertain)

    # ------------------------------------------------------------------ #
    # within-batch recalibration
    # ------------------------------------------------------------------ #
    def recalibrate(self, db: str, items: list[tuple[QuestionMeta, State]],
                    grid: int = 101) -> float:
        """Fit this database's share of correct answers from what its first
        stages observed, and shift the correctness predictions to match it.

        Each stage's observation distribution is stored conditioned on the held
        answer's correctness, so the observed shares of execution errors, empty
        results and agreement identify the share correct without any reference
        label.
        """
        pi = self.estimate_accuracy(items, grid)
        if pi is None:
            return 0.0
        return self.calibrate_to(db, items, pi)

    def calibrate_to(self, db: str, items: list[tuple[QuestionMeta, State]],
                     target: float) -> float:
        """Shift this database's correctness predictions onto a known share."""
        if not items:
            return 0.0
        s_now = float(np.mean(self.s_of_many(items)))
        shift = _shift_to(s_now, target)
        self._logit_shift[db] = self._logit_shift.get(db, 0.0) + shift
        self._cache.clear()
        return self._logit_shift[db]

    def estimate_accuracy(self, items: list[tuple[QuestionMeta, State]],
                          grid: int = 101) -> float | None:
        """The share of correct answers the observed observations imply."""
        rows = []
        for meta, state in items:
            if not state.history:
                continue
            stage, obs = state.history[-1]
            pred = self.predict(meta, state.before_last(), stage)
            try:
                j = pred.labels.index(obs)
            except ValueError:
                continue
            rows.append((float(pred.p_given_correct[j]),
                         float(pred.p_given_wrong[j])))
        if not rows:
            return None
        pis = np.linspace(0.0, 1.0, grid)
        pc = np.array([r[0] for r in rows])
        pw = np.array([r[1] for r in rows])
        like = np.log(np.maximum(np.outer(pis, pc) + np.outer(1.0 - pis, pw), EPS))
        return float(pis[int(np.argmax(like.sum(axis=1)))])

    # ------------------------------------------------------------------ #
    # the held answer
    # ------------------------------------------------------------------ #
    def s_of_state(self, meta: QuestionMeta, state: State) -> float:
        """Predicted correctness of the answer held in a state, obtained by
        replaying the state's own observations through the same arithmetic the
        rollouts use."""
        s = 0.0
        walk = State()
        for i, (stage, obs) in enumerate(state.history):
            pred = self.predict(meta, walk, stage)
            res = pred.resolve(s)
            try:
                j = res.index(obs)
            except ValueError:
                j = 0
            s = res.s_given(j, state.kept(i))
            walk = walk.append(stage, obs, keep=state.kept(i), observables=state.trail[i])
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
                s[i] = res.s_given(j, state.kept(d))
                walk[i] = walk[i].append(stage, obs, keep=state.kept(d),
                                         observables=state.trail[d])
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
            _collect(self, meta, state, wf, needed)
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
            # the branch continues from the answer Select holds
            keep = bool(res.keep_new[j]) if self.cfg.use_select else True
            sub = self._eval(meta, self.child(state, stage, obs, keep),
                             wf.child(obs), res.s_given(j, keep))
            p += w * sub.p
            cost += w * sub.cost
            second += w * (sub.cost ** 2 + sub.var)
            worst = np.maximum(worst, sub.worst_case)
            unc = unc or sub.uncertain
            branch_p.append(w * sub.p)
        var = var + np.maximum(second - (cost - pred.mu) ** 2, 0.0)
        worst = self._worst_case(meta, state, stage) + worst
        spread = float(np.std(branch_p)) if branch_p else 0.0
        return Rollout(float(np.clip(p, 0.0, 1.0)), cost, var, worst, unc, spread)

    def _worst_case(self, meta: QuestionMeta, state: State, stage: Stage) -> np.ndarray:
        """The meter's worst case for one stage at the workload's limits, the
        range c-bar that the Bernstein reserve reads."""
        ctx = stage.ctx if stage.kind == "generate" else state.ctx
        return stage_worst_case(stage, float(meta.prompt_at(ctx)), self.cfg.limits)

    # ------------------------------------------------------------------ #
    # online cost updates
    # ------------------------------------------------------------------ #
    def observe_cost(self, db: str, stage: Stage, predicted: np.ndarray,
                     realized: np.ndarray, completed: bool = True) -> None:
        """A completed call reveals its exact cost.  Cancelled calls are
        censored: what they were charged is not the cost of a completed call.

        ``predicted`` must be the heads' own prediction mu-hat, before any
        correction (``predict_raw``): kappa is the ratio of realised
        to uncorrected predicted cost, and feeding it corrected predictions
        would make it converge to the square root of the true ratio."""
        if not completed or not self.cfg.online_cost_updates:
            return
        hk = head_key(stage)
        heads = self.heads.get(hk)
        base = heads.mean_cost if heads is not None else np.zeros(D)
        nu = self.cfg.prior_calls
        for k in range(D):
            key = (db, hk, k)
            s_real, s_pred = self._cost_obs.get(key, (0.0, 0.0))
            s_real += float(realized[k])
            s_pred += float(predicted[k])
            self._cost_obs[key] = (s_real, s_pred)
            prior = nu * float(base[k])
            denom = s_pred + prior
            self._kappa[key] = (s_real + prior) / denom if denom > EPS else 1.0

    def kappa(self, db: str, stage: Stage) -> np.ndarray:
        return np.array([self._kappa.get((db, head_key(stage), k), 1.0) for k in range(D)])

    def reset_online(self) -> None:
        self._kappa.clear()
        self._cost_obs.clear()
        self._logit_shift.clear()


def _collect(model: "StageWorldModel", meta: QuestionMeta, state: State, wf: Workflow,
             out: list[tuple[QuestionMeta, State, Stage]]) -> None:
    if wf.is_stop:
        return
    out.append((meta, state, wf.stage))
    # Select's decision is only known once the rollout runs, so a replacing
    # stage's children are collected under both
    keeps = (True, False) if wf.stage.kind in REPLACING and state.has_answer else (True,)
    for obs in wf.stage.observations():
        if wf.child(obs).is_stop:
            continue
        for keep in keeps:
            _collect(model, meta, model.child(state, wf.stage, obs, keep),
                     wf.child(obs), out)


@dataclass(frozen=True)
class _Values:
    """Imputed observables of a stage, read by ``Observables.after``."""

    n_distinct: float = 1.0
    completion_tokens: float = 0.0
    equal_to_held: float = 0.0


def _cache_key(meta: QuestionMeta, state: State, stage: Stage) -> tuple:
    """A prediction depends on the branch and on the held answer's observables,
    so a prediction made with imputed observables is never reused for a state
    whose observables were measured."""
    return (meta.qid, state.key, state.observables, stage.id)


def _shifted(s: float, shift: float) -> float:
    s = float(np.clip(s, EPS, 1.0 - EPS))
    odds = s / (1.0 - s) * float(np.exp(shift))
    return float(odds / (1.0 + odds))


def _shift_to(s_now: float, target: float) -> float:
    """The shift in log-odds that moves ``s_now`` onto ``target``."""
    a = float(np.clip(s_now, EPS, 1.0 - EPS))
    b = float(np.clip(target, EPS, 1.0 - EPS))
    return float(np.log(b / (1.0 - b)) - np.log(a / (1.0 - a)))


def _onehot(idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((len(idx), n))
    out[np.arange(len(idx)), idx] = 1.0
    return out
