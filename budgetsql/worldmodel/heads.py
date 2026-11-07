"""Gradient-boosted heads for one stage type.

Each stage type gets three heads: a multi-class head for what the stage will
observe, a head for the answer held after it (correctness at the first
answer-producing stage, a fix-and-regression head at a stage that can replace
the answer), and one squared-loss regressor per budget dimension.

The observation heads are trained conditioned on whether the held answer is
correct.  A stage that only observes is then exactly neutral: averaged over its
observations, the posterior correctness it implies is the prior again, so the
allocation cannot buy accuracy from observing alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from ..config import D
from ..grammar import OBSERVE_ONLY, REPLACING, Stage

EPS = 1e-9


def head_key(stage: Stage) -> str:
    """Heads are shared by stage type; stages of one type whose observation
    alphabets differ (one sample or three) get their own alphabet."""
    return f"{stage.kind}/{len(stage.observations())}"


class _Classifier:
    """A multi-class head that falls back to the training frequencies when a
    class is too rare for a tree to learn anything about it."""

    def __init__(self, labels: tuple[str, ...], min_support: int = 30, **kw):
        self.labels = labels
        self.min_support = min_support
        self.kw = kw
        self.prior = np.full(len(labels), 1.0 / max(1, len(labels)))
        self.counts = np.zeros(len(labels))
        self.model: HistGradientBoostingClassifier | None = None
        self.classes_: list[int] = []

    @property
    def rare(self) -> np.ndarray:
        return self.counts < self.min_support

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_Classifier":
        for i in range(len(self.labels)):
            self.counts[i] = float((y == i).sum())
        total = max(1.0, self.counts.sum())
        self.prior = (self.counts + 0.5) / (total + 0.5 * len(self.labels))
        present = [i for i in range(len(self.labels)) if self.counts[i] >= self.min_support]
        if len(present) >= 2 and len(X) >= 2 * self.min_support:
            mask = np.isin(y, present)
            self.model = HistGradientBoostingClassifier(**self.kw)
            self.model.fit(X[mask], y[mask])
            self.classes_ = list(self.model.classes_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        if self.model is None:
            out = np.tile(self.prior, (n, 1))
        else:
            # rare classes keep their pooled frequency; the trees share the rest
            rare_mass = float(self.prior[self.rare].sum())
            out = np.zeros((n, len(self.labels)))
            out[:, self.rare] = self.prior[self.rare]
            proba = self.model.predict_proba(X)
            for j, c in enumerate(self.classes_):
                out[:, c] = proba[:, j] * (1.0 - rare_mass)
        return out / np.maximum(out.sum(axis=1, keepdims=True), EPS)

    def uncertain(self) -> bool:
        return bool(self.rare.any())


class _Binary:
    def __init__(self, min_support: int = 30, **kw):
        self.min_support = min_support
        self.kw = kw
        self.prior = 0.5
        self.model: HistGradientBoostingClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_Binary":
        y = y.astype(int)
        self.prior = float((y.sum() + 0.5) / (len(y) + 1.0)) if len(y) else 0.5
        if len(np.unique(y)) == 2 and len(X) >= 2 * self.min_support:
            self.model = HistGradientBoostingClassifier(**self.kw)
            self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), self.prior)
        return self.model.predict_proba(X)[:, 1]


class _Regressor:
    """Costs are regressed in their original scale: a budget pays the arithmetic
    mean of a stage's cost, which a squared-error fit of the logarithm would
    under-state."""

    def __init__(self, log_scale: bool = False, min_support: int = 30, **kw):
        self.log_scale = log_scale
        self.min_support = min_support
        self.kw = kw
        self.mean = 0.0
        self.resid_var = 0.0
        self.model: HistGradientBoostingRegressor | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_Regressor":
        self.mean = float(np.mean(y)) if len(y) else 0.0
        target = np.log1p(np.maximum(y, 0.0)) if self.log_scale else y
        if len(X) >= 2 * self.min_support and np.std(target) > 0:
            self.model = HistGradientBoostingRegressor(**self.kw)
            self.model.fit(X, target)
            pred = self.predict(X)
        else:
            pred = np.full(len(y), self.mean)
        self.resid_var = float(np.mean((y - pred) ** 2)) if len(y) else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), self.mean)
        p = self.model.predict(X)
        if self.log_scale:
            p = np.expm1(p)
        return np.maximum(p, 0.0)


@dataclass
class StageHeads:
    """The heads of one stage type."""

    labels: tuple[str, ...]
    kind: str
    obs_given_correct: _Classifier | None = None
    obs_given_wrong: _Classifier | None = None
    obs_marginal: _Classifier | None = None
    success: _Binary | None = None
    fixreg: _Classifier | None = None
    cost: list[_Regressor] = field(default_factory=list)
    mean_cost: np.ndarray = field(default_factory=lambda: np.zeros(D))
    n_rows: int = 0

    @property
    def needs_conditional(self) -> bool:
        return self.kind in OBSERVE_ONLY or self.kind in REPLACING

    def uncertain(self) -> bool:
        heads = [self.obs_given_correct, self.obs_given_wrong, self.obs_marginal, self.fixreg]
        return any(h is not None and h.uncertain() for h in heads)


FIXREG_LABELS = ("none", "fix", "regression")


def make_heads(labels: tuple[str, ...], kind: str, *, min_support: int = 30,
               n_estimators: int = 150, num_leaves: int = 15,
               learning_rate: float = 0.05, log_scale: bool = False) -> StageHeads:
    kw = dict(max_iter=n_estimators, max_leaf_nodes=num_leaves,
              learning_rate=learning_rate, early_stopping=False,
              min_samples_leaf=20, l2_regularization=1.0)
    h = StageHeads(labels=labels, kind=kind)
    if kind in OBSERVE_ONLY or kind in REPLACING:
        h.obs_given_correct = _Classifier(labels, min_support, **kw)
        h.obs_given_wrong = _Classifier(labels, min_support, **kw)
    else:
        h.obs_marginal = _Classifier(labels, min_support, **kw)
    if kind in REPLACING:
        h.fixreg = _Classifier(FIXREG_LABELS, min_support, **kw)
        h.success = _Binary(min_support, **kw)
    if kind == "generate":
        h.success = _Binary(min_support, **kw)
    h.cost = [_Regressor(log_scale, min_support, **kw) for _ in range(D)]
    return h
