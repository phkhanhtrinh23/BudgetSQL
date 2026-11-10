"""One budget for a whole batch of several databases.

A deployment may give its day's questions one budget instead of one per
database, which lets a planner move slack between databases.  Every learned
component must still be fitted without the database it plans, so a pooled batch
is served by the per-database fits, routed by each question's own database.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import PlannerConfig
from ..features import QuestionMeta
from ..grammar import State, Workflow


class PooledModel:
    """The per-database world models behind one interface."""

    def __init__(self, by_db: dict[str, object]):
        self.by_db = dict(by_db)
        first = next(iter(self.by_db.values()))
        self.cfg: PlannerConfig = first.cfg
        self.heads = first.heads

    def _of(self, meta: QuestionMeta):
        return self.by_db[meta.db]

    def clone(self, cfg: PlannerConfig | None = None) -> "PooledModel":
        return PooledModel({d: m.clone(cfg) for d, m in self.by_db.items()})

    def predict(self, meta, state, stage):
        return self._of(meta).predict(meta, state, stage)

    def child(self, state, stage, obs, keep=True, meta=None):
        return self._of(meta).child(state, stage, obs, keep, meta)

    def predict_raw(self, meta, state, stage):
        return self._of(meta).predict_raw(meta, state, stage)

    def predict_batch(self, items):
        by_db: dict[str, list] = {}
        for item in items:
            by_db.setdefault(item[0].db, []).append(item)
        for db, group in by_db.items():
            self.by_db[db].predict_batch(group)

    def s_of_state(self, meta, state) -> float:
        return self._of(meta).s_of_state(meta, state)

    def s_of_many(self, items) -> list[float]:
        return self._route(items, lambda m, g: m.s_of_many(g))

    def select(self, meta, state, stage, obs, s_held) -> bool:
        return self._of(meta).select(meta, state, stage, obs, s_held)

    def rollout(self, meta, state, wf, s_held=None):
        return self._of(meta).rollout(meta, state, wf, s_held)

    def rollout_many(self, items, s_held=None):
        if s_held is None:
            return self._route(items, lambda m, g: m.rollout_many(g))
        index: dict[str, list[int]] = {}
        for i, item in enumerate(items):
            index.setdefault(item[0].db, []).append(i)
        out: list = [None] * len(items)
        for db, idx in index.items():
            got = self.by_db[db].rollout_many([items[i] for i in idx],
                                              [s_held[i] for i in idx])
            for i, r in zip(idx, got):
                out[i] = r
        return out

    def recalibrate(self, db, items, grid: int = 101) -> float:
        return self.by_db[db].recalibrate(db, items, grid)

    def calibrate_to(self, db, items, target) -> float:
        return self.by_db[db].calibrate_to(db, items, target)

    def estimate_accuracy(self, items, grid: int = 101):
        return next(iter(self.by_db.values())).estimate_accuracy(items, grid)

    def observe_cost(self, db, stage, predicted, realized, completed=True):
        self.by_db[db].observe_cost(db, stage, predicted, realized, completed)

    def kappa(self, db: str, stage) -> np.ndarray:
        return self.by_db[db].kappa(db, stage)

    def reset_online(self) -> None:
        for m in self.by_db.values():
            m.reset_online()

    @property
    def _kappa(self) -> dict:
        out: dict = {}
        for m in self.by_db.values():
            out.update(m._kappa)
        return out

    def _route(self, items, call):
        index: dict[str, list[int]] = {}
        for i, item in enumerate(items):
            index.setdefault(item[0].db, []).append(i)
        out: list = [None] * len(items)
        for db, idx in index.items():
            for i, r in zip(idx, call(self.by_db[db], [items[i] for i in idx])):
                out[i] = r
        return out


@dataclass
class PooledValues:
    """The per-database calibrations behind one interface."""

    by_db: dict[str, object] = field(default_factory=dict)
    cfg: PlannerConfig = field(default_factory=PlannerConfig)
    model: PooledModel | None = None

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = PooledModel({d: v.model for d, v in self.by_db.items()})

    def with_cfg(self, cfg: PlannerConfig) -> "PooledValues":
        return PooledValues({d: v.with_cfg(cfg) for d, v in self.by_db.items()}, cfg)

    def bound_forms(self, metas, states, s_held, grammar=None) -> list:
        """Each question's certificate bound from its own database's stacks."""
        out: list = [None] * len(metas)
        index: dict[str, list[int]] = {}
        for i, m in enumerate(metas):
            index.setdefault(m.db, []).append(i)
        for db, idx in index.items():
            got = self.by_db[db].bound_forms([metas[i] for i in idx],
                                             [states[i] for i in idx],
                                             [s_held[i] for i in idx], grammar)
            for i, g in zip(idx, got):
                out[i] = g
        return out

    def values(self, metas: list[QuestionMeta], states: list[State],
               wfs: list[Workflow], rolls: list, s_held: list[float]) -> np.ndarray:
        index: dict[str, list[int]] = {}
        for i, m in enumerate(metas):
            index.setdefault(m.db, []).append(i)
        out = np.zeros(len(metas))
        for db, idx in index.items():
            got = self.by_db[db].values([metas[i] for i in idx],
                                        [states[i] for i in idx],
                                        [wfs[i] for i in idx],
                                        [rolls[i] for i in idx],
                                        [s_held[i] for i in idx])
            out[idx] = np.asarray(got, dtype=float)
        return out
