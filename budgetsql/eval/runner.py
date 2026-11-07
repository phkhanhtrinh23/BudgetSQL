"""Wiring a workload, a world model and a set of methods into runs.

Every learned component that scores a question is fitted on the other budget
groups, so no question is planned with a model that has seen its schema, and
every setting a method tunes is chosen on the other groups too.  Fitted
components are cached on disk, because a sweep re-uses them many times.
"""
from __future__ import annotations

import hashlib
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path


from ..baselines import BASELINES, Oracle
from ..config import (PlannerConfig, Workload, budget_for_group,
                      WORKLOADS)
from ..features import QuestionMeta
from ..grammar import reference_library
from ..planner import BudgetSQL, Meter, PlanningAgent
from ..trace.corpus import TraceCorpus, split_dbs
from ..worldmodel.stacking import CrossFitStacker, StackedValues

CACHE_VERSION = 2


def _hash(*parts) -> str:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode())
    return h.hexdigest()


@dataclass
class BudgetGroup:
    """The questions that share one budget vector."""

    name: str
    dbs: list[str]
    metas: list[QuestionMeta]

    @property
    def size(self) -> int:
        return len(self.metas)


@dataclass
class Workbench:
    """A workload with the components the methods need."""

    corpus: TraceCorpus
    cfg: PlannerConfig = field(default_factory=PlannerConfig)
    workload: Workload = field(default_factory=lambda: WORKLOADS["bird"])
    cache_dir: Path | None = None
    library: dict = field(default_factory=reference_library)
    inner_folds: int = 4

    def __post_init__(self) -> None:
        self._values: dict[str, StackedValues] = {}
        self._methods: dict[tuple[str, str], object] = {}
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- budget groups ----
    def groups(self) -> list[BudgetGroup]:
        if self.workload.budget_groups == "fold":
            folds = split_dbs(self.corpus.dbs, self.workload.folds, seed=0)
            return [BudgetGroup(f"fold{i}", dbs,
                                [m for d in dbs for m in self.corpus.questions_of(d)])
                    for i, dbs in enumerate(folds)]
        return [BudgetGroup(db, [db], self.corpus.questions_of(db))
                for db in self.corpus.dbs]

    def train_dbs(self, group: BudgetGroup) -> list[str]:
        return [d for d in self.corpus.dbs if d not in group.dbs]

    # ---- cached components ----
    def _cached(self, key: str, build):
        if self.cache_dir is None:
            return build()
        path = self.cache_dir / f"{key}.pkl"
        if path.exists():
            try:
                with open(path, "rb") as fh:
                    return pickle.load(fh)
            except Exception:
                pass
        obj = build()
        try:
            with open(path, "wb") as fh:
                pickle.dump(obj, fh)
        except Exception:
            pass
        return obj

    def values(self, group: BudgetGroup, cfg: PlannerConfig | None = None
               ) -> StackedValues:
        cfg = cfg or self.cfg
        key = (f"values-{CACHE_VERSION}-{group.name}-"
               f"{_hash(cfg.training_key(), self.inner_folds)}")
        if key in self._values:
            return self._values[key]

        def build() -> StackedValues:
            stacker = CrossFitStacker(cfg, self.inner_folds, self.library,
                                      self.corpus.grammar)
            return stacker.fit(self.corpus, group.dbs)

        out = self._cached(key, build)
        self._values[key] = out
        return out

    def method(self, name: str, group: BudgetGroup, cfg: PlannerConfig | None = None):
        cfg = cfg or self.cfg
        key = (name, group.name + _hash(cfg.training_key()))
        if key in self._methods:
            return self._methods[key]
        cls = BASELINES.get(name) or {"oracle": Oracle}.get(name)
        if cls is None:
            raise KeyError(f"unknown method {name!r}")

        def build():
            m = cls(name=name, cfg=cfg, library=self.library)
            return m.prepare(self.corpus, self.train_dbs(group))

        obj = self._cached(f"method-{CACHE_VERSION}-{name}-{group.name}-"
                           f"{_hash(cfg.training_key())}", build)
        obj.corpus = self.corpus
        obj.grammar = obj.grammar or self.corpus.grammar
        self._methods[key] = obj
        return obj

    def planner(self, group: BudgetGroup, cfg: PlannerConfig | None = None,
                agent=None) -> BudgetSQL:
        cfg = cfg or self.cfg
        values = self.values(group, cfg).with_cfg(cfg)
        agent = agent if agent is not None else (
            PlanningAgent() if cfg.use_agent else None)
        return BudgetSQL(values, self.corpus, self.corpus.grammar, cfg, agent,
                         self.library)

    # ---- runs ----
    def run_method(self, name: str, budget: str, runs: int | None = None,
                   cfg: PlannerConfig | None = None, agent=None,
                   budget_scale: float = 1.0) -> dict:
        """Run one method on every budget group, over several question orders."""
        cfg = cfg or self.cfg
        runs = runs or self.workload.runs
        per_run: list[dict[str, bool]] = []
        details: list[dict] = []
        for r in range(runs):
            correct: dict[str, bool] = {}
            for group in self.groups():
                metas = list(group.metas)
                random.Random((r + 1) * 7919).shuffle(metas)
                b = budget_for_group(budget, group.size) * budget_scale
                meter = Meter(b, planning_cap=cfg.planning_cap if name == "budgetsql" else None)
                if name == "budgetsql":
                    res = self.planner(group, cfg, agent).run(metas, b, meter)
                else:
                    method = self.method(name, group, cfg)
                    model = getattr(method, "model", None)
                    if model is not None:
                        model.reset_online()   # a cached fit starts each run fresh
                    res = method.run(metas, b, meter)
                correct.update(res.correct)
                details.append({"run": r, "group": group.name, "method": name,
                                "budget": budget, **res.as_dict()})
            per_run.append(correct)
        return {"method": name, "budget": budget, "runs": per_run,
                "details": details}

    def unit_of(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for group in self.groups():
            for m in group.metas:
                out[m.qid] = group.name
        return out
