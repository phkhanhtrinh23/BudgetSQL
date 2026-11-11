"""Wiring a workload, a world model and a set of methods into runs.

Every learned component that scores a question is fitted on the other budget
groups, so no question is planned with a model that has seen its schema, and
every setting a method tunes is chosen on the other groups too.  Fitted
components are cached on disk, because a sweep re-uses them many times.

A run covers the whole batch: each budget group has its own share of the
budget and its own meter, BudgetSQL plans the groups with one agent and one
digest (``BatchPlanner``), and so do the agent baselines.  When BudgetSQL's
turns are modelled, its result carries the modelled effect of the vector
(``budgetsql.eval.effects``).
"""
from __future__ import annotations

import hashlib
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..baselines import BASELINES, Oracle
from ..baselines.agents import AgentDryRuns, AgentNoWorldModel, run_agent_batch
from ..config import (PlannerConfig, Workload, budget_for_group,
                      use_workload_prices, WORKLOADS)
from ..features import QuestionMeta
from ..grammar import reference_library
from ..planner import BatchPlanner, BudgetSQL, Meter, PlanningAgent
from .effects import Effect, ModelledEffects, applies
from ..trace.corpus import TraceCorpus, split_dbs
from ..worldmodel.stacking import CrossFitStacker, StackedValues
from .pooled import PooledValues

CACHE_VERSION = 4


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
    #: The agent's modelled effects per budget vector; none
    #: leaves BudgetSQL at its replayed accuracy.
    effects: ModelledEffects = field(default_factory=ModelledEffects)
    #: Builds the agent whose turns are taken in live mode, for BudgetSQL and
    #: for the agent baselines alike; called with ``probe`` (may it probe).
    #: ``None`` uses the rule-based stand-ins (``PlanningAgent`` for BudgetSQL,
    #: ``BudgetTrackerAgent`` for the baselines); ``chat_agent_from_env`` is the
    #: language-model agent.
    agent_factory: object = None

    def __post_init__(self) -> None:
        use_workload_prices(self.workload)
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

    def pooled_group(self) -> BudgetGroup:
        """Every question under one budget."""
        groups = self.groups()
        return BudgetGroup("pooled", [d for g in groups for d in g.dbs],
                           [m for g in groups for m in g.metas])

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

    def thinned_values(self, group: BudgetGroup, cfg: PlannerConfig | None = None
                       ) -> StackedValues:
        """The same calibration from a corpus that traces one composition per
        training question rather than all of them."""
        cfg = cfg or self.cfg
        key = (f"thin-{CACHE_VERSION}-{group.name}-"
               f"{_hash(cfg.training_key(), self.inner_folds)}")
        if key in self._values:
            return self._values[key]

        def build() -> StackedValues:
            stacker = CrossFitStacker(cfg, self.inner_folds, self.library,
                                      self.corpus.grammar, thin=True)
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

    def pooled_values(self, cfg: PlannerConfig | None = None) -> PooledValues:
        cfg = cfg or self.cfg
        by_db = {}
        for group in self.groups():
            values = self.values(group, cfg)
            for db in group.dbs:
                by_db[db] = values
        return PooledValues(by_db, cfg)

    def pooled_planner(self, cfg: PlannerConfig | None = None, agent=None) -> BudgetSQL:
        cfg = cfg or self.cfg
        values = self.pooled_values(cfg).with_cfg(cfg)
        agent = agent if agent is not None else (
            PlanningAgent() if cfg.use_agent else None)
        return BudgetSQL(values, self.corpus, self.corpus.grammar, cfg, agent,
                         self.library)

    def planner(self, group: BudgetGroup, cfg: PlannerConfig | None = None,
                agent=None, thinned: bool = False) -> BudgetSQL:
        cfg = cfg or self.cfg
        source = self.thinned_values if thinned else self.values
        values = source(group, cfg).with_cfg(cfg)
        agent = agent if agent is not None else (
            PlanningAgent() if cfg.use_agent else None)
        return BudgetSQL(values, self.corpus, self.corpus.grammar, cfg, agent,
                         self.library)

    def effect(self, cfg: PlannerConfig, budget: str | None, vector=None,
               budget_scale: float = 1.0) -> Effect:
        """The modelled effect a BudgetSQL run under ``cfg`` carries."""
        if not applies(cfg):
            return Effect()
        if vector is None and budget_scale != 1.0:
            from ..config import BUDGETS
            vector = BUDGETS[budget] * budget_scale
        return self.effects.at(budget, vector)

    # ---- runs ----
    def order_metas(self, metas: list[QuestionMeta], run: int,
                    order: str = "random") -> list[QuestionMeta]:
        """The order the questions arrive in."""
        out = list(metas)
        random.Random((run + 1) * 7919).shuffle(out)
        rank = {"simple": 0, "moderate": 1, "challenging": 2}
        if order == "easy_first":
            out.sort(key=lambda m: rank.get(m.difficulty, 1))
        elif order == "hard_first":
            out.sort(key=lambda m: -rank.get(m.difficulty, 1))
        return out

    def run_method(self, name: str, budget: str, runs: int | None = None,
                   cfg: PlannerConfig | None = None, agent=None,
                   budget_scale: float = 1.0, order: str = "random",
                   vector=None, thinned: bool = False, agent_factory=None,
                   with_effects: bool = True) -> dict:
        """Run one method on the whole batch, over several question orders.

        Every budget group gets its share of the budget and its own meter.
        BudgetSQL and the agent baselines plan the groups together, with one
        agent whose turns each group pays in proportion to its questions."""
        cfg = cfg or self.cfg
        runs = runs or self.workload.runs
        per_run: list[dict[str, bool]] = []
        details: list[dict] = []
        groups = self.groups()
        for r in range(runs):
            items = []
            for group in groups:
                metas = self.order_metas(group.metas, r, order)
                if vector is None:
                    b = budget_for_group(budget, group.size) * budget_scale
                else:
                    b = np.asarray(vector, dtype=float) * float(group.size)
                items.append((metas, b, Meter(b, planning_cap=cfg.planning_cap)))
            if name == "budgetsql":
                planners = [self.planner(g, cfg, None, thinned) for g in groups]
                if agent_factory is not None:
                    agent = agent_factory()
                elif agent is None and cfg.use_agent:
                    agent = (self.agent_factory(True) if self.agent_factory is not None
                             else PlanningAgent())
                results = BatchPlanner(planners, cfg, agent).run(items)
            elif BASELINES.get(name) in (AgentNoWorldModel, AgentDryRuns):
                methods = [self.method(name, g, cfg).with_cfg(cfg) for g in groups]
                if self.agent_factory is not None:
                    # the same agent model and tools, without the world model
                    for m in methods:
                        m.agent_factory = lambda: self.agent_factory(True)
                results = run_agent_batch(methods, items)
            else:
                results = []
                for group, (metas, b, _) in zip(groups, items):
                    method = self.method(name, group, cfg).with_cfg(cfg)
                    model = getattr(method, "model", None)
                    if model is not None:
                        model.reset_online()
                    results.append(method.run(metas, b, Meter(b)))
            correct: dict[str, bool] = {}
            for group, res in zip(groups, results):
                correct.update(res.correct)
                details.append({"run": r, "group": group.name, "method": name,
                                "budget": budget, **res.as_dict()})
            per_run.append(correct)
        out = {"method": name, "budget": budget, "runs": per_run,
               "details": details, "effect": Effect().as_dict()}
        if name == "budgetsql" and with_effects:
            out["effect"] = self.effect(cfg, budget, vector, budget_scale).as_dict()
        return out

    def unit_of(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for group in self.groups():
            for m in group.metas:
                out[m.qid] = group.name
        return out
