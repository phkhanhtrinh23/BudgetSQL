"""Agent baselines: the same agent, tools and meter without the world model.

``AgentNoWorldModel`` keeps a budget tracker that reports what is left after
every call, and estimates success and cost from stage averages rather than per
question.  ``AgentDryRuns`` first runs each candidate workflow on a sample of
the batch, pays for the sample, and plans with the averages it measured.
``ProfiledPerRequest`` drops the batch allocation as well and plans each request
on its own share from the same averages.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from ..config import PlannerConfig
from ..planner.agent import BudgetTrackerAgent
from ..planner.execute import first_block, run_workflow
from ..planner.meter import Meter
from ..planner.planner import BudgetSQL, RunResult
from ..trace.corpus import TraceCorpus
from ..worldmodel.model import StageWorldModel
from ..worldmodel.stacking import DirectHead, StackedValues, ValueStack
from .base import Method
from .profiles import profile_library


def averages_only(cfg: PlannerConfig) -> PlannerConfig:
    """A world model stripped of its inputs predicts one average per stage and
    observation, which is what a planner has without one."""
    return cfg.with_(use_question_features=False, use_execution_features=False,
                     use_stacking=False)


@dataclass
class AgentNoWorldModel(Method):
    name: str = "llm_agent"
    agent_factory: object = BudgetTrackerAgent

    def prepare(self, corpus: TraceCorpus, train_dbs: list[str]) -> "AgentNoWorldModel":
        super().prepare(corpus, train_dbs)
        self._cfg = averages_only(self.cfg)
        self.model = StageWorldModel(self._cfg).fit(corpus, train_dbs)
        self._values = StackedValues(self.model, DirectHead(), ValueStack(), self._cfg)
        return self

    def _planner(self) -> BudgetSQL:
        return BudgetSQL(self._values, self.corpus, self.grammar, self._cfg,
                         agent=self.agent_factory(), library=self.library)

    def run(self, metas, budget, meter=None) -> RunResult:
        return self._planner().run(metas, np.asarray(budget, dtype=float), meter)


@dataclass
class AgentDryRuns(Method):
    """Pays for a sample of the batch to measure what each candidate costs.

    It runs the opening stage of a few candidates on a sample of the questions,
    charges the sample to the same budget, and corrects its cost estimates with
    what it measured.  The sample says nothing about which questions need more,
    only what the stages cost on this workload.
    """

    name: str = "llm_agent_dry_runs"
    dry_share: float = 0.05
    probe_workflows: int = 4
    seed: int = 0

    def prepare(self, corpus, train_dbs) -> "AgentDryRuns":
        super().prepare(corpus, train_dbs)
        self._cfg = averages_only(self.cfg)
        self.model = StageWorldModel(self._cfg).fit(corpus, train_dbs)
        return self

    def run(self, metas, budget, meter=None) -> RunResult:
        from ..planner.execute import ExecutionContext

        meter = meter or Meter(np.asarray(budget, dtype=float),
                               planning_cap=self.cfg.planning_cap)
        model = self.model.clone(self._cfg)
        rng = random.Random(self.seed)
        sample = list(metas)
        rng.shuffle(sample)
        sample = sample[: max(1, int(round(self.dry_share * len(sample))))]

        # the dry runs are charged to the budget, and what they cost is fed back
        # into the cost estimates the plan is then made with
        ctx = ExecutionContext(self.corpus, meter, self._cfg, self.grammar, model)
        for wf in list(self.library.values())[: self.probe_workflows]:
            for meta in sample:
                run_workflow(ctx, self._records([meta])[meta.qid], first_block(wf))

        values = StackedValues(model, DirectHead(), ValueStack(), self._cfg)
        planner = BudgetSQL(values, self.corpus, self.grammar, self._cfg,
                            agent=BudgetTrackerAgent(), library=self.library)
        return planner.run(metas, np.asarray(budget, dtype=float), meter)


@dataclass
class ProfiledPerRequest(Method):
    """Per-request planning on profiled averages, with no batch allocation."""

    name: str = "profiled_per_request"
    profiles: dict = field(default_factory=dict)

    def prepare(self, corpus, train_dbs) -> "ProfiledPerRequest":
        super().prepare(corpus, train_dbs)
        self.profiles = profile_library(corpus, train_dbs, self.library)
        return self

    def run(self, metas, budget, meter=None) -> RunResult:
        meter = meter or Meter(np.asarray(budget, dtype=float))
        records = self._records(metas)
        per_q = np.asarray(budget, dtype=float) / max(1, len(metas))
        ctx = self._ctx(meter)
        affordable = [n for n, p in self.profiles.items()
                      if p.n and np.all(p.cost <= per_q)]
        name = max(affordable, key=lambda n: self.profiles[n].acc, default=None)
        if name is None:
            name = min(self.profiles, key=lambda n: float(np.sum(self.profiles[n].cost)))
        wf = self.library[name]
        for rec in records.values():
            run_workflow(ctx, rec, wf)
        return self._result(records, meter, self.profiles[name].acc)
