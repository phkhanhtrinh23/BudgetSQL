from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from ..config import PlannerConfig
from ..planner.agent import BudgetTrackerAgent
from ..planner.batch import BatchPlanner
from ..planner.execute import first_block, run_workflow
from ..planner.meter import Meter
from ..planner.planner import BudgetSQL, RunResult
from ..trace.corpus import TraceCorpus
from ..worldmodel.model import StageWorldModel
from ..worldmodel.stacking import DirectHead, StackedValues, ValueStack
from .base import Method
from .adaptive import VineLMStyle


def averages_only(cfg: PlannerConfig) -> PlannerConfig:
    return cfg.with_(use_question_features=False, use_execution_features=False,
                     use_stacking=False, use_select=False, agent_mode="live",
                     pacing=True)


def run_agent_batch(methods: list, groups: list[tuple[list, np.ndarray, Meter]]
                    ) -> list[RunResult]:
    planners = [m.planner_for(metas, budget, meter)
                for m, (metas, budget, meter) in zip(methods, groups)]
    batch = BatchPlanner(planners, planners[0].cfg, methods[0].agent_factory())
    return batch.run(groups)


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

    def with_cfg(self, cfg: PlannerConfig) -> "AgentNoWorldModel":
        self.cfg, self._cfg = cfg, averages_only(cfg)
        return self

    def planner_for(self, metas, budget, meter) -> BudgetSQL:
        return BudgetSQL(self._values, self.corpus, self.grammar, self._cfg,
                         agent=self.agent_factory(), library=self.library)

    def run(self, metas, budget, meter=None) -> RunResult:
        budget = np.asarray(budget, dtype=float)
        meter = meter or Meter(budget, planning_cap=self.cfg.planning_cap)
        return run_agent_batch([self], [(metas, budget, meter)])[0]


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
    agent_factory: object = BudgetTrackerAgent

    def prepare(self, corpus, train_dbs) -> "AgentDryRuns":
        super().prepare(corpus, train_dbs)
        self._cfg = averages_only(self.cfg)
        self.model = StageWorldModel(self._cfg).fit(corpus, train_dbs)
        return self

    def with_cfg(self, cfg: PlannerConfig) -> "AgentDryRuns":
        self.cfg, self._cfg = cfg, averages_only(cfg)
        return self

    def planner_for(self, metas, budget, meter) -> BudgetSQL:
        from ..planner.execute import ExecutionContext

        model = self.model.clone(self._cfg.with_(online_cost_updates=True))
        rng = random.Random(self.seed)
        sample = list(metas)
        rng.shuffle(sample)
        sample = sample[: max(1, int(round(self.dry_share * len(sample))))]
        ctx = ExecutionContext(self.corpus, meter, self._cfg, self.grammar, model,
                               select=self._cfg.use_select)
        for wf in list(self.library.values())[: self.probe_workflows]:
            for meta in sample:
                run_workflow(ctx, self._records([meta])[meta.qid], first_block(wf))
        values = StackedValues(model, DirectHead(), ValueStack(), self._cfg)
        return BudgetSQL(values, self.corpus, self.grammar, self._cfg,
                         agent=self.agent_factory(), library=self.library)

    def run(self, metas, budget, meter=None) -> RunResult:
        budget = np.asarray(budget, dtype=float)
        meter = meter or Meter(budget, planning_cap=self.cfg.planning_cap)
        return run_agent_batch([self], [(metas, budget, meter)])[0]


@dataclass
class ProfiledPerRequest(VineLMStyle):
    """The ablation arm with the world model and the allocation replaced by
    per-request planning on profiled averages.  Each request gets its own
    share of the budget, opens with the composition (any that begins with a
    single call) whose profiled accuracy is highest among those that fit, and,
    after observing its first stage, takes the continuation whose profiled
    accuracy given that observation is highest within what is left of its
    share."""

    name: str = "profiled_per_request"

    def _opening_library(self) -> dict:
        from ..grammar import single_call_first
        return {n: wf for n, wf in self.library.items() if single_call_first(wf)}
