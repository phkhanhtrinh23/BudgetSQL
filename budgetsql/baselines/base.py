"""A common shell for the methods BudgetSQL is compared with.

Every method gets the same stage grammar where its design allows, the same
executors, the same meter and the same pacing rule: a question's workflow is
downgraded to the cheapest feasible call when the remaining budget could not
otherwise cover the questions still to come.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import DIMS, PlannerConfig
from ..features import QuestionMeta
from ..grammar import Grammar, State, Workflow, branch, reference_library
from ..planner.execute import ExecutionContext
from ..planner.meter import Meter
from ..planner.planner import QuestionRecord, RunResult
from ..trace.corpus import TraceCorpus


@dataclass
class Method:
    """Base class: fit whatever the method needs on other databases, then run a
    budget group under one meter."""

    name: str = "method"
    corpus: TraceCorpus | None = None
    cfg: PlannerConfig = field(default_factory=PlannerConfig)
    grammar: Grammar | None = None
    library: dict[str, Workflow] = field(default_factory=reference_library)
    model: object | None = None

    def prepare(self, corpus: TraceCorpus, train_dbs: list[str]) -> "Method":
        self.corpus = corpus
        self.grammar = self.grammar or corpus.grammar
        self.train_dbs = list(train_dbs)
        return self

    # ---- helpers ----
    def _records(self, metas: list[QuestionMeta]) -> dict[str, QuestionRecord]:
        return {m.qid: QuestionRecord(m) for m in metas}

    def _ctx(self, meter: Meter) -> ExecutionContext:
        return ExecutionContext(self.corpus, meter, self.cfg, self.grammar, self.model)

    def _result(self, records: dict[str, QuestionRecord], meter: Meter,
                predicted: float = 0.0, duals: list | None = None) -> RunResult:
        correct = {q: r.held_correct for q, r in records.items()}
        stats = meter.stats
        share = {k: float(p / b) if b else 0.0
                 for k, p, b in zip(DIMS, stats.planning_spend, meter.budget)}
        return RunResult(
            accuracy=float(np.mean([1.0 if v else 0.0 for v in correct.values()]))
            if correct else 0.0,
            correct=correct, spend=stats.spend.copy(), budget=meter.budget.copy(),
            meter=meter.snapshot(), planning_share=share,
            duals_history=duals or [], predicted_accuracy=predicted,
            cancelled=stats.cancelled, refused=stats.refused,
            n_questions=len(records))

    def _pace(self, meter: Meter, remaining_questions: int) -> np.ndarray:
        """What one question may still use if the rest are to get a cheap call."""
        if remaining_questions <= 0:
            return meter.remaining
        return meter.remaining / float(remaining_questions)

    def _cheapest(self, state: State) -> Workflow | None:
        """The cheapest call the grammar still allows from a state."""
        actions = [a for a in self.grammar.actions(state) if a.llm_calls]
        if not actions:
            return None
        a = min(actions, key=lambda s: (s.samples, {"S": 0, "M": 1, "L": 2}.get(s.tier, 3)))
        return branch(a, {o: None for o in a.observations()})

    # ---- interface ----
    def run(self, metas: list[QuestionMeta], budget: np.ndarray,
            meter: Meter | None = None) -> RunResult:
        raise NotImplementedError
