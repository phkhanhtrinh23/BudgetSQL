"""A common shell for the methods BudgetSQL is compared with.

Every method gets the same stage grammar where its design allows, the same
executors, the same meter and the same pacing rule: a question is
downgraded to the cheapest feasible call when the remaining budget could not
otherwise cover the questions still to come.  For a question that already holds
an answer, the cheapest feasible step is to keep it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import DIMS, PlannerConfig, zeros
from ..features import QuestionMeta
from ..grammar import Grammar, State, Workflow, branch, reference_library
from ..planner.execute import ExecutionContext, first_block, pace
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
    #: Questions of the batch still to come after this group, when several
    #: groups share one meter (the pooled budget); the pacing rule covers them.
    batch_later: int = 0

    def prepare(self, corpus: TraceCorpus, train_dbs: list[str]) -> "Method":
        self.corpus = corpus
        self.grammar = self.grammar or corpus.grammar
        self.train_dbs = list(train_dbs)
        self.cheap_cost = self._profile_cheapest(corpus, train_dbs)
        return self

    def _profile_cheapest(self, corpus: TraceCorpus, train_dbs: list[str]) -> np.ndarray:
        """The mean cost, on other databases, of the cheapest first call."""
        wf = self._cheapest(State())
        metas = [m for d in train_dbs for m in corpus.questions_of(d)]
        if wf is None or not metas:
            return zeros()
        costs = [corpus.run(m, State(), first_block(wf))[2] for m in metas]
        return np.mean(costs, axis=0)

    def with_cfg(self, cfg: PlannerConfig) -> "Method":
        """Read the planning settings of ``cfg``; a fitted method is cached by
        what its fit depends on, so a run may change everything else."""
        self.cfg = cfg
        return self

    # ---- helpers ----
    def _records(self, metas: list[QuestionMeta]) -> dict[str, QuestionRecord]:
        return {m.qid: QuestionRecord(m) for m in metas}

    def _ctx(self, meter: Meter) -> ExecutionContext:
        """The baselines' own decision rules keep a replacing stage's new
        answer; a world model, where a method has one, only scores."""
        return ExecutionContext(self.corpus, meter, self.cfg, self.grammar, self.model,
                                select=False)

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
            n_questions=len(records),
            tier_money={t: float(v[0]) for t, v in stats.tier_spend.items()})

    def _paced(self, meter: Meter, expected: np.ndarray, later: int) -> bool:
        """The pacing rule: whether a step of expected cost ``expected`` leaves
        the cheapest call for each of the ``later`` questions."""
        cheap = getattr(self, "cheap_cost", None)
        if cheap is None:
            cheap = zeros()
        return pace(meter, expected, later + self.batch_later, cheap)

    def _paced_opening(self, meter: Meter, rec, wf: Workflow, expected: np.ndarray,
                       later: int) -> Workflow:
        """The workflow a question opens with under the pacing rule: its own,
        or the cheapest feasible call when its own would not leave enough."""
        if self._paced(meter, expected, later):
            return wf
        cheap = self._cheapest(rec.state)
        return cheap if cheap is not None else wf

    def _cheapest(self, state: State) -> Workflow | None:
        """The cheapest call the grammar still allows from a state."""
        actions = [a for a in self.grammar.actions(state) if a.llm_calls]
        if not actions:
            return None
        a = min(actions, key=lambda s: (s.samples, {"S": 0, "M": 1, "L": 2}.get(s.tier, 3)))
        return first_block(branch(a, {o: None for o in a.observations()}))

    # ---- interface ----
    def run(self, metas: list[QuestionMeta], budget: np.ndarray,
            meter: Meter | None = None) -> RunResult:
        raise NotImplementedError
