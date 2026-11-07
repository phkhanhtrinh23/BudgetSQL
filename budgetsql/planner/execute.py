"""Running a workflow's stages on a question under the meter.

Shared by the planner and by every method it is compared with, so all of them
launch, cap and charge calls the same way: a stage runs only if its worst case
fits what is left, its completion limit is whatever the remainder can pay for,
and a call that would exceed that limit is cancelled --- charged what it
consumed and returning nothing, which leaves the question with the answer it
already held.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import PlannerConfig
from ..grammar import ANSWER_PRODUCING, REPLACING, Grammar, Workflow
from ..trace.corpus import TraceCorpus
from .meter import Meter

MIN_USEFUL_COMPLETION = 16


@dataclass
class ExecutionContext:
    corpus: TraceCorpus
    meter: Meter
    cfg: PlannerConfig
    grammar: Grammar
    model: object | None = None      # a world model, when one is available


@dataclass
class StepReport:
    """What running a workflow did: how many stages it got through, and whether
    the meter refused a call, a call was cancelled, or the trace held no record
    of the branch."""

    stages: int = 0
    cancelled: bool = False
    refused: bool = False
    exhausted: bool = False

    @property
    def made_progress(self) -> bool:
        return self.stages > 0 and not (self.cancelled or self.refused)


def run_workflow(ctx: ExecutionContext, rec, wf: Workflow) -> StepReport:
    """Execute ``wf`` from the question's current state, updating the record.

    A refused or cancelled call leaves the question with the answer it already
    held; whether it may try something cheaper instead is the caller's call.
    """
    report = StepReport()
    node = wf
    while not node.is_stop:
        stage = node.stage
        ctx_level = stage.ctx if stage.kind == "generate" else rec.state.ctx
        prompt = float(rec.meta.prompt_at(ctx_level))

        resolved = None
        mu = np.zeros(3)
        if ctx.model is not None:
            pred = ctx.model.predict(rec.meta, rec.state, stage)
            resolved = pred.resolve(rec.s_held)
            mu = pred.mu

        cap = stage.cap
        if stage.llm_calls:
            cap = ctx.meter.affordable_completion(stage, prompt)
            if cap < MIN_USEFUL_COMPLETION:
                report.refused = True
                return report
        worst = ctx.meter.worst_case_of(stage, prompt, ctx.cfg.limits.timeout_s, cap)

        out = ctx.corpus.outcome(rec.meta, rec.state, stage)
        if out is None:
            rec.closed = True
            report.exhausted = True
            return report

        cut = stage.llm_calls > 0 and out.completion_tokens > cap
        cost = np.minimum(worst, out.cost) if cut else out.cost
        charged = ctx.meter.charge(cost, worst, label=stage.id)
        if charged is None:
            report.refused = True
            return report

        rec.spend = rec.spend + charged
        rec.predicted = rec.predicted + mu
        rec.n_stages += 1
        report.stages += 1
        if ctx.model is not None and stage.llm_calls:
            ctx.model.observe_cost(rec.db, stage, mu, out.cost, completed=not cut)
        if cut:
            report.cancelled = True
            return report

        j = 0
        if resolved is not None:
            try:
                j = resolved.index(out.obs)
            except ValueError:
                j = 0
        if stage.kind in ANSWER_PRODUCING:
            keep = True
            if stage.kind in REPLACING and resolved is not None:
                keep = bool(resolved.keep_new[j])
            if keep or not rec.state.has_answer:
                rec.held_correct = bool(out.answer_correct)
        if resolved is not None:
            rec.s_held = float(resolved.s_after[j])
        rec.state = rec.state.append(stage, out.obs)
        node = node.child(out.obs)

    if not ctx.grammar.actions(rec.state):
        rec.closed = True
    return report


def first_block(wf: Workflow) -> Workflow:
    """A workflow's opening stage together with the execution that reveals what
    it returned."""
    from ..grammar import Workflow as W
    if wf.is_stop:
        return wf
    stage = wf.stage
    if stage.kind in ("context", "generate"):
        return W(stage, {o: first_block(wf.child(o)) for o in stage.observations()},
                 wf.name)
    return W(stage, {o: None for o in stage.observations()}, wf.name)
