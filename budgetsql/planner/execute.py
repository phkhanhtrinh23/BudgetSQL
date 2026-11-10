"""Running a workflow's stages on a question under the meter.

Shared by the planner and by every method it is compared with, so all of them
launch, cap and charge calls the same way (Prop. 1): a stage runs only if its
worst case fits what is left, its completion limit and timeouts are whatever the
remainder can pay for, and a call that reaches a limit --- completion tokens,
the LLM timeout or the SQL timeout --- is cancelled: charged what it consumed
up to the limit and returning nothing, which leaves the question with the answer
it already held.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import PlannerConfig
from ..grammar import ANSWER_PRODUCING, REPLACING, Grammar, Workflow
from ..trace.corpus import TraceCorpus
from .meter import Meter


@dataclass
class ExecutionContext:
    corpus: TraceCorpus
    meter: Meter
    cfg: PlannerConfig
    grammar: Grammar
    model: object | None = None      # a world model, when one is available
    #: Apply BudgetSQL's Select rule to a replacing stage's new answer.  A
    #: baseline that reimplements another decision rule keeps the new answer,
    #: as that rule does, and reads the world model only for its scores.
    select: bool = True


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


def pace(meter: Meter, expected: np.ndarray, later: int, cheap: np.ndarray) -> bool:
    """The pacing rule of the baselines (Sec. VII-A): a question may run a step
    of expected cost ``expected`` only if the remaining budget then still
    covers the cheapest feasible call, of expected cost ``cheap``, for each of
    the ``later`` questions; otherwise it is downgraded to that call."""
    left = meter.remaining - np.asarray(expected, dtype=float)
    return bool(np.all(left >= max(0, int(later)) * np.asarray(cheap, dtype=float) - 1e-12))


def known_prompt(rec, stage, out) -> float:
    """The prompt tokens of one sample of a call, which are known before it is
    sent.  A replayed record carries them (its tokens less its completion);
    otherwise the question's prompt at the stage's context level stands in."""
    ctx_level = stage.ctx if stage.kind == "generate" else rec.state.ctx
    fallback = float(rec.meta.prompt_at(ctx_level))
    if out is None or not stage.llm_calls:
        return fallback
    n = max(1, stage.samples)
    prompt = float(out.cost[1]) / n - float(out.completion_tokens)
    return prompt if prompt > 0 else fallback


def run_workflow(ctx: ExecutionContext, rec, wf: Workflow) -> StepReport:
    """Execute ``wf`` from the question's current state, updating the record.

    A refused or cancelled call leaves the question with the answer it already
    held; whether it may try something cheaper instead is the caller's call.
    """
    report = StepReport()
    node = wf
    while not node.is_stop:
        stage = node.stage
        out = ctx.corpus.outcome(rec.meta, rec.state, stage)
        if out is None:
            rec.closed = True
            report.exhausted = True
            return report

        lim = ctx.meter.limits_for(stage, known_prompt(rec, stage, out),
                                   ctx.cfg.limits)
        if lim is None:
            ctx.meter.stats.refused += 1
            report.refused = True
            return report
        worst = lim.worst_case

        resolved = None
        mu = np.zeros(3)
        raw_mu = mu
        if ctx.model is not None:
            pred = ctx.model.predict(rec.meta, rec.state, stage)
            resolved = pred.resolve(rec.s_held)
            mu = pred.mu
            raw_mu = ctx.model.predict_raw(rec.meta, rec.state, stage).mu

        # a call that reaches any of its limits is cancelled at that limit
        cut = bool((stage.llm_calls > 0 and out.completion_tokens > lim.completion)
                   or np.any(out.cost > worst + 1e-12))
        cost = np.minimum(worst, out.cost) if cut else out.cost
        charged = ctx.meter.charge(cost, worst, label=stage.id, cancelled=cut,
                                   tier=stage.tier)
        if charged is None:
            report.refused = True
            return report

        rec.spend = rec.spend + charged
        rec.predicted = rec.predicted + mu
        rec.n_stages += 1
        report.stages += 1
        if ctx.model is not None:
            # Eq. 5 compares realised costs with the uncorrected predictions,
            # for every stage that completes, SQL included
            ctx.model.observe_cost(rec.db, stage, raw_mu, out.cost, completed=not cut)
        if cut:
            report.cancelled = True
            return report

        j = 0
        if resolved is not None:
            try:
                j = resolved.index(out.obs)
            except ValueError:
                j = 0
            if hasattr(rec, "observe"):
                rec.observe(stage, resolved, j)
        keep = True
        if stage.kind in REPLACING and rec.state.has_answer:
            if ctx.select and resolved is not None:
                keep = bool(resolved.keep_new[j])          # Select (Sec. IV-B)
        if stage.kind in ANSWER_PRODUCING and (keep or not rec.state.has_answer):
            rec.held_correct = bool(out.answer_correct)
        if resolved is not None:
            rec.s_held = resolved.s_given(j, keep)
        # a rejected answer leaves the earlier one held, and the branch says so
        rec.state = rec.state.append(stage, out.obs, keep=keep, values=out)
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
