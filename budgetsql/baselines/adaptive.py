"""Methods that observe a first answer and then decide what to spend next.

``CascadeRouting`` escalates while the estimated gain per normalised cost
exceeds a price that tracks the remaining budget.  ``UABStyle`` answers every
question with the small tier and spends what is left on the answers a learned
scorer is least confident about.  ``VineLMStyle`` plans each request on its own
share of the budget from profiled success rates per workflow and observation, so
slack cannot move from one question to another.  All three run under the meter
with the pacing rule of ``Method``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..grammar import State, Workflow, parse_chain
from ..planner.execute import first_block, run_workflow
from ..planner.meter import Meter
from ..planner.planner import RunResult
from ..trace.corpus import TraceCorpus
from .base import Method
from .profiles import DEFAULT_CONTINUATIONS, Profile, lookup, profile_continuations, profile_library

TIER_ORDER = ("S", "M", "L")


def _openers(library: dict[str, Workflow]) -> dict[str, Workflow]:
    return {t: library[t] for t in TIER_ORDER if t in library}


def _ladder(tier: str, obs: str) -> list[str]:
    """The escalation steps a cascade may climb after ``tier`` observed ``obs``,
    each given as the chain run so far: a failed or empty answer is repaired by
    the large tier, a valid one escalates through the tiers above it."""
    if obs.split(".")[0] in ("error", "empty"):
        return ["rep:L"] if tier else []
    above = [t for t in TIER_ORDER if TIER_ORDER.index(t) > TIER_ORDER.index(tier)]
    return [",".join(f"esc:{t}" for t in above[:i + 1]) for i in range(len(above))]


#: The chains a cascade's profiles cover: every rung of every ladder.
CASCADE_CHAINS = ("stop", "rep:L", "esc:M", "esc:L", "esc:M,esc:L")


@dataclass
class CascadeRouting(Method):
    name: str = "cascade_routing"
    first_tier: str = "M"
    profiles: dict[str, Profile] = field(default_factory=dict)
    table: dict = field(default_factory=dict)

    def prepare(self, corpus: TraceCorpus, train_dbs: list[str]) -> "CascadeRouting":
        super().prepare(corpus, train_dbs)
        from ..worldmodel.model import StageWorldModel
        self.model = StageWorldModel(self.cfg).fit(corpus, train_dbs)
        self.profiles = profile_library(corpus, train_dbs, _openers(self.library))
        self.table = profile_continuations(corpus, train_dbs, _openers(self.library),
                                           CASCADE_CHAINS)
        return self

    def _choose_tier(self, per_q: np.ndarray) -> str:
        """The first tier is chosen on other databases: the most accurate one
        whose mean cost leaves room for an escalation."""
        best, best_acc = "S", -1.0
        for t, p in self.profiles.items():
            if p.n and np.all(p.cost <= per_q * 0.6) and p.acc > best_acc:
                best, best_acc = t, p.acc
        return best

    def run(self, metas, budget, meter=None) -> RunResult:
        meter = meter or Meter(budget)
        records = self._records(metas)
        per_q = np.asarray(budget, dtype=float) / max(1, len(metas))
        tier = self._choose_tier(per_q)
        self.first_tier = tier
        ctx = self._ctx(meter)
        opener = first_block(self.library[tier])
        scale = np.maximum(per_q, 1e-12)

        left = len(records)
        for rec in records.values():
            left -= 1
            run_workflow(ctx, rec, self._paced_opening(
                meter, rec, opener, self.profiles[tier].cost, left))
            if rec.closed or rec.state.answer_stage is None \
                    or rec.state.answer_stage.tier != tier:
                continue
            obs = rec.state.answer_obs or "-"
            spent = np.zeros_like(per_q)
            # escalate while the estimated gain per normalised cost exceeds a
            # price that tracks what is left of the budget
            for chain in _ladder(tier, obs):
                prof = lookup(self.table, tier, obs, chain)
                if prof is None:
                    break
                step_cost = np.maximum(prof.cost - spent, 0.0)
                gain = max(0.0, prof.acc - rec.s_held)
                norm = float(np.max(step_cost / scale))
                if norm <= 0:
                    break
                frac_left = float(np.min(meter.remaining / np.maximum(meter.budget, 1e-12)))
                price = (1.0 - frac_left) + 0.05
                if gain / norm <= price or not self._paced(meter, step_cost, left):
                    break
                step = parse_chain(chain.split(",")[-1])
                if not self.grammar.is_wellformed(step, rec.state):
                    break
                report = run_workflow(ctx, rec, step)
                if not report.made_progress:
                    break
                spent = prof.cost
        return self._result(records, meter)


@dataclass
class UABStyle(Method):
    """A cheap answer for every question, then the rest of the budget on the
    least confident answers."""

    name: str = "uab_style"
    first_tier: str = "S"
    escalation: str = "esc:L"

    def prepare(self, corpus, train_dbs) -> "UABStyle":
        super().prepare(corpus, train_dbs)
        from ..worldmodel.model import StageWorldModel
        self.model = StageWorldModel(self.cfg).fit(corpus, train_dbs)
        return self

    def run(self, metas, budget, meter=None) -> RunResult:
        meter = meter or Meter(budget)
        records = self._records(metas)
        ctx = self._ctx(meter)
        opener = first_block(self.library[self.first_tier])
        left = len(records)
        for rec in records.values():
            left -= 1
            expected = self.model.rollout(rec.meta, State(), opener, 0.0).cost
            run_workflow(ctx, rec, self._paced_opening(meter, rec, opener, expected, left))
        ranked = sorted((r for r in records.values() if not r.closed),
                        key=lambda r: r.s_held)
        chain = parse_chain(self.escalation)
        for rec in ranked:
            if not self.grammar.is_wellformed(chain, rec.state):
                continue
            # every question of this group holds an answer; pacing still covers
            # the batch's later questions when groups share a meter
            expected = self.model.rollout(rec.meta, rec.state, chain, rec.s_held).cost
            if not self._paced(meter, expected, 0):
                break
            before = meter.stats.launched
            run_workflow(ctx, rec, chain)
            if meter.stats.refused and meter.stats.launched == before:
                break
        return self._result(records, meter)


@dataclass
class VineLMStyle(Method):
    """Per-request planning: each question gets its share of the budget and
    spends it on the continuation the profiles rate best."""

    name: str = "vinelm_style"
    table: dict = field(default_factory=dict)
    profiles: dict[str, Profile] = field(default_factory=dict)
    specs: tuple[str, ...] = DEFAULT_CONTINUATIONS

    def _opening_library(self) -> dict[str, Workflow]:
        """The workflows a request may open with."""
        return _openers(self.library)

    def prepare(self, corpus, train_dbs) -> "VineLMStyle":
        super().prepare(corpus, train_dbs)
        openers = self._opening_library()
        self.profiles = profile_library(corpus, train_dbs, openers)
        self.table = profile_continuations(corpus, train_dbs, openers, self.specs)
        return self

    def run(self, metas, budget, meter=None) -> RunResult:
        meter = meter or Meter(budget)
        records = self._records(metas)
        per_q = np.asarray(budget, dtype=float) / max(1, len(metas))
        ctx = self._ctx(meter)

        opener_name = max(
            (n for n, p in self.profiles.items() if p.n and np.all(p.cost <= per_q)),
            key=lambda n: self.profiles[n].acc, default="S")
        opener = first_block(self.library[opener_name])

        left = len(records)
        for rec in records.values():
            left -= 1
            run_workflow(ctx, rec, self._paced_opening(
                meter, rec, opener, self.profiles[opener_name].cost, left))
            if rec.closed:
                continue
            obs = rec.state.answer_obs or "-"
            room = per_q - rec.spend
            best, best_val = None, lookup(self.table, opener_name, obs, "stop")
            base = best_val.acc if best_val else rec.s_held
            for spec in self.specs:
                if spec == "stop":
                    continue
                prof = lookup(self.table, opener_name, obs, spec)
                if prof is None or np.any(prof.cost > room):
                    continue
                if not self._paced(meter, prof.cost, left):
                    continue
                if prof.acc > base:
                    base, best = prof.acc, spec
            if best is None:
                continue
            chain = parse_chain(best)
            if self.grammar.is_wellformed(chain, rec.state):
                run_workflow(ctx, rec, chain)
        return self._result(records, meter)
