"""Methods that observe a first answer and then decide what to spend next.

``CascadeRouting`` escalates while the estimated gain per normalised cost
exceeds a price that tracks the remaining budget.  ``UABStyle`` answers every
question with the small tier and spends what is left on the answers a learned
scorer is least confident about.  ``VineLMStyle`` plans each request on its own
share of the budget from profiled success rates per workflow and observation, so
slack cannot move from one question to another.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..grammar import Workflow, parse_chain
from ..planner.execute import first_block, run_workflow
from ..planner.meter import Meter
from ..planner.planner import RunResult
from ..trace.corpus import TraceCorpus
from .base import Method
from .profiles import DEFAULT_CONTINUATIONS, Profile, lookup, profile_continuations, profile_library


def _openers(library: dict[str, Workflow]) -> dict[str, Workflow]:
    return {t: library[t] for t in ("S", "M", "L") if t in library}


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
                                           ("stop", "esc:L", "rep:L"))
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
        opener = self.library[tier]

        left = len(records)
        for rec in records.values():
            left -= 1
            run_workflow(ctx, rec, first_block(opener))
            if rec.closed:
                continue
            obs = rec.state.answer_obs or "-"
            spec = "rep:L" if obs.split(".")[0] in ("error", "empty") else "esc:L"
            prof = lookup(self.table, tier, obs, spec)
            if prof is None:
                continue
            s = rec.s_held
            gain = max(0.0, prof.acc - s)
            scale = np.maximum(per_q, 1e-12)
            norm = float(np.max(prof.cost / scale))
            if norm <= 0:
                continue
            # the price tracks what is left: the tighter the budget, the higher
            # the gain an escalation must promise
            frac_left = float(np.min(meter.remaining / np.maximum(meter.budget, 1e-12)))
            price = (1.0 - frac_left) + 0.05
            if gain / norm > price:
                chain = parse_chain(spec)
                if self.grammar.is_wellformed(chain, rec.state):
                    run_workflow(ctx, rec, chain)
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
        opener = self.library[self.first_tier]
        for rec in records.values():
            run_workflow(ctx, rec, first_block(opener))
        ranked = sorted((r for r in records.values() if not r.closed),
                        key=lambda r: r.s_held)
        chain = parse_chain(self.escalation)
        for rec in ranked:
            if not self.grammar.is_wellformed(chain, rec.state):
                continue
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

    def prepare(self, corpus, train_dbs) -> "VineLMStyle":
        super().prepare(corpus, train_dbs)
        self.profiles = profile_library(corpus, train_dbs, _openers(self.library))
        self.table = profile_continuations(corpus, train_dbs, _openers(self.library),
                                           self.specs)
        return self

    def run(self, metas, budget, meter=None) -> RunResult:
        meter = meter or Meter(budget)
        records = self._records(metas)
        per_q = np.asarray(budget, dtype=float) / max(1, len(metas))
        ctx = self._ctx(meter)

        opener_name = max(
            (n for n, p in self.profiles.items() if p.n and np.all(p.cost <= per_q)),
            key=lambda n: self.profiles[n].acc, default="S")
        opener = self.library[opener_name]

        for rec in records.values():
            run_workflow(ctx, rec, first_block(opener))
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
                if prof.acc > base:
                    base, best = prof.acc, spec
            if best is None:
                continue
            chain = parse_chain(best)
            if self.grammar.is_wellformed(chain, rec.state):
                run_workflow(ctx, rec, chain)
        return self._result(records, meter)
