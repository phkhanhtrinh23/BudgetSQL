"""Methods that decide before observing anything.

``BestFixedMixture`` picks, per budget group, the random mixture of workflows
with the highest accuracy on other databases whose predicted mean cost fits the
group's budget in every dimension.  ``BestFixedWorkflow`` picks one workflow the
same way, from the one-shot workflows of the reference table and the pruned
``M_c+R_L``.  ``OmniRouterStyle`` couples the batch in one
linear program, but solves it once on predictions made before any answer exists.
``Oracle`` is the hindsight optimum of the same relaxation with true outcomes:
it reports the relaxation's value, not a rounded plan's.

All but the oracle run under the shared pacing rule (``Method._paced``).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog

from ..config import zeros
from ..grammar import LIBRARY_ORDER, State, Workflow
from ..planner.allocation import Candidate, solve
from ..planner.meter import Meter
from ..planner.planner import RunResult
from ..trace.corpus import TraceCorpus
from .base import Method
from .profiles import Profile, profile_library


def mixture_lp(names: list[str], profiles: dict[str, Profile],
               per_question_budget: np.ndarray) -> dict[str, float]:
    """The best feasible random mixture of workflows, as shares that sum to one."""
    acc = np.array([profiles[n].acc for n in names])
    cost = np.stack([profiles[n].cost for n in names], axis=1)      # d x m
    res = linprog(-acc, A_ub=cost, b_ub=np.asarray(per_question_budget, dtype=float),
                  A_eq=np.ones((1, len(names))), b_eq=[1.0], bounds=(0.0, None),
                  method="highs-ds")
    if not res.success or res.x is None:
        cheapest = min(names, key=lambda n: float(np.sum(profiles[n].cost)))
        return {cheapest: 1.0}
    return {n: float(v) for n, v in zip(names, res.x) if v > 1e-9}


@dataclass
class BestFixedMixture(Method):
    name: str = "best_fixed_mixture"
    profiles: dict[str, Profile] = field(default_factory=dict)
    seed: int = 0

    def prepare(self, corpus: TraceCorpus, train_dbs: list[str]) -> "BestFixedMixture":
        super().prepare(corpus, train_dbs)
        self.profiles = profile_library(corpus, train_dbs, self.library)
        return self

    def run(self, metas, budget, meter=None) -> RunResult:
        meter = meter or Meter(budget)
        records = self._records(metas)
        names = [n for n in self.library if self.profiles.get(n, Profile()).n]
        per_q = np.asarray(budget, dtype=float) / max(1, len(metas))
        shares = mixture_lp(names, self.profiles, per_q)

        rng = random.Random(self.seed)
        order = list(records)
        rng.shuffle(order)
        plan: dict[str, Workflow] = {}
        picks = list(shares.items())
        cum = np.cumsum([s for _, s in picks])
        for qid in order:
            u = rng.random() * (cum[-1] if len(cum) else 1.0)
            idx = int(np.searchsorted(cum, u))
            plan[qid] = self.library[picks[min(idx, len(picks) - 1)][0]]

        ctx = self._ctx(meter)
        from ..planner.execute import run_workflow
        arrival = list(records)
        for i, qid in enumerate(arrival):
            rec = records[qid]
            wf = plan[qid]
            mean = self.profiles[wf.name or ""].cost if (wf.name in self.profiles) else zeros()
            wf = self._paced_opening(meter, rec, wf, mean, len(arrival) - i - 1)
            run_workflow(ctx, rec, wf)
        predicted = float(sum(self.profiles[n].acc * s for n, s in shares.items()))
        return self._result(records, meter, predicted)


#: What a deployment that runs one workflow for every question may choose from:
#: the one-shot reference workflows and the pruned repair chain.
ONE_SHOT = tuple(LIBRARY_ORDER[:13]) + ("M_c+R_L",)


@dataclass
class BestFixedWorkflow(Method):
    """The most accurate single workflow whose mean cost fits, chosen on the
    other databases."""

    name: str = "best_fixed_workflow"
    profiles: dict[str, Profile] = field(default_factory=dict)
    choice: str = ""

    def prepare(self, corpus: TraceCorpus, train_dbs: list[str]
                ) -> "BestFixedWorkflow":
        super().prepare(corpus, train_dbs)
        self.profiles = profile_library(corpus, train_dbs, self.library)
        return self

    def _choose(self, per_question: np.ndarray) -> str:
        names = [n for n in ONE_SHOT
                 if n in self.library and self.profiles.get(n, Profile()).n]
        fits = [n for n in names
                if np.all(self.profiles[n].cost <= per_question + 1e-12)]
        pool = fits or names
        return max(pool, key=lambda n: self.profiles[n].acc)

    def run(self, metas, budget, meter=None) -> RunResult:
        meter = meter or Meter(budget)
        records = self._records(metas)
        per_q = np.asarray(budget, dtype=float) / max(1, len(metas))
        self.choice = self._choose(per_q)
        wf = self.library[self.choice]
        mean = self.profiles[self.choice].cost

        from ..planner.execute import run_workflow
        ctx = self._ctx(meter)
        recs = list(records.values())
        for i, rec in enumerate(recs):
            run = self._paced_opening(meter, rec, wf, mean, len(recs) - i - 1)
            run_workflow(ctx, rec, run)
        return self._result(records, meter, self.profiles[self.choice].acc)


@dataclass
class OmniRouterStyle(Method):
    """One linear program over the batch, solved once, before anything runs."""

    name: str = "omnirouter_style"

    def prepare(self, corpus, train_dbs) -> "OmniRouterStyle":
        super().prepare(corpus, train_dbs)
        from ..worldmodel.model import StageWorldModel
        self.model = StageWorldModel(self.cfg).fit(corpus, train_dbs)
        return self

    def run(self, metas, budget, meter=None) -> RunResult:
        meter = meter or Meter(budget)
        records = self._records(metas)
        wfs = list(self.library.values())
        items = [(m, State(), wf) for wf in wfs for m in metas]
        rolls = self.model.rollout_many(items, [0.0] * len(items))
        cands: list[Candidate] = []
        k = 0
        for wf in wfs:
            for m in metas:
                r = rolls[k]
                cands.append(Candidate(m.qid, wf.name or "", r.p, r.cost, r.var,
                                       r.worst_case, payload=wf))
                k += 1
        for m in metas:
            cands.append(Candidate(m.qid, "stop", 0.0, zeros(), zeros(), zeros(),
                                   is_stop=True, payload=None))
        alloc = solve(cands, np.asarray(budget, dtype=float))
        from ..planner.execute import run_workflow
        ctx = self._ctx(meter)
        chosen = alloc.chosen()
        recs = list(records.values())
        for i, rec in enumerate(recs):
            cand = chosen.get(rec.qid)
            if cand is None or cand.payload is None:
                continue
            wf = self._paced_opening(meter, rec, cand.payload, cand.cost,
                                     len(recs) - i - 1)
            run_workflow(ctx, rec, wf)
        n = max(1, len(metas))
        return self._result(records, meter, alloc.value / n, [alloc.duals])


@dataclass
class Oracle(Method):
    """The hindsight optimum of the linear relaxation: the same program with
    the outcomes and costs the trace actually produced.  Its accuracy is the
    relaxation's value, so a question split between workflows counts with the
    mixture of their outcomes; nothing is rounded or executed."""

    name: str = "oracle"

    def run(self, metas, budget, meter=None) -> RunResult:
        """The value of the linear relaxation at the true outcomes and costs.
        A question's entry in ``correct`` is its fractional value
        ``sum_w x_iw Y_iw``, so the reported accuracy is the relaxation's
        optimum, not that of a rounded plan."""
        meter = meter or Meter(budget)
        cands: list[Candidate] = []
        for m in metas:
            for name, wf in self.library.items():
                _, ok, cost, _ = self.corpus.run(m, State(), wf)
                cands.append(Candidate(m.qid, name, float(ok), cost, zeros(), cost,
                                       payload=wf))
            cands.append(Candidate(m.qid, "stop", 0.0, zeros(), zeros(), zeros(),
                                   is_stop=True, payload=None))
        alloc = solve(cands, np.asarray(budget, dtype=float))
        correct = {m.qid: 0.0 for m in metas}
        for c, xv in zip(alloc.candidates, alloc.x):
            if xv > 0:
                correct[c.qid] += float(xv) * float(c.value)
        n = max(1, len(metas))
        return RunResult(
            accuracy=alloc.value / n, correct=correct, spend=alloc.spend.copy(),
            budget=np.asarray(budget, dtype=float).copy(), meter=meter.snapshot(),
            duals_history=[alloc.duals], predicted_accuracy=alloc.value / n,
            n_questions=len(metas))
