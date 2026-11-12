from __future__ import annotations

import numpy as np

from ..config import DIMS, PlannerConfig, zeros
from .digest import Digest
from .meter import Meter
from .planner import BudgetSQL, RunResult


class BatchPlanner:
    """Drives the planners of a batch's budget groups and holds the agent."""

    def __init__(self, planners: list[BudgetSQL], cfg: PlannerConfig | None = None,
                 agent=None):
        self.planners = list(planners)
        self.cfg = cfg or self.planners[0].cfg
        self.agent = agent
        self.turns = 0
        self.sessions = 0
        self.session_turns = 0
        self.turn_log: list[dict] = []
        for p in self.planners:
            p.batch = self
        if agent is not None and hasattr(agent, "configure"):
            agent.configure(self.cfg)

    @property
    def uses_agent(self) -> bool:
        if not self.cfg.use_agent:
            return False
        return self.cfg.agent_mode == "modelled" or self.agent is not None

    def turn_worst_case(self) -> np.ndarray:
        return self.planners[0].turn_worst_case()

    def agent_available(self) -> bool:
        if not self.uses_agent:
            return False
        if self.turns >= self.cfg.turns_per_session * self.cfg.max_sessions:
            return False
        if self.cfg.free_turns:
            return True
        return all(p.meter is None or p.meter.planning_allowed(p.turn_charge())
                   for p in self.planners)

    def run(self, groups: list[tuple[list, np.ndarray, Meter | None]]) -> list[RunResult]:
        assert len(groups) == len(self.planners)
        total = sum(len(metas) for metas, _, _ in groups) or 1
        for p, (metas, _, _) in zip(self.planners, groups):
            p.turn_share = len(metas) / total
        self.turns = self.sessions = 0
        self.turn_log = []

        for p, (metas, budget, meter) in zip(self.planners, groups):
            p.start(metas, budget, meter)
        if self.cfg.plan_whole_workflows:
            return self._finish()

        if self.uses_agent:
            for p in self.planners:
                p.resolve()
            self._session(opening=True)

        if not self.cfg.periodic_resolves:
            for p in self.planners:
                if p.resolve() is not None:
                    p.run_next_block(whole_plan=True)
            return self._finish()

        while True:
            ran = False
            for p in self.planners:
                if p.resolve() is not None and p.run_next_block():
                    ran = True
            if not ran:
                break
            flagged = self.flagged() if self.uses_agent else []
            if flagged:
                for p in self.planners:
                    p.add_edits(flagged)
                    p.resolve()
                self._session()
        return self._finish()

    def _finish(self) -> list[RunResult]:
        return [p.finish(self.turns, self.sessions) for p in self.planners]

    def flagged(self) -> list[str]:
        return [c for p in self.planners if p.is_open for c in p.flagged_cells()]

    def digest(self) -> Digest:
        cells = [c for p in self.planners if p.meter is not None for c in p.digest_cells()]
        budget, remaining, spent, planning = zeros(), zeros(), zeros(), zeros()
        duals = zeros()
        for p in self.planners:
            if p.meter is None:
                continue
            budget = budget + p.meter.budget
            remaining = remaining + p.meter.remaining
            spent = spent + p.meter.stats.spend
            planning = planning + p.meter.stats.planning_spend
            if p._last_alloc is not None:
                duals = duals + p.turn_share * p._last_alloc.duals
        n_open = sum(1 for p in self.planners for r in p.records.values() if not r.closed)
        return Digest(cells, remaining, spent, planning, duals, budget,
                      n_open=n_open, turns_used=self.turns,
                      turns_left=max(0, self.cfg.turns_per_session - self.session_turns)
                      ).fit(self.cfg.digest_tokens, self.cfg.max_digest_cells)

    def certificate_passes(self) -> bool:
        price = 0.0
        for p in self.planners:
            alloc = p._last_alloc
            if alloc is not None and not self.cfg.free_turns:
                price += float(alloc.duals @ p.turn_charge())
        psi = 0.0
        for p in self.planners:
            if not p.is_open:
                continue
            cert = p.certificate(need=price - psi)
            psi += cert.psi
            if psi > price:
                return True
        return psi > price

    def _session(self, opening: bool = False) -> None:
        from .tools import ToolBox

        if self.sessions >= self.cfg.max_sessions or not self.agent_available():
            return
        self.sessions += 1
        self.session_turns = 0
        box = ToolBox(self)
        for _ in range(self.cfg.turns_per_session):
            if not self.agent_available():
                break
            if not (opening and self.session_turns == 0) and not self.certificate_passes():
                break
            digest = self.digest()
            if not digest.cells:
                break
            if not self._take_turn(digest, box):
                break

    def _take_turn(self, digest: Digest, box) -> bool:
        reservations = []
        for p in self.planners:
            if self.cfg.free_turns:
                break
            r = p.meter.reserve_planning(p.turn_charge(), label="agent turn")
            if r is None:
                for q, rr in reservations:
                    q.meter.release(rr)
                return False
            reservations.append((p, r))

        cost = None
        box.digest = digest
        if self.cfg.agent_mode == "live" and self.agent is not None:
            try:
                cost = self.agent.turn(digest, box)
            except Exception:
                cost = None
        else:
            for p in self.planners:
                if p.is_open:
                    p.resolve()
        if cost is None and self.cfg.modelled_turn_cost is not None:
            cost = np.asarray(self.cfg.modelled_turn_cost, dtype=float)
        wc = self.turn_worst_case()
        reported = wc if cost is None else np.asarray(cost, dtype=float)
        exceeded = bool(np.any(reported > wc + 1e-12))
        cost = np.minimum(reported, wc)
        for p, r in reservations:
            p.meter.settle(r, p.turn_share * cost, planning=True)
        self.turns += 1
        self.session_turns += 1
        self.turn_log.append({"session": self.sessions, "exceeded": exceeded,
                              **{k: float(v) for k, v in zip(DIMS, reported)}})
        return True
