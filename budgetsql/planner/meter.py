"""The meter: one hard budget for every call, the planner's own included.

The meter never trusts a prediction.  Calls may run in parallel, so budget is
reserved at launch: a call starts only if its worst case fits what is left after
the spend and the reservations in flight.  The worst case is the
call's prompt, whose tokens and price are known before sending, plus its
completion limit and its timeout for every sample it draws, plus the timeout of
every SQL statement it runs.  Those limits are set to what the remainder can pay
for.  On return the unused part of the reservation is released.  A call that
reaches a limit is cancelled, charged what it consumed, and returns nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import DIMS, PRICES, ExecutorLimits, zeros
from ..costs import sql_statements, stage_worst_case
from ..grammar import Stage

#: An LLM call is launched only if the remainder pays for at least this many
#: completion tokens and this many seconds per sample.
MIN_USEFUL_COMPLETION = 16
MIN_USEFUL_TIMEOUT_S = 0.5


@dataclass(frozen=True)
class CallLimits:
    """The limits a call is launched with and the reservation they imply."""

    completion: int
    timeout_s: float
    sql_timeout_s: float
    worst_case: np.ndarray


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Reservation:
    rid: int
    worst_case: np.ndarray
    label: str = ""
    released: bool = False


@dataclass
class MeterStats:
    launched: int = 0
    refused: int = 0
    cancelled: int = 0
    spend: np.ndarray = field(default_factory=zeros)
    planning_spend: np.ndarray = field(default_factory=zeros)
    #: what the calls of each executor tier were charged
    tier_spend: dict[str, np.ndarray] = field(default_factory=dict)

    def tier_money_share(self) -> dict[str, float]:
        """Each tier's share of the money spent on answering."""
        total = sum(float(v[0]) for v in self.tier_spend.values())
        return {t: (float(v[0]) / total if total > 0 else 0.0)
                for t, v in self.tier_spend.items()}

    def as_dict(self) -> dict:
        return {
            "launched": self.launched, "refused": self.refused,
            "cancelled": self.cancelled,
            "tier_money_share": self.tier_money_share(),
            **{f"spend_{k}": float(v) for k, v in zip(DIMS, self.spend)},
            **{f"planning_{k}": float(v) for k, v in zip(DIMS, self.planning_spend)},
        }


class Meter:
    """Charges every call against one budget vector."""

    def __init__(self, budget: np.ndarray, planning_cap: float | None = None):
        self.budget = np.asarray(budget, dtype=float).copy()
        self.planning_cap = planning_cap
        self.stats = MeterStats()
        self._reservations: dict[int, Reservation] = {}
        self._next = 0

    # ---- state ----
    @property
    def spend(self) -> np.ndarray:
        return self.stats.spend

    @property
    def held(self) -> np.ndarray:
        out = zeros()
        for r in self._reservations.values():
            out = out + r.worst_case
        return out

    @property
    def remaining(self) -> np.ndarray:
        return np.maximum(self.budget - self.spend, 0.0)

    @property
    def free(self) -> np.ndarray:
        """What a new call may reserve: the budget less the spend and the
        reservations in flight."""
        return self.budget - self.spend - self.held

    def utilisation(self) -> dict[str, float]:
        out = {}
        for k, s, b in zip(DIMS, self.spend, self.budget):
            out[k] = float(s / b) if b > 0 else 0.0
        return out

    # ---- launching ----
    def fits(self, worst_case: np.ndarray) -> bool:
        return bool(np.all(np.asarray(worst_case, dtype=float) <= self.free + 1e-12))

    def reserve(self, worst_case: np.ndarray, label: str = "") -> Reservation | None:
        wc = np.asarray(worst_case, dtype=float)
        if not self.fits(wc):
            self.stats.refused += 1
            return None
        self._next += 1
        r = Reservation(self._next, wc.copy(), label)
        self._reservations[r.rid] = r
        self.stats.launched += 1
        return r

    def settle(self, r: Reservation, cost: np.ndarray, cancelled: bool = False,
               planning: bool = False, tier: str | None = None) -> np.ndarray:
        """Charge what the call consumed and release the rest of its reservation."""
        if r.released:
            raise BudgetExceeded("reservation settled twice")
        charged = np.minimum(np.asarray(cost, dtype=float), r.worst_case)
        self._reservations.pop(r.rid, None)
        r.released = True
        self.stats.spend = self.stats.spend + charged
        if planning:
            self.stats.planning_spend = self.stats.planning_spend + charged
        if tier:
            self.stats.tier_spend[tier] = self.stats.tier_spend.get(tier, zeros()) + charged
        if cancelled:
            self.stats.cancelled += 1
        if np.any(self.stats.spend > self.budget + 1e-9):
            raise BudgetExceeded(f"spend {self.stats.spend} exceeds budget {self.budget}")
        return charged

    def release(self, r: Reservation) -> None:
        """Release a reservation whose call never started, charging nothing."""
        if r.released:
            raise BudgetExceeded("reservation released twice")
        self._reservations.pop(r.rid, None)
        r.released = True
        self.stats.launched -= 1

    def charge(self, cost: np.ndarray, worst_case: np.ndarray | None = None,
               label: str = "", planning: bool = False,
               cancelled: bool | None = None, tier: str | None = None
               ) -> np.ndarray | None:
        """Reserve, run and settle in one step, for calls run one at a time.
        Returns the charged cost, or ``None`` when the meter refuses the call.
        A call whose cost exceeds its reservation in any dimension reached a
        limit and is cancelled; ``cancelled`` says so explicitly when the
        caller has already cut the cost at the limit."""
        wc = np.asarray(cost if worst_case is None else worst_case, dtype=float)
        if planning and not self._planning_allowed(wc):
            self.stats.refused += 1
            return None
        r = self.reserve(wc, label)
        if r is None:
            return None
        if cancelled is None:
            cancelled = bool(np.any(np.asarray(cost, dtype=float) > r.worst_case + 1e-12))
        return self.settle(r, cost, cancelled=cancelled, planning=planning, tier=tier)

    def reserve_planning(self, worst_case: np.ndarray, label: str = ""
                         ) -> Reservation | None:
        """Reserve a planning call, which must also fit the planning cap."""
        wc = np.asarray(worst_case, dtype=float)
        if not self._planning_allowed(wc):
            self.stats.refused += 1
            return None
        return self.reserve(wc, label)

    def planning_allowed(self, worst_case: np.ndarray) -> bool:
        return self._planning_allowed(np.asarray(worst_case, dtype=float))

    def _planning_allowed(self, worst_case: np.ndarray) -> bool:
        if self.planning_cap is None:
            return True
        cap = self.planning_cap * self.budget[0]
        return bool(self.stats.planning_spend[0] + worst_case[0] <= cap + 1e-12)

    # ---- limits a remaining budget can pay for ----
    def affordable_completion(self, stage: Stage, prompt_tokens: float,
                              ceiling: int | None = None) -> int:
        """The completion limit the remaining budget can pay for."""
        price = PRICES.get(stage.tier)
        free = self.free
        cap = stage.cap if ceiling is None else int(ceiling)
        if price is None:
            return cap
        per_sample = max(1, stage.samples)
        usd_left = max(0.0, free[0]) / per_sample - prompt_tokens * price.inp * 1e-6
        by_usd = int(max(0.0, usd_left) / (price.out * 1e-6)) if price.out > 0 else cap
        tok_left = max(0.0, free[1]) / per_sample - prompt_tokens
        return int(max(0, min(cap, by_usd, tok_left)))

    def limits_for(self, stage: Stage, prompt_tokens: float,
                   limits: ExecutorLimits | None = None) -> CallLimits | None:
        """The limits a stage is launched with: its completion cap and timeouts,
        each lowered to what the remainder can pay for.  ``None`` when the
        remainder cannot pay for a useful call, which the caller treats as a
        refusal."""
        limits = limits or ExecutorLimits()
        free = self.free
        n_sql = sql_statements(stage)
        sql_t = float(limits.sql_timeout_s)
        if stage.kind == "context":
            return CallLimits(0, 0.0, 0.0, stage_worst_case(stage, 0.0, limits))
        if not stage.llm_calls:
            if n_sql:
                sql_t = min(sql_t, max(0.0, float(free[2])) / n_sql)
                if sql_t <= 0.0:
                    return None
            return CallLimits(0, 0.0, sql_t,
                              stage_worst_case(stage, 0.0, limits, sql_timeout_s=sql_t))
        n = max(1, stage.samples)
        ceiling = limits.cap_for(stage.tier, stage.cap)
        cap = self.affordable_completion(stage, prompt_tokens, ceiling)
        if cap < MIN_USEFUL_COMPLETION:
            return None
        llm_time = max(0.0, float(free[2]) - n_sql * sql_t) / n
        t = min(float(limits.timeout_s), llm_time)
        if t < MIN_USEFUL_TIMEOUT_S:
            return None
        return CallLimits(cap, t, sql_t,
                          stage_worst_case(stage, prompt_tokens, limits, cap=cap,
                                           timeout_s=t, sql_timeout_s=sql_t))

    def worst_case_of(self, stage: Stage, prompt_tokens: float,
                      timeout_s: float = 10.0, cap: int | None = None,
                      sql_timeout_s: float | None = None) -> np.ndarray:
        """The worst case of a stage at the given limits; an SQL statement's
        worst case is the SQL timeout, not a share of the LLM timeout."""
        lim = ExecutorLimits(timeout_s=timeout_s,
                             sql_timeout_s=timeout_s if sql_timeout_s is None
                             else sql_timeout_s)
        return stage_worst_case(stage, prompt_tokens, lim,
                                cap=stage.cap if cap is None else cap)

    def snapshot(self) -> dict:
        return {"budget": self.budget.tolist(), **self.stats.as_dict(),
                "utilisation": self.utilisation()}
