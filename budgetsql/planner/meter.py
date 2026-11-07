"""The meter: one hard budget for every call, the planner's own included.

The meter never trusts a prediction.  Calls may run in parallel, so budget is
reserved at launch: a call starts only if its worst case fits what is left after
the spend and the reservations in flight.  The worst case is the call's prompt,
whose tokens and price are known before sending, plus its completion limit and
its timeout for every sample it draws.  On return the unused part of the
reservation is released.  A call that reaches a limit is cancelled, charged what
it consumed, and returns nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import DIMS, PRICES, zeros
from ..grammar import Stage


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

    def as_dict(self) -> dict:
        return {
            "launched": self.launched, "refused": self.refused,
            "cancelled": self.cancelled,
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
               planning: bool = False) -> np.ndarray:
        """Charge what the call consumed and release the rest of its reservation."""
        if r.released:
            raise BudgetExceeded("reservation settled twice")
        charged = np.minimum(np.asarray(cost, dtype=float), r.worst_case)
        self._reservations.pop(r.rid, None)
        r.released = True
        self.stats.spend = self.stats.spend + charged
        if planning:
            self.stats.planning_spend = self.stats.planning_spend + charged
        if cancelled:
            self.stats.cancelled += 1
        if np.any(self.stats.spend > self.budget + 1e-9):
            raise BudgetExceeded(f"spend {self.stats.spend} exceeds budget {self.budget}")
        return charged

    def charge(self, cost: np.ndarray, worst_case: np.ndarray | None = None,
               label: str = "", planning: bool = False) -> np.ndarray | None:
        """Reserve, run and settle in one step, for calls run one at a time.
        Returns the charged cost, or ``None`` when the meter refuses the call."""
        wc = np.asarray(cost if worst_case is None else worst_case, dtype=float)
        if planning and not self._planning_allowed(wc):
            self.stats.refused += 1
            return None
        r = self.reserve(wc, label)
        if r is None:
            return None
        cancelled = bool(np.any(np.asarray(cost, dtype=float) > r.worst_case + 1e-12))
        return self.settle(r, cost, cancelled=cancelled, planning=planning)

    def _planning_allowed(self, worst_case: np.ndarray) -> bool:
        if self.planning_cap is None:
            return True
        cap = self.planning_cap * self.budget[0]
        return bool(self.stats.planning_spend[0] + worst_case[0] <= cap + 1e-12)

    # ---- limits a remaining budget can pay for ----
    def affordable_completion(self, stage: Stage, prompt_tokens: float) -> int:
        """The completion limit the remaining budget can pay for."""
        price = PRICES.get(stage.tier)
        free = self.free
        cap = stage.cap
        if price is None:
            return cap
        per_sample = max(1, stage.samples)
        usd_left = max(0.0, free[0]) / per_sample - prompt_tokens * price.inp * 1e-6
        by_usd = int(max(0.0, usd_left) / (price.out * 1e-6)) if price.out > 0 else cap
        tok_left = max(0.0, free[1]) / per_sample - prompt_tokens
        return int(max(0, min(cap, by_usd, tok_left)))

    def worst_case_of(self, stage: Stage, prompt_tokens: float,
                      timeout_s: float = 10.0, cap: int | None = None) -> np.ndarray:
        price = PRICES.get(stage.tier)
        if price is None:
            return np.array([0.0, 0.0, timeout_s * 0.1])
        c = stage.cap if cap is None else cap
        n = max(1, stage.samples)
        return np.array([
            (prompt_tokens * price.inp + c * price.out) * 1e-6 * n,
            (prompt_tokens + c) * n,
            timeout_s * n,
        ])

    def snapshot(self) -> dict:
        return {"budget": self.budget.tolist(), **self.stats.as_dict(),
                "utilisation": self.utilisation()}
