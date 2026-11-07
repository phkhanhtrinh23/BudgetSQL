"""How much of the budget to hold back for the block that overruns.

Compliance comes from the meter; a reserve only decides what an overrun costs.
A plan that overruns loses the continuations the meter refuses, and those
questions keep their held answers.  Valuing each priced unit of overrun at the
smallest gain per priced unit among the planned continuations gives a
newsvendor condition, whose quantile a Bernstein bound turns into a closed form.
Nothing here is tuned: the variances come from the world model and the ranges
from the meter.
"""
from __future__ import annotations

import math

import numpy as np

from ..config import D
from .allocation import Allocation, Candidate


def theta(alloc: Allocation, tol: float = 1e-12) -> float:
    """The smallest gain per priced unit among the planned continuations."""
    stop = {c.qid: c.value for c in alloc.candidates if c.is_stop}
    ratios = []
    for c, xv in zip(alloc.candidates, alloc.x):
        if xv <= tol or c.is_stop:
            continue
        priced = float(alloc.duals @ c.cost)
        gain = c.value - stop.get(c.qid, 0.0)
        if priced > tol and gain > 0:
            ratios.append(gain / priced)
    if not ratios:
        return 1.0
    return max(1.0, min(ratios))


def run_order(alloc: Allocation) -> list[Candidate]:
    """Continuations run in decreasing order of gain per priced unit; those with
    no priced cost run first, in decreasing order of gain."""
    stop = {c.qid: c.value for c in alloc.candidates if c.is_stop}
    free, priced = [], []
    for c, xv in zip(alloc.candidates, alloc.x):
        if xv <= 1e-12 or c.is_stop:
            continue
        gain = c.value - stop.get(c.qid, 0.0)
        p = float(alloc.duals @ c.cost)
        (free if p <= 0 else priced).append((gain / p if p > 0 else gain, c))
    free.sort(key=lambda t: -t[0])
    priced.sort(key=lambda t: -t[0])
    return [c for _, c in free] + [c for _, c in priced]


def bernstein_reserve(variances: np.ndarray, ranges: np.ndarray,
                      log_term: float) -> np.ndarray:
    """The Bernstein bound on the block's cost deviation."""
    log_term = max(0.0, float(log_term))
    return np.sqrt(2.0 * np.asarray(variances, dtype=float) * log_term) \
        + (2.0 / 3.0) * np.asarray(ranges, dtype=float) * log_term


def reserve_for(alloc: Allocation, block: int, duals: np.ndarray,
                delta: float | None = None) -> np.ndarray:
    """The reserve the next block of continuations deserves.

    With ``delta`` given, the log term is ``log(d/delta)`` and the reserve is the
    one that bounds the miss probability; otherwise it follows the newsvendor
    condition at the plan's own gain per priced unit.
    """
    order = run_order(alloc)[:max(0, block)]
    if not order:
        return np.zeros(D)
    var = np.zeros(D)
    rng = np.zeros(D)
    for c in order:
        var = var + np.asarray(c.var, dtype=float)
        rng = np.maximum(rng, np.asarray(c.worst_case, dtype=float))
    log_term = math.log(D / delta) if delta else math.log(max(1.0, theta(alloc)))
    rho = bernstein_reserve(var, rng, log_term)
    duals = np.asarray(duals, dtype=float)
    rho = np.where(duals > 0, rho, 0.0)     # held only in priced dimensions
    return rho
