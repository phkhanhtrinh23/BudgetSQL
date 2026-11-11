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


def theta(alloc: Allocation, tol: float = 1e-12, block: int | None = None) -> float:
    """The smallest gain per priced unit, at least one (complementary
    slackness), among the continuations planned before the next re-solve: the
    first ``block`` of the run order, or every planned one."""
    stop = {c.qid: c.value for c in alloc.candidates if c.is_stop}
    planned = run_order(alloc) if block is None else run_order(alloc)[:max(0, block)]
    ratios = []
    for c in planned:
        priced = float(alloc.duals @ c.cost)
        gain = c.value - stop.get(c.qid, 0.0)
        if priced > tol and gain > 0:
            ratios.append(gain / priced)
    if not ratios:
        return 1.0
    return max(1.0, min(ratios))


def run_order(alloc: Allocation, tol: float = 1e-7) -> list[Candidate]:
    """The planned continuations in the order they run: decreasing gain per
    priced unit, those with no priced cost first, in decreasing order of gain.

    The extreme point splits at most d questions between candidates, and a
    split question stops at this solve, keeping its held answer: only questions whose whole mass is on one continuation run."""
    stop = {c.qid: c.value for c in alloc.candidates if c.is_stop}
    split = set(alloc.split_questions(tol))
    free, priced = [], []
    for c, xv in zip(alloc.candidates, alloc.x):
        if xv <= tol or c.is_stop or c.qid in split:
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
                delta: float | None = None, z: float | None = None) -> np.ndarray:
    """The Bernstein reserve rho for the next block of continuations.

    Per dimension, the variances of the block's continuations are summed and
    c-bar is the meter's worst case for one continuation.  With ``delta``
    given, the log term is ``log(d/delta)``; with ``z`` the
    reserve is that many predicted standard deviations; otherwise the log term
    is ``log theta`` at the block's smallest gain per priced unit (the
    newsvendor condition).  rho_k = 0 where lambda_k = 0.
    """
    order = run_order(alloc)[:max(0, block)]
    if not order:
        return np.zeros(D)
    var = np.zeros(D)
    rng = np.zeros(D)
    for c in order:
        var = var + np.asarray(c.var, dtype=float)
        rng = np.maximum(rng, np.asarray(c.worst_case, dtype=float))
    if z is not None:
        rho = float(z) * np.sqrt(var)
    else:
        log_term = (math.log(D / delta) if delta
                    else math.log(max(1.0, theta(alloc, block=block))))
        rho = bernstein_reserve(var, rng, log_term)
    duals = np.asarray(duals, dtype=float)
    rho = np.where(duals > 0, rho, 0.0)
    return rho
