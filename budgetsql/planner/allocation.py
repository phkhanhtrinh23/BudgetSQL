"""The allocation tool: a linear program over the agent's candidates.

Each open question spends its one unit of mass on the candidates registered for
it, the budget rows are packing constraints, and the program's dual prices say
what one unit of each dimension is worth.  The solver returns an extreme point,
which splits at most as many questions as there are budget dimensions; taking
the extreme point rather than each question's own maximiser at the optimal
prices matters, because when many questions tie at those prices, independent
maximisers can all choose the expensive side and overrun the budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from ..config import D, DIMS


@dataclass
class Candidate:
    """One candidate workflow of one question, with the value and cost the world
    model predicts for it."""

    qid: str
    wf_id: str
    value: float
    cost: np.ndarray
    var: np.ndarray = field(default_factory=lambda: np.zeros(D))
    worst_case: np.ndarray = field(default_factory=lambda: np.zeros(D))
    is_stop: bool = False
    payload: object = None


@dataclass
class Allocation:
    x: np.ndarray                      # mass on each candidate
    value: float
    spend: np.ndarray
    duals: np.ndarray                  # lambda, one price per dimension
    omega: np.ndarray                  # one price per question
    qids: list[str]
    candidates: list[Candidate]
    status: str = "optimal"

    def chosen(self) -> dict[str, Candidate]:
        """The candidate each question runs.  A split question takes the side
        with the larger share."""
        best: dict[str, tuple[float, Candidate]] = {}
        for c, xv in zip(self.candidates, self.x):
            cur = best.get(c.qid)
            if cur is None or xv > cur[0]:
                best[c.qid] = (xv, c)
        return {q: c for q, (_, c) in best.items()}

    def split_questions(self, tol: float = 1e-7) -> list[str]:
        counts: dict[str, int] = {}
        for c, xv in zip(self.candidates, self.x):
            if xv > tol:
                counts[c.qid] = counts.get(c.qid, 0) + 1
        return [q for q, n in counts.items() if n > 1]

    def price_of(self, dim: str) -> float:
        return float(self.duals[DIMS.index(dim)])

    def binding(self, tol: float = 1e-12) -> list[str]:
        return [k for k, v in zip(DIMS, self.duals) if v > tol]


def solve(candidates: list[Candidate], budget: np.ndarray,
          reserve: np.ndarray | None = None,
          priced_dims: tuple[str, ...] = DIMS) -> Allocation:
    """Solve the allocation program.  ``reserve`` is held back from the budget
    for later planning turns and probes."""
    if not candidates:
        return Allocation(np.zeros(0), 0.0, np.zeros(D), np.zeros(D), np.zeros(0), [], [])

    qids = sorted({c.qid for c in candidates})
    qindex = {q: i for i, q in enumerate(qids)}
    n, m = len(qids), len(candidates)

    rhs = np.asarray(budget, dtype=float).copy()
    if reserve is not None:
        rhs = rhs - np.asarray(reserve, dtype=float)
    rhs = np.maximum(rhs, 0.0)

    dim_rows = [DIMS.index(k) for k in priced_dims]

    c = np.array([-cd.value for cd in candidates])
    rows, cols, vals = [], [], []
    for j, cd in enumerate(candidates):
        rows.append(qindex[cd.qid]); cols.append(j); vals.append(1.0)
    A_eq = coo_matrix((vals, (rows, cols)), shape=(n, m)).tocsr()
    b_eq = np.ones(n)

    rows, cols, vals = [], [], []
    for r, k in enumerate(dim_rows):
        for j, cd in enumerate(candidates):
            v = float(cd.cost[k])
            if v != 0.0:
                rows.append(r); cols.append(j); vals.append(v)
    A_ub = coo_matrix((vals, (rows, cols)), shape=(len(dim_rows), m)).tocsr()
    b_ub = np.array([rhs[k] for k in dim_rows])

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=(0.0, None), method="highs-ds")
    if not res.success or res.x is None:
        # fall back to the cheapest feasible candidate per question
        x = np.zeros(m)
        for q in qids:
            cands = [(j, cd) for j, cd in enumerate(candidates) if cd.qid == q]
            j = min(cands, key=lambda t: (float(np.sum(t[1].cost)), -t[1].value))[0]
            x[j] = 1.0
        spend = _spend(candidates, x)
        value = float(sum(cd.value * xv for cd, xv in zip(candidates, x)))
        return Allocation(x, value, spend, np.zeros(D), np.zeros(n), qids,
                          candidates, status=str(res.status))

    x = np.asarray(res.x, dtype=float)
    duals = np.zeros(D)
    marg_ub = getattr(res, "ineqlin", None)
    if marg_ub is not None and marg_ub.marginals is not None:
        for r, k in enumerate(dim_rows):
            duals[k] = max(0.0, -float(marg_ub.marginals[r]))
    omega = np.zeros(n)
    marg_eq = getattr(res, "eqlin", None)
    if marg_eq is not None and marg_eq.marginals is not None:
        omega = -np.asarray(marg_eq.marginals, dtype=float)

    return Allocation(x, float(-res.fun), _spend(candidates, x), duals, omega,
                      qids, candidates)


def _spend(candidates: list[Candidate], x: np.ndarray) -> np.ndarray:
    out = np.zeros(D)
    for cd, xv in zip(candidates, x):
        if xv:
            out += cd.cost * xv
    return out


def greedy(candidates: list[Candidate], budget: np.ndarray,
           duals: np.ndarray | None = None) -> Allocation:
    """Run continuations in decreasing order of gain per priced cost until the
    budget is spent.  Used only when the allocation program is switched off."""
    by_q: dict[str, list[Candidate]] = {}
    for cd in candidates:
        by_q.setdefault(cd.qid, []).append(cd)
    qids = sorted(by_q)

    lam = np.zeros(D) if duals is None else np.asarray(duals, dtype=float).copy()
    if not lam.any():                                   # price each dimension by its share
        b = np.asarray(budget, dtype=float)
        lam = np.where(b > 0, 1.0 / np.maximum(b, 1e-12), 0.0)

    chosen: dict[str, Candidate] = {}
    for q, cs in by_q.items():
        chosen[q] = next((c for c in cs if c.is_stop),
                         min(cs, key=lambda c: float(np.sum(c.cost))))

    ranked: list[tuple[float, float, Candidate]] = []
    for q, cs in by_q.items():
        base = chosen[q].value
        for c in cs:
            if c.is_stop or c.value <= base:
                continue
            priced = float(lam @ c.cost)
            gain = c.value - base
            ranked.append((gain / priced if priced > 0 else float("inf"), gain, c))
    ranked.sort(key=lambda t: (-t[0], -t[1]))

    spend = np.zeros(D)
    taken: set[str] = set()
    for _, _, c in ranked:
        if c.qid in taken:
            continue
        if np.all(spend + c.cost <= np.asarray(budget, dtype=float) + 1e-12):
            spend = spend + c.cost
            chosen[c.qid] = c
            taken.add(c.qid)

    x = np.array([1.0 if chosen[cd.qid] is cd else 0.0 for cd in candidates])
    value = float(sum(cd.value * xv for cd, xv in zip(candidates, x)))
    return Allocation(x, value, _spend(candidates, x), np.zeros(D),
                      np.zeros(len(qids)), qids, candidates, status="greedy")
