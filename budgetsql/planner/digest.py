"""The fixed-size summary of a batch that the agent reads.

The agent never reads the questions.  They are grouped into cells by database,
by what their first stage observed and by deciles of predicted success and cost;
each cell reports its size, the predictions for its current candidates with
their spread, and the drift between its predicted and realised costs.  The
digest also carries the remaining budget, what planning has spent so far and the
dual prices of the last allocation, which say which dimension binds.

The number of cells is capped, so the digest's size does not grow with the
batch: while there are too many, two cells are merged --- within a database when
one has several, otherwise the two smallest.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import D, DIMS


@dataclass
class CellCandidate:
    wf_id: str
    p: float
    cost: np.ndarray
    spread: float = 0.0
    uncertain: bool = False


@dataclass
class DigestCell:
    cid: str
    db: str
    obs: str
    qids: list[str] = field(default_factory=list)
    s_mean: float = 0.0
    s_spread: float = 0.0
    cost_mean: np.ndarray = field(default_factory=lambda: np.zeros(D))
    drift: np.ndarray = field(default_factory=lambda: np.zeros(D))
    uncertain: bool = False
    flagged: bool = False
    candidates: list[CellCandidate] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.qids)

    def line(self) -> str:
        cand = "; ".join(
            f"{c.wf_id} p={c.p:.2f} "
            f"${c.cost[0] * 1e3:.2f}/k tok={c.cost[1]:.0f} s={c.cost[2]:.1f}"
            + ("?" if c.uncertain else "")
            for c in self.candidates[:4]) or "none"
        flags = ("!" if self.flagged else "") + ("?" if self.uncertain else "")
        return (f"{self.cid}{flags} db={self.db} obs={self.obs} n={self.size} "
                f"s={self.s_mean:.2f}+-{self.s_spread:.2f} "
                f"drift={self.drift[0]:+.0%}/{self.drift[1]:+.0%}/{self.drift[2]:+.0%} "
                f"| {cand}")


@dataclass
class Digest:
    cells: list[DigestCell]
    remaining: np.ndarray
    spent: np.ndarray
    planning: np.ndarray
    duals: np.ndarray
    budget: np.ndarray
    n_open: int = 0
    turns_used: int = 0
    turns_left: int = 0

    def cell(self, cid: str) -> DigestCell | None:
        for c in self.cells:
            if c.cid == cid:
                return c
        return None

    @property
    def flagged(self) -> list[DigestCell]:
        return [c for c in self.cells if c.flagged]

    def binding(self) -> list[str]:
        return [k for k, v in zip(DIMS, self.duals) if v > 0]

    def to_text(self) -> str:
        head = [
            "BATCH DIGEST",
            f"open questions: {self.n_open} in {len(self.cells)} cells",
            "budget left: " + ", ".join(
                f"{k}={r:.4g} of {b:.4g}" for k, r, b in zip(DIMS, self.remaining, self.budget)),
            "planning spent: " + ", ".join(
                f"{k}={p:.4g}" for k, p in zip(DIMS, self.planning)),
            "dual prices: " + ", ".join(f"{k}={v:.4g}" for k, v in zip(DIMS, self.duals)),
            f"binding: {', '.join(self.binding()) or 'none'}",
            f"turns: {self.turns_used} used, {self.turns_left} left",
            "cells (! flagged, ? thin support):",
        ]
        return "\n".join(head + ["  " + c.line() for c in self.cells])


def merge_cells(cells: list[DigestCell], max_cells: int) -> list[DigestCell]:
    """Merge until the digest fits.  Each merge removes one cell, so this ends."""
    cells = list(cells)
    while len(cells) > max_cells:
        by_db: dict[str, list[DigestCell]] = {}
        for c in cells:
            by_db.setdefault(c.db, []).append(c)
        multi = {db: cs for db, cs in by_db.items() if len(cs) > 1}
        if multi:
            # the smallest cell among databases that have several, merged with
            # the cell of its own database whose predicted success is nearest
            a = min((c for cs in multi.values() for c in cs), key=lambda c: c.size)
            siblings = [c for c in multi[a.db] if c is not a]
            b = min(siblings, key=lambda c: abs(c.s_mean - a.s_mean))
        else:
            order = sorted(cells, key=lambda c: c.size)
            a, b = order[0], order[1]
        cells = [c for c in cells if c is not a and c is not b]
        cells.append(_merge(a, b))
    return sorted(cells, key=lambda c: (c.db, c.obs, -c.size))


def _merge(a: DigestCell, b: DigestCell) -> DigestCell:
    n = a.size + b.size
    wa, wb = (a.size / n, b.size / n) if n else (0.5, 0.5)
    obs = a.obs if a.obs == b.obs else "mixed"
    db = a.db if a.db == b.db else "mixed"
    out = DigestCell(
        cid=f"{a.cid}+{b.cid}" if len(a.cid) + len(b.cid) < 18 else a.cid,
        db=db, obs=obs, qids=a.qids + b.qids,
        s_mean=wa * a.s_mean + wb * b.s_mean,
        s_spread=max(a.s_spread, b.s_spread) + abs(a.s_mean - b.s_mean) / 2.0,
        cost_mean=wa * a.cost_mean + wb * b.cost_mean,
        drift=wa * a.drift + wb * b.drift,
        uncertain=a.uncertain or b.uncertain,
        flagged=a.flagged or b.flagged,
    )
    seen: dict[str, CellCandidate] = {}
    for c in a.candidates + b.candidates:
        got = seen.get(c.wf_id)
        if got is None:
            seen[c.wf_id] = CellCandidate(c.wf_id, c.p, c.cost.copy(), c.spread, c.uncertain)
        else:
            got.p = 0.5 * (got.p + c.p)
            got.cost = 0.5 * (got.cost + c.cost)
            got.spread = max(got.spread, c.spread)
            got.uncertain = got.uncertain or c.uncertain
    out.candidates = sorted(seen.values(), key=lambda c: -c.p)
    return out
