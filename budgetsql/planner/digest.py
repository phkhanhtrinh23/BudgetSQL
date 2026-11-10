"""The fixed-size summary of a batch that the agent reads.

The agent never reads the questions.  They are grouped into cells by database,
by what their first stage observed and by deciles of predicted success and
predicted cost; each cell reports its size, the predictions for its current
candidates with their spread, and the drift between its predicted and realised
costs.  A cell is flagged when that drift, or the shift of its observation
shares from their predictions, exceeds the drift threshold.  The digest also
carries the remaining budget, what planning has spent so far and the dual
prices of the last allocation, which say which dimension binds.

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
    #: The planner cells a merged digest cell stands for; the tools act on them.
    members: list[str] = field(default_factory=list)
    #: The short name the agent reads and the tools take (``c01``).
    alias: str = ""

    def __post_init__(self) -> None:
        if not self.members:
            self.members = [self.cid]

    @property
    def size(self) -> int:
        return len(self.qids)

    def line(self) -> str:
        """One line per cell: its alias and flags, database, first-stage
        observation and size, the held answers' predicted success with its
        spread, the cost drift, and its best candidates as compact chains
        with predicted success and money (m$), tokens and seconds."""
        cand = " ".join(
            f"{c.wf_id}:{c.p:.2f}/{c.cost[0] * 1e3:.2f}/{c.cost[1]:.0f}/{c.cost[2]:.1f}"
            + ("?" if c.uncertain else "")
            for c in self.candidates[:3]) or "-"
        flags = ("!" if self.flagged else "") + ("?" if self.uncertain else "")
        return (f"{self.alias or self.cid}{flags} {self.db} {self.obs} n={self.size} "
                f"s={self.s_mean:.2f}~{self.s_spread:.2f} "
                f"d={self.drift[0]:+.0%}/{self.drift[1]:+.0%}/{self.drift[2]:+.0%} "
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
        """A cell by its alias or its id."""
        for c in self.cells:
            if c.alias == cid or c.cid == cid:
                return c
        return None

    def fit(self, max_tokens: int, max_cells: int, count=None) -> "Digest":
        """The digest the agent reads: at most ``max_cells`` cells, flagged
        cells first, each named by a short alias, and merged further until
        the text fits ``max_tokens``, so the digest's size, and so a turn's
        prompt, does not grow with the batch.  No cell is dropped."""
        count = count or approx_tokens
        cells = merge_cells(list(self.cells), max_cells)
        while True:
            cells = sorted(cells, key=lambda c: (not c.flagged, c.db, c.obs, -c.size))
            for i, c in enumerate(cells):
                c.alias = f"c{i + 1:02d}"
            out = Digest(cells, self.remaining, self.spent, self.planning, self.duals,
                         self.budget, self.n_open, self.turns_used, self.turns_left)
            if len(cells) <= 1 or count(out.to_text()) <= max_tokens:
                return out
            cells = merge_cells(cells, len(cells) - 1)

    @property
    def flagged(self) -> list[DigestCell]:
        return [c for c in self.cells if c.flagged]

    def binding(self) -> list[str]:
        return [k for k, v in zip(DIMS, self.duals) if v > 0]

    def to_text(self) -> str:
        """The digest as the agent reads it (``fit`` bounds its size)."""
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
            "cells: alias(! flagged, ? thin support) db obs n s=mean~spread "
            "d=cost drift $/tok/s | candidate:p/m$/tok/s",
        ]
        return "\n".join(head + ["  " + c.line() for c in self.cells])


def approx_tokens(text: str) -> int:
    """A conservative token count (three characters a token, where tokenizers
    average about four), so a reservation made from it covers the prompt."""
    return -(-len(text) // 3)


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
        members=a.members + b.members,
    )
    out.alias = a.alias
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
