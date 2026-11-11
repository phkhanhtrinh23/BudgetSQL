"""The tools the agent acts through.

The agent decides which few candidates each cell deserves, which cells to probe
and how much to hold back; the tools do the arithmetic, which language models do
poorly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import DIMS, zeros
from ..grammar import parse_stage, parse_workflow, shape

TOOL_SPEC = """
compose(cell, spec)   register a workflow as a candidate for a digest cell
                      (cell is its alias, e.g. "c03").  spec is a chain of
                      stages, "esc:L" or "ver:M,esc:L", or a stage whose
                      observations continue differently, written as the
                      digest writes candidates: "ver:M[doubt>esc:L]".
                      Stages: gen:<tier>[x<samples>][@<context>], exec[:<samples>],
                      rep:<tier>, ver:<tier>, esc:<tier>, ctx:prune1|prune2.
imagine(cell, spec)   predicted success and cost of a candidate on that cell.
allocate(reserve)     split the remaining budget over all registered candidates,
                      holding back one turn's worst case for later planning;
                      returns predicted accuracy, spend per dimension and the
                      dual prices.  reserve is a further share of money held back.
probe(cell, stage, n) run one stage on n questions of a cell to observe it.
run(k)                execute the next k planned continuations under the meter.
"""


@dataclass
class ToolResult:
    ok: bool
    data: dict

    def __str__(self) -> str:  # pragma: no cover - display only
        body = ", ".join(f"{k}={_fmt(v)}" for k, v in self.data.items())
        return ("ok " if self.ok else "rejected ") + body


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, tuple, np.ndarray)):
        return "[" + ", ".join(_fmt(x) for x in np.asarray(v).tolist()) + "]"
    return str(v)


class ToolBox:
    """Bound to one batch: a digest cell may stand for several planner cells,
    possibly of different budget groups, and each tool acts on all of them."""

    def __init__(self, target):
        from .batch import BatchPlanner
        if isinstance(target, BatchPlanner):
            self.batch = target
        elif getattr(target, "batch", None) is not None:
            self.batch = target.batch
        else:
            self.batch = BatchPlanner([target], target.cfg, target.agent)
        self.planners = self.batch.planners
        #: the digest the agent is reading this turn
        self.digest = None

    @property
    def planner(self):
        """The planner of a single-group batch."""
        return self.planners[0]

    def _members(self, cell: str) -> list[tuple[object, str]]:
        """The (planner, planner cell) pairs a digest cell stands for."""
        members = [cell]
        digest = self.digest
        if digest is None:
            try:
                digest = self.batch.digest()
            except Exception:
                digest = None
        got = None if digest is None else digest.cell(cell)
        if got is not None:
            members = list(got.members)
        out = []
        for m in members:
            for p in self.planners:
                if p.meter is not None and p.owns(m):
                    out.append((p, m))
        return out

    # ---- compose / imagine ----
    def compose(self, cell: str, spec: str) -> ToolResult:
        try:
            wf = parse_workflow(spec)
        except Exception as exc:
            return ToolResult(False, {"error": str(exc)})
        ok = False
        for p, m in self._members(cell):
            ok = p.register(m, wf) or ok
        return ToolResult(ok, {"cell": cell, "spec": spec,
                               "id": shape(wf) if ok else "ill-formed"})

    def imagine(self, cell: str, spec: str) -> ToolResult:
        try:
            wf = parse_workflow(spec)
        except Exception as exc:
            return ToolResult(False, {"error": str(exc)})
        rolls, n = [], 0
        for p, m in self._members(cell):
            recs = [r for r in p.records.values()
                    if r.cell == m and not r.closed
                    and p.grammar.is_wellformed(wf, r.state)]
            n += len(recs)
            sample = recs[:16]
            if sample:
                rolls += p.model.rollout_many(
                    [(r.meta, r.state, wf) for r in sample], [r.s_held for r in sample])
        if not rolls:
            return ToolResult(False, {"error": "no open question of that cell can run it"})
        p_mean = float(np.mean([r.p for r in rolls]))
        cost = np.mean([r.cost for r in rolls], axis=0)
        return ToolResult(True, {
            "cell": cell, "spec": spec, "n": n, "p": p_mean,
            "spread": float(np.std([r.p for r in rolls])),
            **{k: float(v) for k, v in zip(DIMS, cost)},
            "uncertain": any(r.uncertain for r in rolls),
        })

    # ---- allocate ----
    def allocate(self, reserve: float = 0.0) -> ToolResult:
        """Solve the allocation program in every group with the planning reserve r = rho_turn
        held back while the agent can still be called, plus ``reserve`` (a
        share of the money) if the agent asks."""
        value, n = 0.0, 0
        spend, duals = zeros(), zeros()
        split = 0
        for p in self.planners:
            if not p.is_open:
                continue
            res = zeros()
            res[0] = float(reserve) * float(p.meter.budget[0])
            alloc = p._solve_continuations(reserve=p._planning_reserve() + res)
            if alloc is None:
                continue
            p._last_alloc = alloc
            value += alloc.value
            n += len(alloc.qids)
            spend = spend + alloc.spend
            duals = duals + p.turn_share * alloc.duals
            split += len(alloc.split_questions())
        if not n:
            return ToolResult(False, {"error": "nothing open"})
        return ToolResult(True, {
            "predicted_accuracy": value / n,
            **{f"spend_{k}": float(v) for k, v in zip(DIMS, spend)},
            **{f"price_{k}": float(v) for k, v in zip(DIMS, duals)},
            "binding": ",".join(k for k, v in zip(DIMS, duals) if v > 1e-12) or "none",
            "split": split,
        })

    # ---- probe / run ----
    def probe(self, cell: str, stage: str, n: int = 8) -> ToolResult:
        try:
            st = parse_stage(stage)
        except Exception as exc:
            return ToolResult(False, {"error": str(exc)})
        out = {"probed": 0, "observations": {}, "cost": [0.0] * len(DIMS)}
        left = max(0, int(n))
        for p, m in self._members(cell):
            if left <= 0:
                break
            got = p.probe(m, st, left)
            left -= got["probed"]
            out["probed"] += got["probed"]
            for k, v in got["observations"].items():
                out["observations"][k] = out["observations"].get(k, 0) + v
            out["cost"] = [a + b for a, b in zip(out["cost"], got["cost"])]
        return ToolResult(out["probed"] > 0, out)

    def run(self, k: int = 25) -> ToolResult:
        """Execute the next ``k`` planned continuations of every group."""
        from . import reserve as reserve_mod
        ran = 0
        for p in self.planners:
            alloc = p._last_alloc or p.resolve()
            if alloc is None:
                continue
            order = reserve_mod.run_order(alloc)[: max(0, int(k))]
            p._run_block(order)
            ran += len(order)
        if not ran:
            return ToolResult(False, {"error": "nothing open"})
        return ToolResult(True, {"ran": ran})

    # ---- dispatch for a text-driven agent ----
    def call(self, name: str, args: dict) -> ToolResult:
        fn = {"compose": self.compose, "imagine": self.imagine,
              "allocate": self.allocate, "probe": self.probe,
              "run": self.run}.get(name)
        if fn is None:
            return ToolResult(False, {"error": f"unknown tool {name!r}"})
        try:
            return fn(**args)
        except TypeError as exc:
            return ToolResult(False, {"error": str(exc)})
