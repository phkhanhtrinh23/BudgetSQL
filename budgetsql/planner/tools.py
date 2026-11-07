"""The tools the agent acts through.

The agent decides which few candidates each cell deserves, which cells to probe
and how much to hold back; the tools do the arithmetic, which language models do
poorly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import DIMS, zeros
from ..grammar import parse_chain, parse_stage, shape

TOOL_SPEC = """
compose(cell, spec)   register a workflow as a candidate for a digest cell.
                      spec is a comma-separated chain of stages, for example
                      "esc:L" or "ver:M,esc:L" or "rep:L".
                      Stages: gen:<tier>[x<samples>][@<context>], exec[:<samples>],
                      rep:<tier>, ver:<tier>, esc:<tier>, ctx:prune1|prune2.
imagine(cell, spec)   predicted success and cost of a candidate on that cell.
allocate(reserve)     split the remaining budget over all registered candidates;
                      returns predicted accuracy, spend per dimension and the
                      dual prices.  reserve is the share of money held back.
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
    """Bound to one planner and one budget group."""

    def __init__(self, planner):
        self.planner = planner

    # ---- compose / imagine ----
    def compose(self, cell: str, spec: str) -> ToolResult:
        try:
            wf = parse_chain(spec)
        except Exception as exc:
            return ToolResult(False, {"error": str(exc)})
        ok = self.planner.register(cell, wf)
        return ToolResult(ok, {"cell": cell, "spec": spec,
                               "id": shape(wf) if ok else "ill-formed"})

    def imagine(self, cell: str, spec: str) -> ToolResult:
        try:
            wf = parse_chain(spec)
        except Exception as exc:
            return ToolResult(False, {"error": str(exc)})
        recs = [r for r in self.planner.records.values()
                if r.cell == cell and not r.closed]
        recs = [r for r in recs if self.planner.grammar.is_wellformed(wf, r.state)]
        if not recs:
            return ToolResult(False, {"error": "no open question of that cell can run it"})
        sample = recs[:16]
        rolls = self.planner.model.rollout_many(
            [(r.meta, r.state, wf) for r in sample], [r.s_held for r in sample])
        p = float(np.mean([r.p for r in rolls]))
        cost = np.mean([r.cost for r in rolls], axis=0)
        return ToolResult(True, {
            "cell": cell, "spec": spec, "n": len(recs), "p": p,
            "spread": float(np.std([r.p for r in rolls])),
            **{k: float(v) for k, v in zip(DIMS, cost)},
            "uncertain": any(r.uncertain for r in rolls),
        })

    # ---- allocate ----
    def allocate(self, reserve: float = 0.0) -> ToolResult:
        res = zeros()
        res[0] = float(reserve)
        alloc = self.planner._solve_continuations(
            reserve=self.planner._turn_worst_case() + res)
        if alloc is None:
            return ToolResult(False, {"error": "nothing open"})
        self.planner._last_alloc = alloc
        n = max(1, len(alloc.qids))
        return ToolResult(True, {
            "predicted_accuracy": alloc.value / n,
            **{f"spend_{k}": float(v) for k, v in zip(DIMS, alloc.spend)},
            **{f"price_{k}": float(v) for k, v in zip(DIMS, alloc.duals)},
            "binding": ",".join(alloc.binding()) or "none",
            "split": len(alloc.split_questions()),
        })

    # ---- probe / run ----
    def probe(self, cell: str, stage: str, n: int = 8) -> ToolResult:
        try:
            st = parse_stage(stage)
        except Exception as exc:
            return ToolResult(False, {"error": str(exc)})
        out = self.planner.probe(cell, st, n)
        return ToolResult(out["probed"] > 0, out)

    def run(self, k: int = 25) -> ToolResult:
        from . import reserve as reserve_mod
        alloc = self.planner._last_alloc or self.planner._solve_continuations()
        if alloc is None:
            return ToolResult(False, {"error": "nothing open"})
        order = reserve_mod.run_order(alloc)[: max(0, int(k))]
        self.planner._run_block(order)
        return ToolResult(True, {"ran": len(order)})

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
