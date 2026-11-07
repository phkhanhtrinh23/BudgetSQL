"""The agents that drive the tools.

Two are provided.  The planning agent is a deliberate policy over the digest: it
spends its turns where the digest says a cell is mispriced, thinly supported or
drifting, and leaves every budget decision to the allocation tool.  The chat
agent hands the same digest and the same tools to a language model.  Both see
only the digest, never the questions.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol


from .digest import Digest, DigestCell
from .tools import TOOL_SPEC, ToolBox

SYSTEM = """You plan a batch of Text-to-SQL questions under a hard budget on \
money, tokens and service time.  One budget pays for answering and for your own \
turns.  You never see the questions, only a digest of cells.

Tools:
{tools}
Reply with one JSON object per line: {{"tool": <name>, "args": {{...}}}}.
Register a few candidates for the cells where they can pay, probe a cell whose \
predictions are marked uncertain or whose spread is wide, then allocate.  Prefer \
cheap stages in the dimension the dual prices say binds.  Reply with nothing \
else."""


class Agent(Protocol):
    def turn(self, digest: Digest, tools: ToolBox) -> None:
        ...


# --------------------------------------------------------------------------- #
# the planning agent
# --------------------------------------------------------------------------- #
@dataclass
class PlanningAgent:
    """Composes candidates where the digest says they could pay.

    The candidates it adds beyond the one-stage edits are escalations and
    repairs by a larger tier, a verified escalation, and a compressed context
    where tokens carry the price.
    """

    probe_questions: int = 8
    max_compose: int = 4
    reserve: float = 0.0
    log: list[str] = field(default_factory=list)

    def turn(self, digest: Digest, tools: ToolBox) -> None:
        binding = digest.binding()
        cells = sorted(digest.cells, key=self._priority, reverse=True)
        for cell in cells[: self.max_compose]:
            for spec in self._specs(cell, binding):
                res = tools.compose(cell.cid, spec)
                self.log.append(f"compose {cell.cid} {spec} -> {res.ok}")
        probe = next((c for c in cells if c.uncertain or c.s_spread > 0.2), None)
        if probe is not None:
            res = tools.probe(probe.cid, "exec", self.probe_questions)
            self.log.append(f"probe {probe.cid} -> {res.ok}")
        out = tools.allocate(self.reserve)
        self.log.append(f"allocate -> {out}")

    @staticmethod
    def _priority(cell: DigestCell) -> float:
        head = {"error": 1.0, "empty": 0.9}.get(cell.obs.split(".")[0], 0.0)
        split = 0.4 if cell.obs.endswith("split") else 0.0
        return (cell.size ** 0.5) * (1.0 + head + split
                                     + (0.5 if cell.flagged else 0.0)
                                     + (0.3 if cell.uncertain else 0.0)
                                     + (1.0 - cell.s_mean))

    @staticmethod
    def _specs(cell: DigestCell, binding: list[str]) -> list[str]:
        head = cell.obs.split(".")[0]
        out: list[str] = []
        if head in ("error", "empty"):
            out += ["rep:L", "rep:M"]
        else:
            out += ["esc:L"]
            if "tok" not in binding and "sec" not in binding:
                out += ["ver:M,esc:L"]
        if "tok" in binding:
            out = [s for s in out if "ver:" not in s]
        return out[:3]


# --------------------------------------------------------------------------- #
# the chat agent
# --------------------------------------------------------------------------- #
@dataclass
class ChatAgent:
    """Hands the digest and the tools to a language model."""

    client: object                      # anything with .chat(messages, **kw)
    max_calls_per_turn: int = 8
    temperature: float = 0.0
    max_tokens: int = 400
    log: list[str] = field(default_factory=list)

    def turn(self, digest: Digest, tools: ToolBox) -> None:
        messages = [
            {"role": "system", "content": SYSTEM.format(tools=TOOL_SPEC)},
            {"role": "user", "content": digest.to_text()},
        ]
        reply = self.client.chat(messages, temperature=self.temperature,
                                 max_tokens=self.max_tokens)
        text = reply["text"] if isinstance(reply, dict) else str(reply)
        for name, args in parse_calls(text)[: self.max_calls_per_turn]:
            res = tools.call(name, args)
            self.log.append(f"{name}({args}) -> {res}")


def parse_calls(text: str) -> list[tuple[str, dict]]:
    """Read ``{"tool": ..., "args": {...}}`` objects out of a reply."""
    out: list[tuple[str, dict]] = []
    for raw in re.findall(r"\{.*?\}\s*\}|\{[^{}]*\}", text, flags=re.S):
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        name = obj.get("tool") or obj.get("name")
        args = obj.get("args") or obj.get("arguments") or {}
        if isinstance(name, str) and isinstance(args, dict):
            out.append((name, args))
    return out


# --------------------------------------------------------------------------- #
# a tracker-only agent, for the baseline that plans without a world model
# --------------------------------------------------------------------------- #
@dataclass
class BudgetTrackerAgent:
    """Plans from the remaining budget alone: no predictions per question, so
    it composes one escalation for the cells with the largest share of the batch
    and otherwise keeps what the first stage returned."""

    log: list[str] = field(default_factory=list)

    def turn(self, digest: Digest, tools: ToolBox) -> None:
        left = float(digest.remaining[0]) / max(1.0, float(digest.budget[0]))
        cells = sorted(digest.cells, key=lambda c: -c.size)
        spec = "esc:L" if left > 0.5 else "esc:M"
        for cell in cells[:3]:
            tools.compose(cell.cid, spec)
        tools.allocate(0.0)
