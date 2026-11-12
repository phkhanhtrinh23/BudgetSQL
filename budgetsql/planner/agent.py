from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol


import numpy as np

from .digest import Digest, DigestCell, approx_tokens
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


MESSAGE_OVERHEAD_TOKENS = 8
PROMPT_OVERHEAD_TOKENS = approx_tokens(SYSTEM.format(tools=TOOL_SPEC)) \
    + 2 * MESSAGE_OVERHEAD_TOKENS + approx_tokens("\nRESULTS OF YOUR LAST TOOL CALLS\n")


class Agent(Protocol):
    def turn(self, digest: Digest, tools: ToolBox) -> np.ndarray | None:
        ...


@dataclass
class PlanningAgent:
    probe_questions: int = 8
    max_compose: int = 4
    reserve: float = 0.0
    probe_cells: bool = True
    log: list[str] = field(default_factory=list)

    def turn(self, digest: Digest, tools: ToolBox) -> None:
        binding = digest.binding()
        cells = sorted(digest.cells, key=self._priority, reverse=True)
        for cell in cells[: self.max_compose]:
            for spec in self._specs(cell, binding):
                res = tools.compose(cell.cid, spec)
                self.log.append(f"compose {cell.cid} {spec} -> {res.ok}")
        probe = next((c for c in cells if c.uncertain or c.s_spread > 0.2), None)
        if self.probe_cells and probe is not None:
            res = tools.probe(probe.cid, "ver:M", self.probe_questions)
            self.log.append(f"probe {probe.cid} -> {res.ok}")
        out = tools.allocate(self.reserve)
        self.log.append(f"allocate -> {out}")
        return None

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


def chat_agent_from_env(probe: bool = True) -> "ChatAgent":
    from ..executors.llm import tiers_from_env
    allowed = ("compose", "imagine", "allocate", "run") + (("probe",) if probe else ())
    return ChatAgent(tiers_from_env()["agent"], allowed=allowed)


@dataclass
class ChatAgent:
    client: object
    max_calls_per_turn: int = 8
    temperature: float = 0.0
    max_tokens: int = 400
    timeout_s: float = 6.0
    allowed: tuple[str, ...] = ("compose", "imagine", "allocate", "probe", "run")
    result_tokens: int = 200
    log: list[str] = field(default_factory=list)
    last_results: str = ""

    def configure(self, cfg) -> None:
        self.result_tokens = int(cfg.agent_result_tokens)
        self.max_tokens = int(cfg.agent_completion_tokens)
        self.timeout_s = float(cfg.agent_timeout_s)

    def messages(self, digest: Digest) -> list[dict]:
        user = digest.to_text()
        if self.last_results:
            user += "\nRESULTS OF YOUR LAST TOOL CALLS\n" + self.last_results
        return [{"role": "system", "content": SYSTEM.format(tools=TOOL_SPEC)},
                {"role": "user", "content": user}]

    def turn(self, digest: Digest, tools: ToolBox) -> np.ndarray | None:
        messages = self.messages(digest)
        reply = self.client.chat(messages, temperature=self.temperature,
                                 max_tokens=self.max_tokens, timeout=self.timeout_s,
                                 retries=1)
        text = reply["text"] if isinstance(reply, dict) else str(reply)
        results = []
        for name, args in parse_calls(text)[: self.max_calls_per_turn]:
            if name not in self.allowed:
                results.append(f"{name}({args}) -> not allowed")
                continue
            res = tools.call(name, args)
            results.append(f"{name}({args}) -> {res}")
        self.log += results
        self.last_results = "\n".join(results)[: 3 * self.result_tokens]
        if isinstance(reply, dict) and reply.get("ok", True):
            return np.array([float(reply.get("usd", 0.0)),
                             float(reply.get("prompt_tokens", 0))
                             + float(reply.get("completion_tokens", 0)),
                             float(reply.get("latency", 0.0))])
        return None


def parse_calls(text: str) -> list[tuple[str, dict]]:
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


@dataclass
class BudgetTrackerAgent:
    log: list[str] = field(default_factory=list)

    def turn(self, digest: Digest, tools: ToolBox) -> None:
        left = float(digest.remaining[0]) / max(1e-12, float(digest.budget[0]))
        cells = sorted(digest.cells, key=lambda c: -c.size)
        spec = "esc:L" if left > 0.5 else "esc:M"
        for cell in cells[:3]:
            tools.compose(cell.cid, spec)
        tools.allocate(0.0)
