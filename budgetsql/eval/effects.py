from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import BUDGETS, PlannerConfig

COMPONENTS = ("composition", "probing", "replanning")


@dataclass
class Effect:
    mean: float = 0.0
    var: float = 0.0

    def as_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var}


def applies(cfg: PlannerConfig) -> bool:
    return bool(cfg.use_agent and cfg.agent_mode == "modelled")


@dataclass
class ModelledEffects:
    """Per named budget vector, each component's effect."""

    by_vector: dict[str, dict[str, Effect]] = field(default_factory=dict)

    def total(self, name: str) -> Effect:
        comps = self.by_vector.get(name, {})
        return Effect(float(sum(e.mean for e in comps.values())),
                      float(sum(e.var for e in comps.values())))

    def at(self, budget: str | None = None, vector=None) -> Effect:
        if not self.by_vector:
            return Effect()
        if vector is None:
            if budget in self.by_vector:
                return self.total(budget)
            if budget in BUDGETS:
                vector = BUDGETS[budget]
            else:
                return Effect()
        v = np.log(np.maximum(np.asarray(vector, dtype=float), 1e-12))
        known = [(n, np.log(np.maximum(BUDGETS[n], 1e-12)))
                 for n in self.by_vector if n in BUDGETS]
        if not known:
            return Effect()
        dist = np.array([float(np.linalg.norm(v - k)) for _, k in known])
        if np.any(dist < 1e-9):
            return self.total(known[int(np.argmin(dist))][0])
        w = 1.0 / dist ** 2
        w = w / w.sum()
        totals = [self.total(n) for n, _ in known]
        return Effect(float(sum(wi * t.mean for wi, t in zip(w, totals))),
                      float(sum(wi ** 2 * t.var for wi, t in zip(w, totals))))

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {b: {c: e.as_dict() for c, e in comps.items()}
             for b, comps in self.by_vector.items()}, indent=2))

    @staticmethod
    def load(path: str | Path) -> "ModelledEffects":
        raw = json.loads(Path(path).read_text())
        return ModelledEffects({b: {c: Effect(float(e.get("mean", 0.0)),
                                              float(e.get("var", 0.0)))
                                    for c, e in comps.items()}
                                for b, comps in raw.items()})


def estimate(bench, budgets: list[str] | None = None, runs: int | None = None,
             agent_factory=None) -> ModelledEffects:
    from ..planner import PlanningAgent
    from .stats import paired_contrast

    make = (agent_factory or bench.agent_factory
            or (lambda probe: PlanningAgent(probe_cells=probe)))
    cfg = bench.cfg
    budgets = budgets or list(bench.workload.budgets)
    live = cfg.with_(agent_mode="live")
    arms = [("composition", live.with_(max_sessions=1), lambda: make(False)),
            ("probing", live.with_(max_sessions=1), lambda: make(True)),
            ("replanning", live, lambda: make(True))]
    out = ModelledEffects()
    for b in budgets:
        prev = bench.run_method("budgetsql", b, runs, cfg=cfg.with_(agent_mode="modelled"),
                                with_effects=False)["runs"]
        comps: dict[str, Effect] = {}
        for name, arm_cfg, agent in arms:
            res = bench.run_method("budgetsql", b, runs, cfg=arm_cfg,
                                   agent_factory=agent)["runs"]
            c = paired_contrast(res, prev)
            comps[name] = Effect(c.mean, c.standard_error ** 2)
            prev = res
        out.by_vector[b] = comps
    return out
