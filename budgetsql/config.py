from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

DIMS: tuple[str, ...] = ("usd", "tok", "sec")
D = len(DIMS)

TIERS: tuple[str, ...] = ("S", "M", "L")


def zeros() -> np.ndarray:
    return np.zeros(D, dtype=float)


def vec(usd: float = 0.0, tok: float = 0.0, sec: float = 0.0) -> np.ndarray:
    return np.array([usd, tok, sec], dtype=float)


def as_dict(v) -> dict[str, float]:
    return {k: float(x) for k, x in zip(DIMS, np.asarray(v, dtype=float))}


@dataclass(frozen=True)
class Price:
    """List price in USD per million tokens."""

    inp: float
    out: float

    def usd(self, prompt_tokens: float, completion_tokens: float) -> float:
        return (prompt_tokens * self.inp + completion_tokens * self.out) * 1e-6


PRICES: dict[str, Price] = {
    "S": Price(0.10, 0.20),
    "M": Price(0.40, 1.60),
    "L": Price(2.00, 8.00),
    "agent": Price(1.25, 10.00),
}


DEFAULT_COMPLETION = 600


@dataclass(frozen=True)
class ExecutorLimits:
    max_completion_tokens: int = DEFAULT_COMPLETION
    timeout_s: float = 10.0
    sql_timeout_s: float = 10.0
    context_timeout_s: float = 1.0
    large_completion_tokens: int | None = None

    def cap_for(self, tier: str | None = None, stage_cap: int | None = None) -> int:
        if stage_cap is not None and stage_cap < DEFAULT_COMPLETION:
            return int(stage_cap)
        if tier == "L" and self.large_completion_tokens:
            return int(self.large_completion_tokens)
        return int(self.max_completion_tokens)


@dataclass(frozen=True)
class PlannerConfig:
    """Planner settings.  The defaults are what :class:`budgetsql.planner.BudgetSQL`
    uses when nothing is overridden."""

    min_support: int = 30
    prior_calls: float = 20.0
    n_estimators: int = 150
    num_leaves: int = 15
    learning_rate: float = 0.05
    log_scale_costs: bool = False

    replan_block: int = 25
    drift_threshold: float = 0.15
    cost_reserve: bool = False
    reserve_delta: float | None = None
    reserve_z: float | None = None
    use_allocation_lp: bool = True
    periodic_resolves: bool = True
    priced_dims: tuple[str, ...] = DIMS

    max_digest_cells: int = 40
    digest_tokens: int = 1400
    latent_tokens: int = 64
    agent_completion_tokens: int = 400
    agent_timeout_s: float = 6.0
    agent_result_tokens: int = 200
    turns_per_session: int = 6
    max_sessions: int = 11
    planning_cap: float = 0.10
    max_candidates: int = 8
    probe_questions: int = 8
    use_agent: bool = True
    agent_mode: str = "modelled"
    modelled_turn_cost: tuple[float, float, float] | None = None
    free_turns: bool = False
    pacing: bool = False
    certificate_depth: int | None = None
    score_from_empty: bool = False
    plan_whole_workflows: bool = False

    use_question_features: bool = True
    use_execution_features: bool = True
    use_differential_features: bool = True
    use_rollouts: bool = True
    use_stacking: bool = True
    use_select: bool = True
    online_cost_updates: bool = False
    recalibrate_success: bool = False

    limits: ExecutorLimits = field(default_factory=ExecutorLimits)
    seed: int = 0

    def with_(self, **kw) -> "PlannerConfig":
        return replace(self, **kw)

    def training_key(self) -> tuple:
        return (self.min_support, self.n_estimators, self.num_leaves,
                self.learning_rate, self.log_scale_costs,
                self.use_question_features, self.use_execution_features,
                self.use_differential_features,
                self.use_rollouts, self.use_select)


BUDGETS: dict[str, np.ndarray] = {
    "V1": vec(0.8e-3, 6000, 8.0),
    "V2": vec(1.0e-3, 6000, 2.0),
    "V3": vec(1.0e-3, 1300, 8.0),
    "V4": vec(0.8e-3, 1500, 2.2),
    "SP": vec(11.0e-3, 10500, 81.0),
}

REFERENCE_WORKFLOW = "M+R_L"


@dataclass(frozen=True)
class Workload:
    name: str
    budget_groups: str = "db"
    folds: int = 1
    runs: int = 5
    budgets: tuple[str, ...] = ("V1", "V2", "V3", "V4")
    n_questions: int = 1530
    n_dbs: int = 11
    limits: ExecutorLimits = field(default_factory=ExecutorLimits)
    models: dict[str, str] = field(default_factory=lambda: {
        "S": "qwen2.5-7b-instruct", "M": "gpt-4.1-mini", "L": "gpt-4.1"})
    prices: dict[str, Price] = field(default_factory=dict)


BASE_PRICES: dict[str, Price] = dict(PRICES)


def use_workload_prices(workload: "Workload") -> None:
    for tier in TIERS:
        PRICES[tier] = workload.prices.get(tier, BASE_PRICES[tier])


SPIDER_LIMITS = ExecutorLimits(max_completion_tokens=1600, timeout_s=120.0,
                               sql_timeout_s=120.0,
                               large_completion_tokens=8000)

WORKLOADS: dict[str, Workload] = {
    "bird": Workload("bird", budget_groups="db", runs=5,
                     budgets=("V1", "V2", "V3", "V4"),
                     n_questions=1530, n_dbs=11),
    "spider2_lite": Workload("spider2_lite", budget_groups="fold", folds=10,
                             runs=5, budgets=("SP",),
                             n_questions=135, n_dbs=30, limits=SPIDER_LIMITS,
                             models={"S": "qwen2.5-7b-instruct",
                                     "M": "gpt-4.1-mini", "L": "gpt-5"},
                             prices={"L": Price(1.25, 10.00)}),
}

BOOTSTRAP_RESAMPLES = 4000

GRID_AXES: dict[str, tuple[float, ...]] = {
    "usd": (0.6e-3, 0.8e-3, 1.0e-3, 1.25e-3, 1.6e-3),
    "tok": (1000.0, 1300.0, 1800.0, 2600.0, 6000.0),
    "sec": (1.6, 2.0, 2.6, 3.4, 8.0),
}

GRID_SLACK: dict[str, float] = {"usd": 4.0e-3, "tok": 20000.0, "sec": 40.0}

SWEEP_MONEY: tuple[float, ...] = (0.3e-3, 0.5e-3, 0.8e-3, 1.2e-3, 2.0e-3)


def budget_for_group(name: str, n_questions: int) -> np.ndarray:
    return BUDGETS[name] * float(n_questions)


def dump_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def enc(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=enc, sort_keys=True)
