"""Budget dimensions, model prices, budget vectors and planner defaults."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

#: Budget dimensions, in order. Money is USD, tokens are prompt+completion,
#: service time is worker-pool busy seconds.
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


#: Executor and planner list prices (USD / 1M tokens).
PRICES: dict[str, Price] = {
    "S": Price(0.10, 0.20),
    "M": Price(0.40, 1.60),
    "L": Price(2.00, 8.00),
    "agent": Price(1.25, 10.00),
}


@dataclass(frozen=True)
class ExecutorLimits:
    max_completion_tokens: int = 600
    timeout_s: float = 10.0
    sql_timeout_s: float = 10.0


@dataclass(frozen=True)
class PlannerConfig:
    """Planner settings.  The defaults are what :class:`budgetsql.planner.BudgetSQL`
    uses when nothing is overridden."""

    # world model
    min_support: int = 30          # (a, o) pairs below this back off to the pooled frequency
    prior_calls: float = 20.0      # pseudo-calls nu of the online cost correction
    n_estimators: int = 150
    num_leaves: int = 15
    learning_rate: float = 0.05
    log_scale_costs: bool = False  # ablation: regress log cost and exponentiate

    # allocation / adaptation
    replan_block: int = 25         # K continuations between re-solves
    drift_threshold: float = 0.15  # relative cost drift that flags a digest cell
    #: Miss probability of the Bernstein reserve.  Left unset, the reserve
    #: follows the newsvendor condition at the plan's own gain per priced
    #: unit, which is what a hard meter leaves it to do.
    reserve_delta: float | None = None
    use_allocation_lp: bool = True # ablation: greedy ranking instead of the LP
    periodic_resolves: bool = True
    priced_dims: tuple[str, ...] = DIMS  # ablation: ("usd",) prices money alone

    # agent
    max_digest_cells: int = 40     # G_max
    digest_tokens: int = 1400
    latent_tokens: int = 64
    turns_per_session: int = 6
    planning_cap: float = 0.10     # share of the money budget planning may take
    max_candidates: int = 8        # W_max per question
    probe_questions: int = 8
    use_agent: bool = True
    #: Score continuations from the question alone, ignoring what the first
    #: stage actually returned.
    score_from_empty: bool = False
    #: Give every question its complete workflow before anything runs.
    plan_whole_workflows: bool = False

    # world-model inputs (ablations)
    use_question_features: bool = True
    use_execution_features: bool = True
    use_differential_features: bool = True
    use_rollouts: bool = True
    use_stacking: bool = True
    online_cost_updates: bool = True

    limits: ExecutorLimits = field(default_factory=ExecutorLimits)
    seed: int = 0

    def with_(self, **kw) -> "PlannerConfig":
        return replace(self, **kw)

    def training_key(self) -> tuple:
        """The settings a fitted world model depends on.  Everything else is a
        planning choice, so arms that differ only in those reuse one fit."""
        return (self.min_support, self.n_estimators, self.num_leaves,
                self.learning_rate, self.log_scale_costs,
                self.use_question_features, self.use_execution_features,
                self.use_differential_features)


#: Per-question budget vectors: money in USD, prompt+completion tokens, and
#: service seconds.  A dimension a vector constrains gets a little more than the
#: reference workflow's mean cost in it; every other dimension gets several
#: times that, so it does not bind.
BUDGETS: dict[str, np.ndarray] = {
    "V1": vec(0.8e-3, 6000, 8.0),   # money
    "V2": vec(1.0e-3, 6000, 2.0),   # money + service time
    "V3": vec(1.0e-3, 1300, 8.0),   # money + tokens
    "V4": vec(0.8e-3, 1500, 2.2),   # all three
    "SP": vec(11.0e-3, 10500, 81.0),  # Spider 2.0-Lite
}

#: The workflow whose mean cost the budget vectors are set against.
REFERENCE_WORKFLOW = "M+R_L"


@dataclass(frozen=True)
class Workload:
    name: str
    budget_groups: str = "db"   # "db" or "fold"
    folds: int = 1
    runs: int = 5
    budgets: tuple[str, ...] = ("V1", "V2", "V3", "V4")


WORKLOADS: dict[str, Workload] = {
    "bird": Workload("bird", budget_groups="db", runs=5,
                     budgets=("V1", "V2", "V3", "V4")),
    "spider2_lite": Workload("spider2_lite", budget_groups="fold", folds=10,
                             runs=5, budgets=("SP",)),
}

BOOTSTRAP_RESAMPLES = 4000


def budget_for_group(name: str, n_questions: int) -> np.ndarray:
    """A budget group's vector: the per-question vector times its questions."""
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
