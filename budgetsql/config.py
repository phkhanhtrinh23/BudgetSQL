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


#: The completion cap a stage carries unless it asks for less, as ``verify``
#: does.  A workload's :class:`ExecutorLimits` replaces it.
DEFAULT_COMPLETION = 600


@dataclass(frozen=True)
class ExecutorLimits:
    max_completion_tokens: int = DEFAULT_COMPLETION
    #: Timeout of one LLM call (one sample).
    timeout_s: float = 10.0
    #: Timeout of one SQL statement.
    sql_timeout_s: float = 10.0
    #: Time limit of a local Context stage (schema retrieval, no LLM call).
    context_timeout_s: float = 1.0
    #: Completion cap of the large tier, where it reasons before answering.
    #: ``None`` gives it the same cap as the others.
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

    # world model
    min_support: int = 30          # (a, o) pairs below this back off to the pooled frequency
    prior_calls: float = 20.0      # pseudo-calls nu of the online cost correction
    n_estimators: int = 150
    num_leaves: int = 15
    learning_rate: float = 0.05
    log_scale_costs: bool = False  # ablation: regress log cost and exponentiate

    # allocation / adaptation
    replan_block: int = 25         # K continuations between re-solves
    #: Relative cost drift, and shift of observation shares, that flags a
    #: digest cell.
    drift_threshold: float = 0.15
    #: Hold the cost reserve rho of Eq. 8 back from the program's right-hand
    #: side.  Off in the replay of Sec. VII-A, where rho = 0 and the meter
    #: alone enforces the budget; the ablation "with a cost reserve" turns it on.
    cost_reserve: bool = False
    #: Miss probability of the Bernstein reserve (the log(d/delta) variant of
    #: Prop. 3).  Left unset, the reserve follows the newsvendor condition at
    #: the block's own gain per priced unit.
    reserve_delta: float | None = None
    #: A reserve of ``reserve_z`` predicted standard deviations instead of the
    #: Bernstein bound (the ablation uses one standard deviation).
    reserve_z: float | None = None
    use_allocation_lp: bool = True # ablation: greedy ranking instead of the LP
    #: Re-solve every K continuations.  Off, the continuation program is solved
    #: once after the opening session and its whole plan is run.
    periodic_resolves: bool = True
    priced_dims: tuple[str, ...] = DIMS  # ablation: ("usd",) prices money alone

    # agent
    max_digest_cells: int = 40     # G_max
    digest_tokens: int = 1400
    latent_tokens: int = 64
    #: Completion cap and timeout of one agent turn: with the digest they give
    #: the turn's worst case rho_turn, known before the turn.
    agent_completion_tokens: int = 400
    agent_timeout_s: float = 6.0
    #: The results of the agent's previous tool calls it reads with the next
    #: digest, capped so the turn's prompt stays bounded.
    agent_result_tokens: int = 200
    turns_per_session: int = 6
    max_sessions: int = 11
    planning_cap: float = 0.10     # share of the money budget planning may take
    max_candidates: int = 8        # W_max per question
    probe_questions: int = 8
    use_agent: bool = True
    #: How the agent's turns enter a run.  ``"modelled"`` is the replay of
    #: Sec. VII-A: the certificate decides when a turn would be taken, the turn
    #: is charged to the budget, and its accuracy effect enters the evaluation
    #: as a modelled, additive per-vector effect (``eval.effects``).
    #: ``"live"`` runs the agent's tool calls and charges what the turn cost.
    agent_mode: str = "modelled"
    #: Cost charged for a modelled turn, or for a live turn whose agent does
    #: not report what it cost; ``None`` charges the turn's worst case.
    modelled_turn_cost: tuple[float, float, float] | None = None
    #: Take turns without reserving or charging them, so a run measures what
    #: the turns add apart from their cost (the derived agent rows).
    free_turns: bool = False
    #: Downgrade a question to the cheapest feasible call when the remaining
    #: budget could not cover the later questions (the baselines' pacing rule).
    pacing: bool = False
    #: Depth of the certificate's search over the grammar; ``None`` searches
    #: every composition the grammar admits from a state, as Eq. 7 states.
    certificate_depth: int | None = None
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
    #: Select (Sec. IV-B): keep a replacing stage's new answer iff f-hat > r-hat.
    #: Off, a new answer is always kept (the planners without a world model).
    use_select: bool = True
    #: Online cost updates of Eq. 5.  Off in the replay until Sec. VII-D.
    online_cost_updates: bool = False
    #: Within-batch recalibration of the success heads from observed execution
    #: rates (Sec. VI), part of the online updates of Sec. VII-D.
    recalibrate_success: bool = False

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
                self.use_differential_features,
                # the stacks are fitted on what the planner reads (rollouts or
                # the direct head) and on outcomes realised under Select
                self.use_rollouts, self.use_select)


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
    n_questions: int = 1530
    n_dbs: int = 11
    limits: ExecutorLimits = field(default_factory=ExecutorLimits)
    #: The model each tier names on this workload.  Spider 2.0-Lite needs a
    #: reasoning model as the large tier to answer anything.
    models: dict[str, str] = field(default_factory=lambda: {
        "S": "qwen2.5-7b-instruct", "M": "gpt-4.1-mini", "L": "gpt-4.1"})
    #: List prices of the tiers this workload's models change (USD / 1M
    #: tokens); the meter's worst cases and simulated costs read them.
    prices: dict[str, Price] = field(default_factory=dict)


#: The executor prices of the default workload, before any is applied.
BASE_PRICES: dict[str, Price] = dict(PRICES)


def use_workload_prices(workload: "Workload") -> None:
    """Set the tier prices of a workload (BIRD's list prices, or those of the
    models a workload names instead, as GPT-5 for Spider 2.0-Lite's large
    tier)."""
    for tier in TIERS:
        PRICES[tier] = workload.prices.get(tier, BASE_PRICES[tier])


#: Spider 2.0-Lite prompts are several times as long as BIRD's and its large
#: tier reasons, so both caps and the timeout are larger.
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

#: The axes the budget grid crosses: money in USD, tokens and service seconds
#: per question.  Money crosses time and money crosses tokens, 25 cells each.
GRID_AXES: dict[str, tuple[float, ...]] = {
    "usd": (0.6e-3, 0.8e-3, 1.0e-3, 1.25e-3, 1.6e-3),
    "tok": (1000.0, 1300.0, 1800.0, 2600.0, 6000.0),
    "sec": (1.6, 2.0, 2.6, 3.4, 8.0),
}

#: The slack a dimension the grid does not vary receives.
GRID_SLACK: dict[str, float] = {"usd": 4.0e-3, "tok": 20000.0, "sec": 40.0}

#: Money per question of the money sweep (Fig. 3(a)), tokens and time slack.
SWEEP_MONEY: tuple[float, ...] = (0.3e-3, 0.5e-3, 0.8e-3, 1.2e-3, 2.0e-3)


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
