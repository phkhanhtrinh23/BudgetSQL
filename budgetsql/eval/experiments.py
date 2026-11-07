"""The comparisons this package is built to run.

Each function takes a workbench and returns a plain dictionary, so results can
be written to JSON and rendered later.
"""
from __future__ import annotations

import random

import numpy as np
from sklearn.metrics import roc_auc_score

from ..analysis.escalation import frontier, stats as esc_stats
from ..baselines import BASELINES
from ..baselines.agents import ProfiledPerRequest
from ..config import DIMS, PlannerConfig, budget_for_group
from ..grammar import State, parse_chain
from ..planner import Meter
from .runner import BudgetGroup, Workbench
from .stats import accuracy, lead, paired_contrast

PLANNER = "budgetsql"


# --------------------------------------------------------------------------- #
# accuracy at matched budgets
# --------------------------------------------------------------------------- #
def main_table(bench: Workbench, methods: list[str] | None = None,
               budgets: list[str] | None = None, runs: int | None = None) -> dict:
    methods = methods or list(BASELINES)
    budgets = budgets or list(bench.workload.budgets)
    unit_of = bench.unit_of()
    out: dict = {"budgets": {}, "methods": [PLANNER] + methods}
    for b in budgets:
        ours = bench.run_method(PLANNER, b, runs)
        rows: dict[str, dict] = {}
        for name in methods:
            res = bench.run_method(name, b, runs)
            acc, spread = accuracy(res["runs"])
            rows[name] = {"accuracy": acc, "spread": spread,
                          "lead_over": lead(ours["runs"], res["runs"], unit_of).as_dict(),
                          "utilisation": _mean_util(res["details"])}
        acc, spread = accuracy(ours["runs"])
        best = max(rows, key=lambda n: rows[n]["accuracy"]) if rows else None
        out["budgets"][b] = {
            "planner": {"accuracy": acc, "spread": spread,
                        "utilisation": _mean_util(ours["details"]),
                        "planning_share": _mean_planning(ours["details"]),
                        "binding_fraction": _mean_binding(ours["details"]),
                        "turns": _mean_key(ours["details"], "turns")},
            "baselines": rows,
            "strongest_baseline": best,
            "lead": rows[best]["lead_over"] if best else None,
        }
    return out


def budget_sweep(bench: Workbench, methods: list[str] | None = None,
                 scales: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
                 base: str = "V1", runs: int = 1) -> dict:
    """Accuracy against the money budget, the other dimensions left slack."""
    methods = methods or ["best_fixed_mixture", "cascade_routing", "uab_style",
                          "llm_agent"]
    out: dict = {"base": base, "scales": list(scales), "curves": {}}
    for name in [PLANNER] + methods:
        curve = []
        for s in scales:
            res = bench.run_method(name, base, runs, budget_scale=s)
            acc, _ = accuracy(res["runs"])
            curve.append(acc)
        out["curves"][name] = curve
    return out


# --------------------------------------------------------------------------- #
# ablations
# --------------------------------------------------------------------------- #
def ablation_arms(cfg: PlannerConfig) -> dict[str, object]:
    """Each arm is either a configuration of the planner or another method."""
    return {
        "profiled_averages_no_world_model": ProfiledPerRequest,
        "full_data_heads_no_stacks": cfg.with_(use_stacking=False),
        "flat_world_model_no_rollouts": cfg.with_(use_rollouts=False),
        "direct_heads_only_no_stack": cfg.with_(use_rollouts=False, use_stacking=False),
        "no_differential_features": cfg.with_(use_differential_features=False),
        "scored_before_first_stage": cfg.with_(score_from_empty=True),
        "log_scale_cost_heads": cfg.with_(log_scale_costs=True),
        "edits_only_no_agent": cfg.with_(use_agent=False),
        "whole_workflows_before_running": cfg.with_(plan_whole_workflows=True,
                                                    use_agent=False),
        "money_only_prices": cfg.with_(priced_dims=("usd",)),
        "greedy_ranking_no_allocation": cfg.with_(use_allocation_lp=False),
        "opening_plan_no_resolves": cfg.with_(periodic_resolves=False),
        "world_model_question_only": cfg.with_(use_execution_features=False),
        "world_model_execution_only": cfg.with_(use_question_features=False),
        "no_online_cost_updates": cfg.with_(online_cost_updates=False),
    }


def ablation(bench: Workbench, budgets: list[str] | None = None,
             arms: dict | None = None, runs: int | None = None) -> dict:
    budgets = budgets or list(bench.workload.budgets)
    arms = arms if arms is not None else ablation_arms(bench.cfg)
    out: dict = {"budgets": {}}
    for b in budgets:
        full = bench.run_method(PLANNER, b, runs)
        acc, _ = accuracy(full["runs"])
        rows: dict[str, dict] = {}
        for name, arm in arms.items():
            if isinstance(arm, PlannerConfig):
                res = bench.run_method(PLANNER, b, runs, cfg=arm)
            else:
                res = _run_other(bench, arm, b, runs)
            contrast = paired_contrast(res["runs"], full["runs"])
            a, _ = accuracy(res["runs"])
            rows[name] = {"accuracy": a, "change": contrast.mean,
                          "half_width": contrast.half_width,
                          "excludes_zero": contrast.excludes_zero}
        out["budgets"][b] = {"planner_accuracy": acc, "arms": rows}
    return out


def _run_other(bench: Workbench, cls, budget: str, runs: int | None) -> dict:
    runs = runs or bench.workload.runs
    per_run = []
    details = []
    for r in range(runs):
        correct: dict[str, bool] = {}
        for group in bench.groups():
            metas = list(group.metas)
            random.Random((r + 1) * 7919).shuffle(metas)
            method = cls(name=cls.__name__, cfg=bench.cfg, library=bench.library)
            method.prepare(bench.corpus, bench.train_dbs(group))
            b = budget_for_group(budget, group.size)
            res = method.run(metas, b, Meter(b))
            correct.update(res.correct)
            details.append({"run": r, "group": group.name, **res.as_dict()})
        per_run.append(correct)
    return {"method": cls.__name__, "budget": budget, "runs": per_run,
            "details": details}


# --------------------------------------------------------------------------- #
# sensitivity
# --------------------------------------------------------------------------- #
def sensitivity(bench: Workbench, budget: str = "V4", runs: int | None = None
                ) -> dict:
    cfg = bench.cfg
    grid = {
        "drift_threshold": [0.05, 0.15, 0.40],
        "replan_block": [5, 25, 100],
        "planning_cap": [0.02, 0.07, 0.10, 0.20],
    }
    out: dict = {"budget": budget, "settings": {}}
    for key, values in grid.items():
        rows = []
        for v in values:
            res = bench.run_method(PLANNER, budget, runs, cfg=cfg.with_(**{key: v}))
            acc, _ = accuracy(res["runs"])
            rows.append({"value": v, "accuracy": acc,
                         "turns": _mean_key(res["details"], "turns"),
                         "planning_share": _mean_planning(res["details"])})
        out["settings"][key] = rows
    return out


# --------------------------------------------------------------------------- #
# world-model quality and online cost updates
# --------------------------------------------------------------------------- #
def world_model_quality(bench: Workbench, openers: tuple[str, ...] = ("S", "M")
                        ) -> dict:
    """How well the model ranks questions before and after a stage has run, how
    well it detects fixes and regressions, and how far its cost estimates are
    off on a database it has never seen."""
    out: dict = {"openers": {}, "cost_error": {}}
    cont = parse_chain("esc:L")
    for opener in openers:
        pre, post, real = [], [], []
        fix_p, fix_y, reg_p, reg_y = [], [], [], []
        cont_p, cont_y = [], []
        mre: dict[str, list[float]] = {k: [] for k in DIMS}
        for group in bench.groups():
            values = bench.values(group)
            model = values.model
            wf = bench.library[opener]
            states, oks, costs = [], [], []
            for meta in group.metas:
                state, ok, cost, _ = bench.corpus.run(meta, State(), wf)
                states.append(state); oks.append(ok); costs.append(cost)
            posts = model.s_of_many(list(zip(group.metas, states)))
            rolls = model.rollout_many(
                [(m, State(), wf) for m in group.metas], [0.0] * len(group.metas))
            for meta, state, ok, cost, roll, sp in zip(
                    group.metas, states, oks, costs, rolls, posts):
                pre.append(roll.p)
                post.append(sp)
                real.append(int(ok))
                for k, p, c in zip(DIMS, roll.cost, cost):
                    if c > 0:
                        mre[k].append(abs(p - c) / c)
                if not bench.corpus.grammar.is_wellformed(cont, state):
                    continue
                s_held = sp
                res = model.predict(meta, state, cont.stage).resolve(s_held)
                _, ok2, _, outs = bench.corpus.run(meta, state, cont, held=ok)
                cont_p.append(model.rollout(meta, state, cont, s_held).p)
                cont_y.append(int(ok2))
                if outs:
                    o = outs[0]
                    try:
                        j = res.index(o.obs)
                    except ValueError:
                        j = 0
                    fix_p.append(float(res.fix[j])); fix_y.append(int(o.is_fix))
                    reg_p.append(float(res.reg[j])); reg_y.append(int(o.is_regression))
        out["openers"][opener] = {
            "auroc_before": _auc(real, pre),
            "auroc_after": _auc(real, post),
            "auroc_fix": _auc(fix_y, fix_p),
            "auroc_regression": _auc(reg_y, reg_p),
            "auroc_continuation": _auc(cont_y, cont_p),
            "brier_after": float(np.mean((np.array(post) - np.array(real)) ** 2)),
        }
        out["cost_error"][opener] = {k: float(np.median(v)) if v else 0.0
                                     for k, v in mre.items()}
    return out


def online_learning(bench: Workbench, days: int = 5, day_n: int = 300,
                    budget: str = "V4", seed: int = 0) -> dict:
    """Daily batches drawn from databases the world model has not seen, with and
    without the online cost corrections."""
    rng = random.Random(seed)
    groups = bench.groups()
    out: dict = {"days": [], "budget": budget}
    arms = {"updated": bench.cfg.with_(online_cost_updates=True),
            "frozen": bench.cfg.with_(online_cost_updates=False)}
    # each arm keeps its own cost corrections across the days
    planners = {k: bench.planner(groups[0], c) for k, c in arms.items()}
    assert planners["updated"].model is not planners["frozen"].model
    pool = [m for g in groups[1:] for m in g.metas] or list(groups[0].metas)
    for day in range(days):
        batch = rng.sample(pool, min(day_n, len(pool)))
        row: dict = {"day": day + 1}
        for k, planner in planners.items():
            b = budget_for_group(budget, len(batch))
            res = planner.run(batch, b, Meter(b, planning_cap=bench.cfg.planning_cap))
            row[k] = {"accuracy": res.accuracy * 100.0,
                      "cancelled": res.cancelled,
                      "cost_error_usd": _median_kappa(planner)}
        row["gain"] = row["updated"]["accuracy"] - row["frozen"]["accuracy"]
        out["days"].append(row)
    return out


def _median_kappa(planner) -> float:
    vals = [abs(v - 1.0) for (_, _, k), v in planner.model._kappa.items() if k == 0]
    return float(np.median(vals)) if vals else 0.0


# --------------------------------------------------------------------------- #
# planning overhead and batch size
# --------------------------------------------------------------------------- #
def overhead(bench: Workbench, budgets: list[str] | None = None,
             batch_sizes: tuple[int, ...] = (150, 500), runs: int = 1) -> dict:
    budgets = budgets or list(bench.workload.budgets)
    out: dict = {"budgets": {}, "batches": []}
    for b in budgets:
        res = bench.run_method(PLANNER, b, runs)
        out["budgets"][b] = {
            "planning_share": _mean_planning(res["details"]),
            "turns": _mean_key(res["details"], "turns"),
            "sessions": _mean_key(res["details"], "sessions"),
            "wall_seconds": _mean_key(res["details"], "wall_seconds"),
        }
    groups = bench.groups()
    for n in batch_sizes:
        # take as many budget groups as the batch needs and hold all of them out,
        # so the batch is planned by a model that has seen none of its schemas
        taken, pool = [], []
        for g in groups:
            taken.append(g)
            pool.extend(g.metas)
            if len(pool) >= n:
                break
        if len(pool) < n:
            continue
        batch = random.Random(n).sample(pool, n)
        held = BudgetGroup(f"batch{n}", [d for g in taken for d in g.dbs], batch)
        b = budget_for_group(budgets[0], n)
        planner = bench.planner(held)
        res = planner.run(batch, b, Meter(b, planning_cap=bench.cfg.planning_cap))
        out["batches"].append({"n": n, "accuracy": res.accuracy * 100.0,
                               "turns": res.turns,
                               "planning_share": res.planning_share})
    return out


# --------------------------------------------------------------------------- #
# where the gain from escalating comes from
# --------------------------------------------------------------------------- #
def trace_frontier(bench: Workbench, pairs=(("M", "L"), ("S", "L"), ("S", "M"))
                   ) -> dict:
    """The held-out gain of escalation rules that see only the question, only
    the executed stage's status, or both."""
    corpus = bench.corpus
    lib = bench.library
    metas = corpus.questions
    out: dict = {"pairs": {}}
    for keep, esc in pairs:
        y_keep, y_esc, status, qbin, dbs = [], [], [], [], []
        for m in metas:
            state, ok, _, _ = corpus.run(m, State(), lib[keep])
            _, ok2, _, _ = corpus.run(m, State(), lib[esc])
            y_keep.append(int(ok)); y_esc.append(int(ok2))
            status.append((state.answer_obs or "-").split(".")[0])
            qbin.append(_question_bin(m))
            dbs.append(m.db)
        y_keep = np.array(y_keep); y_esc = np.array(y_esc)
        dbs = np.array(dbs)
        rows: dict[str, dict] = {}
        for label, bins in (("question", np.array(qbin)),
                            ("status", np.array(status)),
                            ("status+question", np.array([f"{a}|{b}" for a, b in
                                                          zip(status, qbin)]))):
            folds = []
            for db in sorted(set(dbs.tolist())):
                test = np.where(dbs == db)[0]
                train = np.where(dbs != db)[0]
                if len(test) == 0 or len(train) == 0:
                    continue
                folds.append(frontier(bins, y_keep, y_esc, train, test))
            if folds:
                rows[label] = {
                    "predicted": float(np.mean([f["predicted"] for f in folds]) * 100),
                    "realised": float(np.mean([f["realised"] for f in folds]) * 100),
                    "standard_error": float(np.mean([f["standard_error"] for f in folds]) * 100),
                }
        st = esc_stats(y_keep, y_esc)
        out["pairs"][f"{keep}->{esc}"] = {
            **{k: v * 100 if k in ("fix_rate", "regression_rate", "m",
                                   "ceiling_matched", "ceiling_unconstrained")
               else v for k, v in st.as_dict().items()},
            "rules": rows,
        }
    return out


def _question_bin(meta) -> str:
    size = "big" if meta.n_schema_tokens > 700 else "small"
    length = "long" if meta.n_question_tokens > 25 else "short"
    return f"{size}.{length}.{meta.difficulty or 'na'}"


def predicted_versus_replayed(bench: Workbench, budgets: list[str] | None = None,
                              runs: int = 1) -> dict:
    """The opening plan's predicted value against what the run realises, with
    and without the held-out stacks."""
    budgets = budgets or list(bench.workload.budgets)
    out: dict = {"budgets": {}}
    for b in budgets:
        row = {}
        for label, cfg in (("stacked", bench.cfg.with_(use_agent=False)),
                           ("full_data_heads", bench.cfg.with_(use_agent=False,
                                                               use_stacking=False))):
            res = bench.run_method(PLANNER, b, runs, cfg=cfg)
            acc, _ = accuracy(res["runs"])
            pred = 100.0 * _mean_key(res["details"], "predicted_accuracy")
            row[label] = {"predicted": pred, "replayed": acc, "gap": pred - acc}
        out["budgets"][b] = row
    return out


# --------------------------------------------------------------------------- #
# deployment scenarios
# --------------------------------------------------------------------------- #
def scenarios(bench: Workbench, budget: str = "V4", runs: int = 1) -> dict:
    """The planner against the strongest baseline with open-weight executors,
    an open-weight agent, or both."""
    from ..config import PRICES, Price
    unit_of = bench.unit_of()
    presets = {
        "api_agent_api_executors": {},
        "api_agent_open_executors": {"S": Price(0.05, 0.08), "M": Price(0.12, 0.20),
                                     "L": Price(0.30, 0.50)},
        "open_agent_api_executors": {"agent": Price(0.20, 0.30)},
        "open_agent_open_executors": {"S": Price(0.05, 0.08), "M": Price(0.12, 0.20),
                                      "L": Price(0.30, 0.50),
                                      "agent": Price(0.20, 0.30)},
    }
    out: dict = {"budget": budget, "configurations": {}}
    original = dict(PRICES)
    try:
        for name, override in presets.items():
            PRICES.update(original)
            PRICES.update(override)
            ours = bench.run_method(PLANNER, budget, runs)
            best_name, best_runs, best_acc = None, None, -1.0
            for b in ("cascade_routing", "uab_style", "best_fixed_mixture"):
                res = bench.run_method(b, budget, runs)
                acc, _ = accuracy(res["runs"])
                if acc > best_acc:
                    best_name, best_runs, best_acc = b, res["runs"], acc
            acc, _ = accuracy(ours["runs"])
            out["configurations"][name] = {
                "accuracy": acc, "strongest_baseline": best_name,
                "baseline_accuracy": best_acc,
                "lead": lead(ours["runs"], best_runs, unit_of).as_dict(),
            }
    finally:
        PRICES.update(original)
    return out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _auc(y, p) -> float:
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    if len(y) < 4 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _mean_key(details: list[dict], key: str) -> float:
    vals = [d.get(key, 0.0) for d in details if key in d]
    return float(np.mean(vals)) if vals else 0.0


def _mean_util(details: list[dict]) -> dict:
    out = {}
    for k in DIMS:
        vals = [d["utilisation"][k] for d in details if "utilisation" in d]
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out


def _mean_planning(details: list[dict]) -> dict:
    out = {}
    for k in DIMS:
        vals = [d["planning_share"].get(k, 0.0) for d in details
                if "planning_share" in d]
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out


def _mean_binding(details: list[dict]) -> dict:
    out = {}
    for k in DIMS:
        vals = [d["binding_fraction"].get(k, 0.0) for d in details
                if "binding_fraction" in d]
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out
