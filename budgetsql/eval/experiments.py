"""The comparisons this package is built to run.

Each function takes a workbench and returns a plain dictionary, so results can
be written to JSON and rendered later.
"""
from __future__ import annotations

import random
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from ..analysis.escalation import frontier, stats as esc_stats
from ..baselines import BASELINES
from ..baselines.agents import ProfiledPerRequest
from ..config import (DIMS, GRID_AXES, GRID_SLACK, PlannerConfig,
                      budget_for_group)
from ..config import zeros
from ..grammar import State, parse_chain
from ..planner import Meter
from ..planner.allocation import Candidate, solve
from .runner import BudgetGroup, Workbench
from .stats import accuracy, lead, paired_contrast

PLANNER = "budgetsql"


# --------------------------------------------------------------------------- #
# accuracy at matched budgets
# --------------------------------------------------------------------------- #
BIRD_ONLY = ("best_fixed_workflow",)


def default_methods(bench: Workbench) -> list[str]:
    """Every baseline the workload admits.  One fixed workflow per database is
    a BIRD row, because a Spider fold spans several databases."""
    if bench.workload.budget_groups == "db":
        return list(BASELINES)
    return [m for m in BASELINES if m not in BIRD_ONLY]


def main_table(bench: Workbench, methods: list[str] | None = None,
               budgets: list[str] | None = None, runs: int | None = None,
               oracle: bool = True) -> dict:
    methods = methods or default_methods(bench)
    if oracle:
        methods = methods + ["oracle"]
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
        ranked = {n: r for n, r in rows.items() if n != "oracle"}
        best = max(ranked, key=lambda n: ranked[n]["accuracy"]) if ranked else None
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


def budget_grid(bench: Workbench, pairs: tuple[tuple[str, str], ...] =
                (("usd", "sec"), ("usd", "tok")), runs: int = 3,
                methods: list[str] | None = None) -> dict:
    """The lead when two budget dimensions vary and the third is slack."""
    methods = methods or default_methods(bench)
    unit_of = bench.unit_of()
    out: dict = {"axes": {k: list(v) for k, v in GRID_AXES.items()}, "pairs": {}}
    for x, y in pairs:
        cells = []
        for vx in GRID_AXES[x]:
            for vy in GRID_AXES[y]:
                vector = np.array([GRID_SLACK[k] for k in DIMS], dtype=float)
                vector[DIMS.index(x)] = vx
                vector[DIMS.index(y)] = vy
                ours = bench.run_method(PLANNER, DIMS[0], runs, vector=vector)
                best_name, best_runs, best_acc = None, None, -1.0
                for name in methods:
                    res = bench.run_method(name, DIMS[0], runs, vector=vector)
                    acc, _ = accuracy(res["runs"])
                    if acc > best_acc:
                        best_name, best_runs, best_acc = name, res["runs"], acc
                acc, _ = accuracy(ours["runs"])
                ld = lead(ours["runs"], best_runs, unit_of)
                cells.append({x: vx, y: vy, "accuracy": acc,
                              "strongest_baseline": best_name,
                              "baseline_accuracy": best_acc,
                              "lead": ld.as_dict()})
        leads = [c["lead"]["mean"] for c in cells]
        out["pairs"][f"{x}x{y}"] = {
            "cells": cells, "n": len(cells),
            "min": min(leads) if leads else 0.0,
            "max": max(leads) if leads else 0.0,
            "excludes_zero": sum(1 for c in cells if c["lead"]["excludes_zero"]),
        }
    return out


def pooled_budget(bench: Workbench, budgets: list[str] | None = None,
                  methods: tuple[str, ...] = ("best_fixed_workflow",
                                              "best_fixed_mixture",
                                              "cascade_routing", "uab_style"),
                  runs: int | None = None) -> dict:
    """One budget for the whole batch instead of one per database.

    The planner reads the pooled batch as a single budget group and can move
    slack between databases; a baseline still plans each database on its own,
    under the one shared meter.
    """
    runs = runs or bench.workload.runs
    budgets = budgets or list(bench.workload.budgets)
    group = bench.pooled_group()
    unit_of = bench.unit_of()
    out: dict = {"budgets": {}, "n_questions": group.size}
    for b in budgets:
        budget = budget_for_group(b, group.size)
        ours_runs, ours_details = [], []
        for r in range(runs):
            metas = bench.order_metas(group.metas, r)
            planner = bench.pooled_planner()
            meter = Meter(budget, planning_cap=bench.cfg.planning_cap)
            res = planner.run(metas, budget, meter)
            ours_runs.append(dict(res.correct))
            ours_details.append(res.as_dict())
        rows: dict[str, dict] = {}
        for name in [m for m in methods if m in BASELINES]:
            per_run = []
            for r in range(runs):
                meter = Meter(budget)
                correct: dict[str, bool] = {}
                for sub in bench.groups():
                    method = bench.method(name, sub)
                    model = getattr(method, "model", None)
                    if model is not None:
                        model.reset_online()
                    metas = bench.order_metas(sub.metas, r)
                    correct.update(method.run(metas, meter.remaining, meter).correct)
                per_run.append(correct)
            acc, spread = accuracy(per_run)
            rows[name] = {"accuracy": acc, "spread": spread,
                          "lead_over": lead(ours_runs, per_run, unit_of).as_dict()}
        acc, spread = accuracy(ours_runs)
        best = max(rows, key=lambda n: rows[n]["accuracy"]) if rows else None
        out["budgets"][b] = {
            "planner": {"accuracy": acc, "spread": spread,
                        "utilisation": _mean_util(ours_details)},
            "baselines": rows, "strongest_baseline": best,
            "lead": rows[best]["lead_over"] if best else None,
        }
    return out


def arrival_orders(bench: Workbench, budgets: list[str] | None = None,
                   orders: tuple[str, ...] = ("random", "easy_first", "hard_first"),
                   methods: list[str] | None = None, runs: int | None = None) -> dict:
    """The lead when the questions arrive sorted by difficulty."""
    budgets = budgets or list(bench.workload.budgets)
    methods = methods or default_methods(bench)
    unit_of = bench.unit_of()
    out: dict = {"budgets": {}}
    for b in budgets:
        rows: dict[str, dict] = {}
        for order in orders:
            ours = bench.run_method(PLANNER, b, runs, order=order)
            best_runs, best_name, best_acc = None, None, -1.0
            for name in methods:
                res = bench.run_method(name, b, runs, order=order)
                acc, _ = accuracy(res["runs"])
                if acc > best_acc:
                    best_name, best_runs, best_acc = name, res["runs"], acc
            acc, _ = accuracy(ours["runs"])
            rows[order] = {"accuracy": acc, "strongest_baseline": best_name,
                           "baseline_accuracy": best_acc,
                           "lead": lead(ours["runs"], best_runs, unit_of).as_dict()}
        base = rows["random"]
        for order, row in rows.items():
            row["change_vs_random"] = row["lead"]["mean"] - base["lead"]["mean"]
        out["budgets"][b] = rows
    leads = [r["lead"]["mean"] for d in out["budgets"].values() for r in d.values()]
    out["min_lead"] = min(leads) if leads else 0.0
    out["max_change"] = max((abs(r["change_vs_random"])
                             for d in out["budgets"].values() for r in d.values()),
                            default=0.0)
    return out


def strata(bench: Workbench, budgets: list[str] | None = None,
           methods: tuple[str, ...] = ("best_fixed_workflow", "best_fixed_mixture",
                                       "uab_style"),
           runs: int | None = None) -> dict:
    """Accuracy by BIRD difficulty and by schema size."""
    budgets = budgets or list(bench.workload.budgets)
    label_of = {"difficulty": {}, "schema": {}}
    sizes = sorted(m.n_schema_tokens for g in bench.groups() for m in g.metas)
    lo = sizes[len(sizes) // 3] if sizes else 0
    hi = sizes[2 * len(sizes) // 3] if sizes else 0
    for group in bench.groups():
        for m in group.metas:
            label_of["difficulty"][m.qid] = m.difficulty or "na"
            label_of["schema"][m.qid] = ("small" if m.n_schema_tokens <= lo
                                         else "large" if m.n_schema_tokens > hi
                                         else "medium")
    out: dict = {"budgets": {}}
    for b in budgets:
        ours = bench.run_method(PLANNER, b, runs)
        base = {n: bench.run_method(n, b, runs)["runs"]
                for n in methods if n in BASELINES}
        row: dict = {}
        for kind, labels in label_of.items():
            row[kind] = {}
            for label in sorted(set(labels.values())):
                qids = [q for q, v in labels.items() if v == label]
                ours_acc = _accuracy_on(ours["runs"], qids)
                best = max((_accuracy_on(r, qids) for r in base.values()),
                           default=0.0)
                row[kind][label] = {"accuracy": ours_acc, "baseline": best,
                                    "gain": ours_acc - best,
                                    "share": len(qids) / max(1, len(labels))}
        out["budgets"][b] = row
    return out


def per_database(bench: Workbench, budgets: list[str] | None = None,
                 methods: list[str] | None = None, runs: int | None = None) -> dict:
    """The lead over each budget group's own strongest replayed baseline."""
    budgets = budgets or list(bench.workload.budgets)
    methods = methods or [m for m in default_methods(bench)
                          if not m.startswith("llm_agent")]
    out: dict = {"budgets": {}}
    for b in budgets:
        ours = bench.run_method(PLANNER, b, runs)
        base = {n: bench.run_method(n, b, runs)["runs"] for n in methods}
        rows: dict[str, dict] = {}
        for group in bench.groups():
            qids = [m.qid for m in group.metas]
            ours_acc = _accuracy_on(ours["runs"], qids)
            best_name, best_acc = None, -1.0
            for name, r in base.items():
                acc = _accuracy_on(r, qids)
                if acc > best_acc:
                    best_name, best_acc = name, acc
            rows[group.name] = {"accuracy": ours_acc, "strongest_baseline": best_name,
                                "baseline_accuracy": best_acc,
                                "gain": ours_acc - best_acc}
        out["budgets"][b] = {"groups": rows,
                             "ahead": sum(1 for v in rows.values() if v["gain"] > 0),
                             "units": len(rows)}
    return out


def _accuracy_on(runs: list[dict[str, bool]], qids: list[str]) -> float:
    if not qids:
        return 0.0
    means = [100.0 * np.mean([1.0 if r.get(q) else 0.0 for q in qids])
             for r in runs if r]
    return float(np.mean(means)) if means else 0.0


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
        "cost_reserve_one_sigma": cfg.with_(reserve_z=1.0),
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
        "drift_threshold": [0.05, 0.10, 0.15, 0.25, 0.40],
        "replan_block": [5, 10, 25, 50, 100],
        "planning_cap": [0.02, 0.04, 0.07, 0.10, 0.20],
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
                    budget: str = "V4", seed: int = 0,
                    label_share: float = 0.30) -> dict:
    """Daily batches drawn from databases the world model has not seen.

    Four arms: a never-updated model, the online cost corrections of the
    paper's ratio estimator, those plus within-batch recalibration of the
    correctness predictions, which needs no reference labels, and those plus
    reference labels arriving late for a share of the previous days' answers.
    """
    rng = random.Random(seed)
    groups = bench.groups()
    out: dict = {"days": [], "budget": budget, "label_share": label_share}
    arms = {"frozen": bench.cfg.with_(online_cost_updates=False),
            "updated": bench.cfg.with_(online_cost_updates=True),
            "recalibrated": bench.cfg.with_(online_cost_updates=True),
            "labelled": bench.cfg.with_(online_cost_updates=True)}
    planners = {k: bench.planner(groups[0], c) for k, c in arms.items()}
    assert planners["updated"].model is not planners["frozen"].model
    pool = [m for g in groups[1:] for m in g.metas] or list(groups[0].metas)
    seen: dict[str, list[tuple]] = {k: [] for k in arms}
    for day in range(days):
        batch = rng.sample(pool, min(day_n, len(pool)))
        row: dict = {"day": day + 1}
        for k, planner in planners.items():
            b = budget_for_group(budget, len(batch))
            res = planner.run(batch, b, Meter(b, planning_cap=bench.cfg.planning_cap))
            row[k] = {"accuracy": res.accuracy * 100.0,
                      "cancelled": res.cancelled,
                      "cost_error_usd": _median_kappa(planner)}
            states = [(r.meta, r.state) for r in planner.records.values()]
            if k == "recalibrated":
                _recalibrate(planner, states)
            elif k == "labelled":
                labelled = [(m, st) for m, st in states
                            if rng.random() < label_share]
                seen[k].extend((m, st, planner.records[m.qid].held_correct)
                               for m, st in labelled)
                _calibrate_to_labels(planner, seen[k])
        for k in ("updated", "recalibrated", "labelled"):
            row[f"gain_{k}"] = row[k]["accuracy"] - row["frozen"]["accuracy"]
        row["gain"] = row["gain_updated"]
        out["days"].append(row)
    return out


def _recalibrate(planner, states: list[tuple]) -> None:
    by_db: dict[str, list[tuple]] = {}
    for meta, state in states:
        by_db.setdefault(meta.db, []).append((meta, state))
    for db, items in by_db.items():
        planner.model.recalibrate(db, items)


def _calibrate_to_labels(planner, rows: list[tuple]) -> None:
    by_db: dict[str, list[tuple]] = {}
    for meta, state, ok in rows:
        by_db.setdefault(meta.db, []).append((meta, state, ok))
    for db, items in by_db.items():
        target = float(np.mean([1.0 if ok else 0.0 for _, _, ok in items]))
        planner.model.calibrate_to(db, [(m, st) for m, st, _ in items], target)


def _median_kappa(planner) -> float:
    vals = [abs(v - 1.0) for (_, _, k), v in planner.model._kappa.items() if k == 0]
    return float(np.median(vals)) if vals else 0.0


def calibration(bench: Workbench, budget: str = "V4", bins: int = 10,
                runs: int | None = None) -> dict:
    """Reliability of the continuations the planner selects: its predicted
    correctness against what the replay realised."""
    runs = runs or 1
    pred, real = [], []
    for r in range(runs):
        for group in bench.groups():
            metas = bench.order_metas(group.metas, r)
            b = budget_for_group(budget, group.size)
            planner = bench.planner(group)
            planner.run(metas, b, Meter(b, planning_cap=bench.cfg.planning_cap))
            for rec in planner.records.values():
                pred.append(float(rec.s_held))
                real.append(1.0 if rec.held_correct else 0.0)
    p = np.asarray(pred)
    y = np.asarray(real)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows, ece = [], 0.0
    for i in range(bins):
        sel = (p >= edges[i]) & (p < edges[i + 1] if i + 1 < bins else p <= 1.0)
        if not sel.any():
            continue
        share = float(sel.mean())
        gap = abs(float(p[sel].mean()) - float(y[sel].mean()))
        ece += share * gap
        rows.append({"lo": float(edges[i]), "hi": float(edges[i + 1]),
                     "predicted": float(p[sel].mean()),
                     "realised": float(y[sel].mean()),
                     "n": int(sel.sum()),
                     "standard_error": float(np.sqrt(
                         max(y[sel].mean() * (1 - y[sel].mean()), 1e-12)
                         / sel.sum()))})
    return {"budget": budget, "bins": rows,
            "brier": float(np.mean((p - y) ** 2)) if len(p) else 0.0,
            "ece": float(ece)}


def unprofiled_trace(bench: Workbench, budget: str = "V4", opener: str = "S",
                     continuation: str = "esc:L", runs: int = 1) -> dict:
    """What a world model loses when no question has a whole composition.

    The replay traces every composition on every question, which is what a
    model scoring whole workflows needs.  Thinning the training corpus to one
    composition per question leaves that model without a profile, while the
    stage-level rollouts still compose stages each question did run.  The cost
    in accuracy is read off the measured sensitivity of accuracy to the
    ranking's AUROC.
    """
    cont = parse_chain(continuation)
    arms = {"stage": bench.cfg, "flat": bench.cfg.with_(use_rollouts=False)}
    out: dict = {"budget": budget, "arms": {}}
    for label, cfg in arms.items():
        auc, acc = [], []
        for thin in (False, True):
            y, p = [], []
            for group in bench.groups():
                values = bench.thinned_values(group, cfg) if thin \
                    else bench.values(group, cfg)
                values = values.with_cfg(cfg)
                model = values.model
                wf = bench.library[opener]
                items = []
                for meta in group.metas:
                    state, ok, _, _ = bench.corpus.run(meta, State(), wf)
                    if not bench.corpus.grammar.is_wellformed(cont, state):
                        continue
                    items.append((meta, state, ok))
                if not items:
                    continue
                helds = model.s_of_many([(m, st) for m, st, _ in items])
                rolls = model.rollout_many([(m, st, cont) for m, st, _ in items],
                                           helds)
                vals = values.values([m for m, _, _ in items],
                                     [st for _, st, _ in items],
                                     [cont] * len(items), rolls, helds)
                for (meta, state, ok), v in zip(items, vals):
                    _, ok2, _, _ = bench.corpus.run(meta, state, cont, held=ok)
                    p.append(float(v))
                    y.append(int(ok2))
            auc.append(_auc(y, p))
            res = bench.run_method(PLANNER, budget, runs, cfg=cfg, thinned=thin)
            a, _ = accuracy(res["runs"])
            acc.append(a)
        out["arms"][label] = {"auroc_full_trace": auc[0], "auroc_thinned": auc[1],
                              "accuracy_full_trace": acc[0],
                              "accuracy_thinned": acc[1]}
    rows = [(r["auroc_full_trace"], r["accuracy_full_trace"]) for r in
            out["arms"].values()]
    rows += [(r["auroc_thinned"], r["accuracy_thinned"]) for r in
             out["arms"].values()]
    rows = [(x, y) for x, y in rows if np.isfinite(x)]
    slope = 0.0
    if len(rows) >= 2:
        xs = np.array([x for x, _ in rows])
        ys = np.array([y for _, y in rows])
        if xs.std() > 1e-9:
            slope = float(np.polyfit(xs, ys, 1)[0])
    out["points_per_auroc"] = slope
    best = max((r["auroc_full_trace"] for r in out["arms"].values()
                if np.isfinite(r["auroc_full_trace"])), default=0.0)
    for r in out["arms"].values():
        gap = best - r["auroc_thinned"] if np.isfinite(r["auroc_thinned"]) else 0.0
        r["drop"] = gap * slope
    return out


# --------------------------------------------------------------------------- #
# planning overhead and batch size
# --------------------------------------------------------------------------- #
def overhead(bench: Workbench, budgets: list[str] | None = None,
             batch_sizes: tuple[int, ...] = (150, 500), runs: int = 1,
             sessions: tuple[int, ...] = (0, 4, 11, 30, 60)) -> dict:
    budgets = budgets or list(bench.workload.budgets)
    out: dict = {"budgets": {}, "batches": [], "sessions": []}
    for b in budgets:
        res = bench.run_method(PLANNER, b, runs)
        out["budgets"][b] = {
            "planning_share": _mean_planning(res["details"]),
            "turns": _mean_key(res["details"], "turns"),
            "sessions": _mean_key(res["details"], "sessions"),
            "wall_seconds": _mean_key(res["details"], "wall_seconds"),
        }
    for requested in sessions:
        cfg = bench.cfg.with_(use_agent=requested > 0,
                              max_sessions=max(1, requested))
        res = bench.run_method(PLANNER, budgets[0], runs, cfg=cfg)
        acc, _ = accuracy(res["runs"])
        out["sessions"].append({
            "requested": requested, "accuracy": acc,
            "turns": _mean_key(res["details"], "turns"),
            "planning_share": _mean_planning(res["details"])})

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


def scaling(bench: Workbench, sizes: tuple[int, ...] = (1530, 5000, 15000,
                                                       50000, 150000),
            budget: str = "V1", group_size: int = 139,
            limit_s: float = 60.0) -> dict:
    """The local CPU time planning needs as the batch grows.

    A batch resampled from the workload is scored once, its digest built, and
    the allocation solved per budget group of ``group_size`` questions and once
    for the whole batch.  The whole-batch program is abandoned at ``limit_s``.
    """
    groups = bench.groups()
    pool = [m for g in groups for m in g.metas]
    held = groups[0]
    values = bench.values(held)
    model = values.model
    cont = parse_chain("esc:L")
    out: dict = {"limit_s": limit_s, "group_size": group_size, "rows": []}
    for n in sizes:
        rng = random.Random(n)
        batch = [pool[rng.randrange(len(pool))] for _ in range(n)]
        wf = bench.library["S"]
        states = []
        for meta in batch[:min(n, 2000)]:
            state, ok, _, _ = bench.corpus.run(meta, State(), wf)
            states.append((meta, state, ok))
        reps = -(-n // len(states))
        items = (states * reps)[:n]

        t0 = time.perf_counter()
        helds = model.s_of_many([(m, st) for m, st, _ in items])
        rolls = model.rollout_many([(m, st, cont) for m, st, _ in items], helds)
        values.values([m for m, _, _ in items], [st for _, st, _ in items],
                      [cont] * n, rolls, helds)
        score_s = time.perf_counter() - t0

        cands = []
        for i, ((meta, state, _), roll, h) in enumerate(zip(items, rolls, helds)):
            qid = f"{meta.qid}#{i}"
            cands.append(Candidate(qid, "esc:L", roll.p, roll.cost, roll.var,
                                   roll.worst_case))
            cands.append(Candidate(qid, "stop", h, zeros(), zeros(), zeros(),
                                   is_stop=True))
        rhs = budget_for_group(budget, n)

        t0 = time.perf_counter()
        n_groups = 0
        for start in range(0, n, group_size * 2):
            chunk = cands[start:start + group_size * 2]
            if not chunk:
                break
            solve(chunk, rhs * (len(chunk) / 2.0 / n))
            n_groups += 1
        group_s = time.perf_counter() - t0

        whole_s, whole_done = None, None
        if n <= 50000:
            t0 = time.perf_counter()
            try:
                solve(cands, rhs, time_limit=limit_s)
                whole_s = time.perf_counter() - t0
                whole_done = whole_s <= limit_s
            except Exception:
                whole_s = time.perf_counter() - t0
                whole_done = False
        out["rows"].append({
            "n": n, "score_s": score_s, "groups": n_groups,
            "group_total_s": group_s,
            "group_ms": 1000.0 * group_s / max(1, n_groups),
            "whole_s": whole_s, "whole_done": whole_done})
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
    """The planner against the strongest baseline when the agent, the
    executors or the digest changes.

    An open agent reads either the text digest or a latent digest of a few
    dozen soft tokens, which costs a fraction of the prompt.
    """
    from ..config import PRICES, Price
    unit_of = bench.unit_of()
    OPEN_EXECUTORS = {"S": Price(0.05, 0.08), "M": Price(0.12, 0.20),
                      "L": Price(0.30, 0.50)}
    MINI_AGENT = {"agent": Price(0.40, 1.60)}
    OPEN_AGENT = {"agent": Price(0.20, 0.30)}
    presets = {
        "api_agent_api_executors": ({}, False),
        "mini_agent_api_executors": (MINI_AGENT, False),
        "api_agent_open_executors": (OPEN_EXECUTORS, False),
        "open_agent_api_executors_digest": (OPEN_AGENT, False),
        "open_agent_api_executors_latent": (OPEN_AGENT, True),
        "open_agent_open_executors_digest": ({**OPEN_EXECUTORS, **OPEN_AGENT}, False),
        "open_agent_open_executors_latent": ({**OPEN_EXECUTORS, **OPEN_AGENT}, True),
    }
    out: dict = {"budget": budget, "configurations": {}}
    original = dict(PRICES)
    try:
        for name, (override, latent) in presets.items():
            PRICES.update(original)
            PRICES.update(override)
            cfg = bench.cfg
            if latent:
                cfg = cfg.with_(digest_tokens=cfg.latent_tokens)
            ours = bench.run_method(PLANNER, budget, runs, cfg=cfg)
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
                "planning_share": _mean_planning(ours["details"]),
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
