from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from ..analysis.escalation import frontier, stats as esc_stats
from ..baselines import BASELINES
from ..baselines.agents import ProfiledPerRequest
from ..config import (BUDGETS, DIMS, GRID_AXES, GRID_SLACK, SWEEP_MONEY,
                      PlannerConfig, budget_for_group)
from ..config import zeros
from ..grammar import STOP, State, parse_chain
from ..planner import Meter
from ..planner.allocation import Candidate, solve
from ..worldmodel.stacking import selector
from .effects import Effect
from .runner import BudgetGroup, Workbench
from .stats import accuracy, lead, paired_contrast

PLANNER = "budgetsql"


def _eff(res: dict) -> Effect:
    e = res.get("effect") or {}
    return Effect(float(e.get("mean", 0.0)), float(e.get("var", 0.0)))


def _acc(res: dict) -> tuple[float, float]:
    return accuracy(res["runs"], _eff(res).mean)


def _lead(a: dict, b: dict, unit_of: dict) -> object:
    ea, eb = _eff(a), _eff(b)
    return lead(a["runs"], b["runs"], unit_of, offset=ea.mean - eb.mean,
                offset_var=ea.var + eb.var)


def _contrast(a: dict, b: dict) -> object:
    ea, eb = _eff(a), _eff(b)
    same = ea.mean == eb.mean and ea.var == eb.var
    return paired_contrast(a["runs"], b["runs"], offset=ea.mean - eb.mean,
                           offset_var=0.0 if same else ea.var + eb.var)


BIRD_ONLY = ("best_fixed_workflow",)


def default_methods(bench: Workbench) -> list[str]:
    if bench.workload.budget_groups == "db":
        return list(BASELINES)
    return [m for m in BASELINES if m not in BIRD_ONLY]


DERIVED_AGENTS = ("llm_agent", "llm_agent_dry_runs")


def main_table(bench: Workbench, methods: list[str] | None = None,
               budgets: list[str] | None = None, runs: int | None = None,
               oracle: bool = True) -> dict:
    methods = methods or default_methods(bench)
    budgets = budgets or list(bench.workload.budgets)
    unit_of = bench.unit_of()
    derive = bench.workload.budget_groups == "db"
    replayed = [m for m in methods if not (derive and m in DERIVED_AGENTS)]
    derived = derived_agent_rows(bench, [m for m in methods if derive and m in DERIVED_AGENTS],
                                 budgets, runs) if derive else {}
    out: dict = {"budgets": {}, "methods": [PLANNER] + methods
                 + (["oracle"] if oracle else []),
                 "derived": sorted(derived)}
    for b in budgets:
        ours = bench.run_method(PLANNER, b, runs)
        results = {name: bench.run_method(name, b, runs) for name in replayed}
        for name, per_budget in derived.items():
            results[name] = per_budget[b]
        rows: dict[str, dict] = {}
        for name, res in results.items():
            acc, spread = _acc(res)
            rows[name] = {"accuracy": acc, "spread": spread,
                          "lead_over": _lead(ours, res, unit_of).as_dict(),
                          "utilisation": _mean_util(res.get("details", [])),
                          "derived": name in derived}
        if oracle:
            res = bench.run_method("oracle", b, runs)
            acc, spread = _acc(res)
            rows["oracle"] = {"accuracy": acc, "spread": spread,
                              "lead_over": _lead(ours, res, unit_of).as_dict(),
                              "utilisation": _mean_util(res["details"])}
        acc, spread = _acc(ours)
        ranked = {n: r for n, r in rows.items() if n != "oracle"}
        best = max(ranked, key=lambda n: ranked[n]["accuracy"]) if ranked else None
        out["budgets"][b] = {
            "planner": {"accuracy": acc, "spread": spread,
                        "modelled_effect": ours["effect"],
                        "utilisation": _mean_util(ours["details"]),
                        "planning_share": _mean_planning(ours["details"]),
                        "binding_fraction": _mean_binding(ours["details"]),
                        "turns": _mean_key(ours["details"], "turns")},
            "baselines": rows,
            "strongest_baseline": best,
            "lead": rows[best]["lead_over"] if best else None,
        }
    return out


def derived_agent_rows(bench: Workbench, agents: list[str], budgets: list[str],
                       runs: int | None = None, base: str = "vinelm_style",
                       anchor: str = "V1") -> dict[str, dict[str, dict]]:
    return {name: {b: derived_agent_at(bench, name, b, None, runs, base, anchor)
                   for b in budgets} for name in agents}


def derived_agent_at(bench: Workbench, name: str, budget: str | None, vector=None,
                     runs: int | None = None, base: str = "vinelm_style",
                     anchor: str = "V1") -> dict:
    cache = bench.__dict__.setdefault("_derived_anchor", {})
    key = (name, base, anchor, runs)
    if key not in cache:
        base_v1 = bench.run_method(base, anchor, runs)
        free = bench.run_method(name, anchor, runs, cfg=bench.cfg.with_(free_turns=True))
        charged = bench.run_method(name, anchor, runs)
        off = paired_contrast(free["runs"], base_v1["runs"])
        n = sum(g.size for g in bench.groups())
        per_q = _planning_spend(charged["details"]) / max(1, n) / max(1, len(charged["runs"]))
        cache[key] = (off, per_q, charged["details"])
    off, per_q, details = cache[key]
    vec_b = np.asarray(BUDGETS[budget] if vector is None else vector, dtype=float)
    here = bench.run_method(base, budget, runs, vector=vector)
    shrunk = bench.run_method(base, budget, runs, vector=np.maximum(vec_b - per_q, 0.0))
    turn_cost = _acc(here)[0] - _acc(shrunk)[0]
    return {"method": name, "budget": budget, "runs": here["runs"],
            "details": details if (vector is None and budget == anchor) else [],
            "effect": {"mean": off.mean - turn_cost, "var": off.standard_error ** 2},
            "offset_v1": off.mean, "turn_cost": turn_cost}


def replayed_baselines(bench: Workbench) -> list[str]:
    if bench.workload.budget_groups == "db":
        return [m for m in default_methods(bench) if m not in DERIVED_AGENTS]
    return default_methods(bench)


def _result(bench: Workbench, name: str, budget: str | None, runs: int | None,
            vector=None) -> dict:
    if bench.workload.budget_groups == "db" and name in DERIVED_AGENTS:
        return derived_agent_at(bench, name, budget, vector, runs)
    return bench.run_method(name, budget, runs, vector=vector)


def budget_sweep(bench: Workbench, methods: list[str] | None = None,
                 money: tuple[float, ...] = SWEEP_MONEY, runs: int = 1) -> dict:
    methods = methods or ["best_fixed_mixture", "cascade_routing", "uab_style",
                          "omnirouter_style", "vinelm_style", "llm_agent"]
    out: dict = {"money": list(money), "curves": {}}
    for name in [PLANNER] + methods:
        curve = []
        for usd in money:
            vector = np.array([usd, GRID_SLACK["tok"], GRID_SLACK["sec"]])
            curve.append(_acc(_result(bench, name, None, runs, vector))[0])
        out["curves"][name] = curve
    return out


def budget_grid(bench: Workbench, pairs: tuple[tuple[str, str], ...] =
                (("usd", "sec"), ("usd", "tok")), runs: int = 3,
                methods: list[str] | None = None) -> dict:
    methods = methods or replayed_baselines(bench)
    unit_of = bench.unit_of()
    out: dict = {"axes": {k: list(v) for k, v in GRID_AXES.items()}, "pairs": {}}
    for x, y in pairs:
        cells = []
        for vx in GRID_AXES[x]:
            for vy in GRID_AXES[y]:
                vector = np.array([GRID_SLACK[k] for k in DIMS], dtype=float)
                vector[DIMS.index(x)] = vx
                vector[DIMS.index(y)] = vy
                ours = bench.run_method(PLANNER, None, runs, vector=vector)
                best_name, best_res, best_acc = None, None, -1.0
                for name in methods:
                    res = bench.run_method(name, None, runs, vector=vector)
                    acc, _ = _acc(res)
                    if acc > best_acc:
                        best_name, best_res, best_acc = name, res, acc
                acc, _ = _acc(ours)
                ld = _lead(ours, best_res, unit_of)
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
        ours = {"runs": ours_runs, "effect": bench.effect(bench.cfg, b).as_dict()}
        rows: dict[str, dict] = {}
        for name in [m for m in methods if m in BASELINES]:
            per_run = []
            for r in range(runs):
                meter = Meter(budget)
                correct: dict[str, bool] = {}
                subs = bench.groups()
                later = sum(g.size for g in subs)
                for sub in subs:
                    later -= sub.size
                    method = bench.method(name, sub)
                    model = getattr(method, "model", None)
                    if model is not None:
                        model.reset_online()
                    method.batch_later = later
                    metas = bench.order_metas(sub.metas, r)
                    try:
                        correct.update(method.run(metas, budget_for_group(b, sub.size),
                                                  meter).correct)
                    finally:
                        method.batch_later = 0
                per_run.append(correct)
            acc, spread = accuracy(per_run)
            rows[name] = {"accuracy": acc, "spread": spread,
                          "lead_over": _lead(ours, {"runs": per_run}, unit_of).as_dict()}
        acc, spread = _acc(ours)
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
    budgets = budgets or list(bench.workload.budgets)
    methods = methods or replayed_baselines(bench)
    unit_of = bench.unit_of()
    out: dict = {"budgets": {}}
    for b in budgets:
        rows: dict[str, dict] = {}
        for order in orders:
            ours = bench.run_method(PLANNER, b, runs, order=order)
            best_res, best_name, best_acc = None, None, -1.0
            for name in methods:
                res = bench.run_method(name, b, runs, order=order)
                acc, _ = _acc(res)
                if acc > best_acc:
                    best_name, best_res, best_acc = name, res, acc
            acc, _ = _acc(ours)
            rows[order] = {"accuracy": acc, "strongest_baseline": best_name,
                           "baseline_accuracy": best_acc,
                           "lead": _lead(ours, best_res, unit_of).as_dict()}
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
                ours_acc = _accuracy_on(ours["runs"], qids) + _eff(ours).mean
                best = max((_accuracy_on(r, qids) for r in base.values()),
                           default=0.0)
                row[kind][label] = {"accuracy": ours_acc, "baseline": best,
                                    "gain": ours_acc - best,
                                    "share": len(qids) / max(1, len(labels))}
        out["budgets"][b] = row
    return out


def per_database(bench: Workbench, budgets: list[str] | None = None,
                 methods: list[str] | None = None, runs: int | None = None) -> dict:
    budgets = budgets or list(bench.workload.budgets)
    methods = methods or replayed_baselines(bench)
    out: dict = {"budgets": {}}
    for b in budgets:
        ours = bench.run_method(PLANNER, b, runs)
        base = {n: bench.run_method(n, b, runs)["runs"] for n in methods}
        rows: dict[str, dict] = {}
        for group in bench.groups():
            qids = [m.qid for m in group.metas]
            ours_acc = _accuracy_on(ours["runs"], qids) + _eff(ours).mean
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
    means = [100.0 * np.mean([float(r.get(q) or 0.0) for q in qids])
             for r in runs if r]
    return float(np.mean(means)) if means else 0.0


def ablation_arms(cfg: PlannerConfig) -> dict[str, object]:
    return {
        "profiled_averages_no_world_model": ProfiledPerRequest,
        "full_data_heads_no_stacks": cfg.with_(use_stacking=False),
        "flat_world_model_no_rollouts": cfg.with_(use_rollouts=False),
        "direct_heads_only_no_stack": cfg.with_(use_rollouts=False, use_stacking=False),
        "no_differential_features": cfg.with_(use_differential_features=False),
        "scored_before_first_stage": cfg.with_(score_from_empty=True),
        "log_scale_cost_heads": cfg.with_(log_scale_costs=True),
        "core_only_no_agent": cfg.with_(use_agent=False),
        "whole_workflows_before_running": cfg.with_(plan_whole_workflows=True,
                                                    use_agent=False),
        "money_only_prices": cfg.with_(priced_dims=("usd",)),
        "greedy_ranking_no_allocation": cfg.with_(use_allocation_lp=False),
        "with_a_cost_reserve": cfg.with_(cost_reserve=True, reserve_z=1.0),
        "opening_plan_no_periodic_resolves": cfg.with_(periodic_resolves=False),
        "world_model_question_only": cfg.with_(use_execution_features=False),
        "world_model_execution_only": cfg.with_(use_question_features=False),
    }


def ablation(bench: Workbench, budgets: list[str] | None = None,
             arms: dict | None = None, runs: int | None = None) -> dict:
    budgets = budgets or list(bench.workload.budgets)
    arms = arms if arms is not None else ablation_arms(bench.cfg)
    out: dict = {"budgets": {}}
    for b in budgets:
        full = bench.run_method(PLANNER, b, runs)
        acc, _ = _acc(full)
        rows: dict[str, dict] = {}
        for name, arm in arms.items():
            if isinstance(arm, PlannerConfig):
                res = bench.run_method(PLANNER, b, runs, cfg=arm)
            else:
                res = _run_other(bench, arm, b, runs)
            contrast = _contrast(res, full)
            a, _ = _acc(res)
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
            metas = bench.order_metas(group.metas, r)
            method = cls(name=cls.__name__, cfg=bench.cfg, library=bench.library)
            method.prepare(bench.corpus, bench.train_dbs(group))
            b = budget_for_group(budget, group.size)
            res = method.run(metas, b, Meter(b))
            correct.update(res.correct)
            details.append({"run": r, "group": group.name, **res.as_dict()})
        per_run.append(correct)
    return {"method": cls.__name__, "budget": budget, "runs": per_run,
            "details": details}


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
            acc, _ = _acc(res)
            rows.append({"value": v, "accuracy": acc,
                         "turns": _mean_key(res["details"], "turns"),
                         "planning_share": _mean_planning(res["details"])})
        out["settings"][key] = rows
    return out


def world_model_quality(bench: Workbench, openers: tuple[str, ...] = ("S", "M")
                        ) -> dict:
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
                _, ok2, _, outs = bench.corpus.run(meta, state, cont, held=ok,
                                                   select=selector(model, meta, s_held))
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


def online_learning(bench: Workbench, days: int = 5, day_n: int = 1500,
                    budget: str = "V4", seed: int = 0, label_share: float = 0.30,
                    day_corpus=None) -> dict:
    from ..worldmodel.model import StageWorldModel

    rng = random.Random(seed)
    if day_corpus is not None:
        fit_group = BudgetGroup("online-fit-all", [], [])
        pool_corpus = day_corpus
        pool = list(day_corpus.questions)
    else:
        dbs = bench.corpus.dbs
        held = dbs[: max(1, len(dbs) // 2)]
        fit_group = BudgetGroup("online-fit-" + "-".join(held), held, [])
        pool_corpus = bench.corpus
        pool = [m for d in held for m in bench.corpus.questions_of(d)]
    online = dict(online_cost_updates=True)
    arms = {"frozen": bench.cfg.with_(online_cost_updates=False, recalibrate_success=False),
            "updated": bench.cfg.with_(**online),
            "recalibrated": bench.cfg.with_(**online, recalibrate_success=True),
            "labelled": bench.cfg.with_(**online, recalibrate_success=True)}
    planners = {}
    for k, c in arms.items():
        p = bench.planner(fit_group, c)
        p.corpus = pool_corpus
        planners[k] = p
    labelled_rows: list = []
    train_dbs = [d for d in bench.corpus.dbs if d not in fit_group.dbs]
    out: dict = {"days": [], "budget": budget, "label_share": label_share,
                 "day_questions": min(day_n, len(pool)),
                 "unseen_databases": len({m.db for m in pool})}
    for day in range(days):
        batch = rng.sample(pool, min(day_n, len(pool)))
        row: dict = {"day": day + 1}
        for k, planner in planners.items():
            row[k] = {"cost_error_usd": _median_cost_error(planner, batch)}
            b = budget_for_group(budget, len(batch))
            res = planner.run(batch, b, Meter(b, planning_cap=bench.cfg.planning_cap))
            row[k].update({"accuracy": res.accuracy * 100.0,
                           "cancelled_per_1k": 1000.0 * res.cancelled / max(1, len(batch))})
            if k == "labelled":
                labelled = [r for r in planner.records.values()
                            if rng.random() < label_share]
                labelled_rows += _rows_from_history(pool_corpus, labelled)
                _refit_with(planner, bench.corpus, train_dbs, labelled_rows,
                            StageWorldModel)
        for k in ("updated", "recalibrated", "labelled"):
            row[f"gain_{k}"] = row[k]["accuracy"] - row["frozen"]["accuracy"]
        row["gain"] = row["gain_updated"]
        out["days"].append(row)
    return out


def _rows_from_history(corpus, records) -> list:
    from ..grammar import ANSWER_PRODUCING
    from ..trace.corpus import TrainingRow
    rows = []
    for rec in records:
        state, held = State(), None
        for i, (stage, _) in enumerate(rec.state.history):
            out = corpus.outcome(rec.meta, state, stage)
            if out is None:
                break
            out.held_before = held
            rows.append(TrainingRow(rec.meta, state, stage, out, held))
            keep = rec.state.kept(i)
            if stage.kind in ANSWER_PRODUCING and (keep or held is None):
                held = out.answer_correct
            state = state.append(stage, out.obs, keep=keep, values=out)
    return rows


def _refit_with(planner, corpus, train_dbs: list[str], extra: list, model_cls) -> None:
    old = planner.model
    model = model_cls(old.cfg)
    model.train_dbs = list(train_dbs)
    model._fit_rows(corpus.training_rows(train_dbs) + list(extra))
    model._kappa, model._cost_obs = old._kappa, old._cost_obs
    model._logit_shift = old._logit_shift
    planner.model = model
    planner.values.model = model


def _median_cost_error(planner, metas) -> float:
    from ..grammar import gen
    errs = []
    stage = gen("S")
    for m in metas[:300]:
        out = planner.corpus.outcome(m, State(), stage)
        if out is None or out.cost[0] <= 0:
            continue
        pred = planner.model.predict(m, State(), stage).mu[0]
        errs.append(abs(pred - out.cost[0]) / out.cost[0])
    return float(np.median(errs)) if errs else 0.0


def calibration(bench: Workbench, budget: str = "V4", bins: int = 10,
                runs: int | None = None) -> dict:
    from ..planner import BatchPlanner, PlanningAgent
    runs = runs or 1
    pred, real = [], []
    groups = bench.groups()
    for r in range(runs):
        planners = [bench.planner(g) for g in groups]
        items = []
        for g in groups:
            b = budget_for_group(budget, g.size)
            items.append((bench.order_metas(g.metas, r), b,
                          Meter(b, planning_cap=bench.cfg.planning_cap)))
        agent = None
        if bench.cfg.use_agent:
            agent = (bench.agent_factory(True) if bench.agent_factory is not None
                     else PlanningAgent())
        BatchPlanner(planners, bench.cfg, agent).run(items)
        for planner in planners:
            for rec in planner.records.values():
                if rec.selected_value is None:
                    continue
                pred.append(float(rec.selected_value))
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
                for (meta, state, ok), v, v_held in zip(items, vals, helds):
                    _, ok2, _, _ = bench.corpus.run(meta, state, cont, held=ok,
                                                    select=selector(model, meta, v_held))
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


def overhead(bench: Workbench, budgets: list[str] | None = None,
             batch_sizes: tuple[int, ...] = (150, 500), runs: int = 1,
             sessions: tuple[int, ...] = (0, 4, 11, 30, 60)) -> dict:
    budgets = budgets or list(bench.workload.budgets)
    out: dict = {"budgets": {}, "batches": [], "sessions": []}
    for b in budgets:
        res = bench.run_method(PLANNER, b, runs)
        out["budgets"][b] = {
            "accuracy": _acc(res)[0],
            "planning_share": _mean_planning(res["details"]),
            "turns": _mean_key(res["details"], "turns"),
            "sessions": _mean_key(res["details"], "sessions"),
            "wall_seconds": _mean_key(res["details"], "wall_seconds"),
        }
    for requested in sessions:
        cfg = bench.cfg.with_(use_agent=requested > 0,
                              max_sessions=max(1, requested))
        res = bench.run_method(PLANNER, budgets[0], runs, cfg=cfg)
        acc, _ = _acc(res)
        out["sessions"].append({
            "requested": requested, "accuracy": acc,
            "turns": _mean_key(res["details"], "turns"),
            "planning_share": _mean_planning(res["details"])})

    groups = bench.groups()
    for n in batch_sizes:
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


def _continuations(grammar, state: State, w_max: int) -> list:
    from ..grammar import branch, shape
    out: dict[str, object] = {}
    for a in grammar.actions(state):
        out.setdefault(shape(branch(a, {o: None for o in a.observations()})),
                       branch(a, {o: None for o in a.observations()}))
        if a.kind == "verify":
            for tier in ("M", "L"):
                chain = parse_chain(f"ver:{a.tier},esc:{tier}")
                if grammar.is_wellformed(chain, state):
                    out.setdefault(shape(chain), chain)
    return list(out.values())[:max(0, w_max)]


def scaling(bench: Workbench, sizes: tuple[int, ...] = (1530, 5000, 15000,
                                                       50000, 150000),
            budget: str = "V1", group_size: int = 139,
            limit_s: float = 60.0, whole_up_to: int = 50000) -> dict:
    groups = bench.groups()
    pool = [m for g in groups for m in g.metas]
    held = groups[0]
    values = bench.values(held)
    model = values.model
    grammar = bench.corpus.grammar
    w_max = bench.cfg.max_candidates
    out: dict = {"limit_s": limit_s, "group_size": group_size, "w_max": w_max,
                 "rows": []}
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
        conts = [_continuations(grammar, st, w_max) for _, st, _ in items]

        t0 = time.perf_counter()
        helds = model.s_of_many([(m, st) for m, st, _ in items])
        flat = [(m, st, c, h) for (m, st, _), cs, h in zip(items, conts, helds)
                for c in cs]
        rolls = model.rollout_many([(m, st, c) for m, st, c, _ in flat],
                                   [h for *_, h in flat])
        vals = values.values([f[0] for f in flat], [f[1] for f in flat],
                             [f[2] for f in flat], rolls, [f[3] for f in flat])
        stop_vals = values.values([m for m, _, _ in items], [st for _, st, _ in items],
                                  [STOP] * n, [None] * n, helds)
        score_s = time.perf_counter() - t0

        per_q: list[list[Candidate]] = []
        k = 0
        for i, ((meta, _, _), cs) in enumerate(zip(items, conts)):
            qid = f"{meta.qid}#{i}"
            row = [Candidate(qid, "stop", float(stop_vals[i]), zeros(), zeros(),
                             zeros(), is_stop=True)]
            for c in cs:
                roll = rolls[k]
                row.append(Candidate(qid, f"c{k}", float(vals[k]), roll.cost, roll.var,
                                     roll.worst_case))
                k += 1
            per_q.append(row)
        rhs = budget_for_group(budget, n)

        t0 = time.perf_counter()
        n_groups = 0
        for start in range(0, n, group_size):
            chunk = [c for row in per_q[start:start + group_size] for c in row]
            if not chunk:
                break
            solve(chunk, rhs * (min(group_size, n - start) / n))
            n_groups += 1
        group_s = time.perf_counter() - t0

        whole_s, whole_done = None, None
        if n <= whole_up_to:
            cands = [c for row in per_q for c in row]
            t0 = time.perf_counter()
            try:
                res = solve(cands, rhs, time_limit=limit_s)
                whole_s = time.perf_counter() - t0
                whole_done = res.status == "optimal" and whole_s <= limit_s
            except Exception:
                whole_s = time.perf_counter() - t0
                whole_done = False
        out["rows"].append({
            "n": n, "candidates": len(flat) + n, "score_s": score_s,
            "groups": n_groups, "group_total_s": group_s,
            "group_ms": 1000.0 * group_s / max(1, n_groups),
            "whole_s": whole_s, "whole_done": whole_done})
    return out


def trace_frontier(bench: Workbench, pairs=(("M", "L"), ("S", "L"), ("S", "M"))
                   ) -> dict:
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
                              runs: int = 1, base: str = "best_fixed_mixture") -> dict:
    budgets = budgets or list(bench.workload.budgets)
    unit_of = bench.unit_of()
    out: dict = {"budgets": {}, "base": base}
    for b in budgets:
        mix = bench.run_method(base, b, runs)
        mix_acc, _ = _acc(mix)
        row = {}
        for label, cfg in (("stacked", bench.cfg),
                           ("full_data_heads", bench.cfg.with_(use_stacking=False))):
            replay = bench.run_method(PLANNER, b, runs, cfg=cfg, with_effects=False)
            full = bench.run_method(PLANNER, b, runs, cfg=cfg)
            pred = 100.0 * _mean_key(replay["details"], "predicted_accuracy")
            rep_acc, _ = _acc(replay)
            full_acc, _ = _acc(full)
            row[label] = {"predicted": pred - mix_acc, "replayed": rep_acc - mix_acc,
                          "full": full_acc - mix_acc,
                          "overestimate": pred - rep_acc,
                          "se": _lead(replay, mix, unit_of).standard_error}
        out["budgets"][b] = row
    return out


@dataclass
class Scenario:
    name: str
    agent_price: object = None
    bench: object = None
    executor_prices: dict | None = None
    time_scale: float = 1.0
    effects: object = None
    latent: bool = False


def default_scenarios(bench: Workbench, open_bench: Workbench | None = None,
                      open_prices: dict | None = None, open_time_scale: float = 1.0,
                      open_agent_price=None, effects_by_agent: dict | None = None
                      ) -> list[Scenario]:
    from ..config import Price
    eff = effects_by_agent or {}
    gpt5, mini, qwen = None, Price(0.40, 1.60), open_agent_price
    out = [Scenario("gpt5/api/digest", gpt5, effects=eff.get("gpt5")),
           Scenario("4.1-mini/api/digest", mini, effects=eff.get("4.1-mini"))]
    open_kw = dict(bench=open_bench, executor_prices=open_prices,
                   time_scale=open_time_scale)
    out.append(Scenario("gpt5/open/digest", gpt5, effects=eff.get("gpt5"), **open_kw))
    for latent in (False, True):
        tag = "latent" if latent else "digest"
        out.append(Scenario(f"qwen/api/{tag}", qwen, effects=eff.get("qwen"),
                            latent=latent))
        out.append(Scenario(f"qwen/open/{tag}", qwen, effects=eff.get("qwen"),
                            latent=latent, **open_kw))
    return out


def scenarios(bench: Workbench, budget: str = "V4", runs: int = 1,
              presets: list[Scenario] | None = None) -> dict:
    from ..config import PRICES
    presets = presets if presets is not None else default_scenarios(bench)
    out: dict = {"budget": budget, "configurations": {}}
    original = dict(PRICES)
    for sc in presets:
        open_exec = "/open/" in sc.name
        if (open_exec and sc.bench is None) or (sc.name.startswith("qwen")
                                                and sc.agent_price is None):
            out["configurations"][sc.name] = {
                "skipped": "needs " + ("the open-weight executors' trace"
                                       if open_exec and sc.bench is None
                                       else "the open-weight agent's price")}
            continue
        sb = sc.bench or bench
        saved = sb.effects
        try:
            PRICES.update(original)
            if sc.agent_price is not None:
                PRICES["agent"] = sc.agent_price
            if sc.executor_prices:
                PRICES.update(sc.executor_prices)
            if sc.effects is not None:
                sb.effects = sc.effects
            cfg = sb.cfg.with_(digest_tokens=sb.cfg.latent_tokens) if sc.latent else sb.cfg
            vector = BUDGETS[budget].copy()
            vector[2] *= sc.time_scale
            unit_of = sb.unit_of()
            ours = sb.run_method(PLANNER, budget, runs, cfg=cfg, vector=vector)
            best_name, best_res, best_acc = None, None, -1.0
            for b in replayed_baselines(sb):
                res = sb.run_method(b, budget, runs, vector=vector)
                acc, _ = _acc(res)
                if acc > best_acc:
                    best_name, best_res, best_acc = b, res, acc
            acc, _ = _acc(ours)
            out["configurations"][sc.name] = {
                "accuracy": acc, "strongest_baseline": best_name,
                "baseline_accuracy": best_acc,
                "planning_share": _mean_planning(ours["details"]),
                "lead": _lead(ours, best_res, unit_of).as_dict(),
            }
        finally:
            sb.effects = saved
            PRICES.clear()
            PRICES.update(original)
    return out


def price_sensitivity(bench: Workbench, factors: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
                      base: str = "V1", tier: str = "L", runs: int | None = None,
                      methods: list[str] | None = None) -> dict:
    methods = methods or replayed_baselines(bench)
    v = BUDGETS[base]
    out: dict = {"base": base, "tier": tier, "factors": list(factors), "methods": {}}
    for name in [PLANNER] + methods:
        res = bench.run_method(name, base, runs)
        spent = {}
        for d in res["details"]:
            for t, usd in (d.get("tier_money") or {}).items():
                spent[t] = spent.get(t, 0.0) + float(usd)
        total = sum(spent.values())
        share = spent.get(tier, 0.0) / total if total > 0 else 0.0
        curve = []
        for k in factors:
            vector = np.array([v[0] / (1.0 + (k - 1.0) * share), v[1], v[2]])
            curve.append(_acc(bench.run_method(name, None, runs, vector=vector))[0])
        out["methods"][name] = {"share": share, "accuracy": curve}
    base_rows = {n: r for n, r in out["methods"].items() if n != PLANNER}
    out["strongest_baseline"] = [
        max(base_rows, key=lambda n: base_rows[n]["accuracy"][i]) if base_rows else None
        for i in range(len(factors))]
    return out


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


def _planning_spend(details: list[dict]) -> np.ndarray:
    out = np.zeros(len(DIMS))
    for d in details:
        share = d.get("planning_share") or {}
        budget = d.get("budget") or {}
        out += np.array([float(share.get(k, 0.0)) * float(budget.get(k, 0.0))
                         for k in DIMS])
    return out
