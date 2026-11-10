"""Command line entry point."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import BUDGETS, PlannerConfig, WORKLOADS, dump_json
from .eval import report
from .eval.runner import Workbench
from .grammar import Grammar, State, reference_library

PROFILES = {
    "smoke": dict(n_estimators=40, num_leaves=8, turns_per_session=2),
    "fast": dict(n_estimators=80, num_leaves=12),
    "full": dict(),
}


# --------------------------------------------------------------------------- #
# building a workbench
# --------------------------------------------------------------------------- #
def make_corpus(args):
    from .trace import RecordedTrace, SimulatedTrace, TraceCorpus
    workload = WORKLOADS.get(args.workload, WORKLOADS["bird"])
    if args.trace:
        source = RecordedTrace(args.trace)
    else:
        source = SimulatedTrace(
            n_questions=args.questions or workload.n_questions,
            n_dbs=args.dbs or workload.n_dbs,
            seed=args.seed, name=args.workload)
    return TraceCorpus(source, Grammar(), max_states=args.max_states, seed=args.seed)


def make_config(args) -> PlannerConfig:
    workload = WORKLOADS.get(args.workload, WORKLOADS["bird"])
    cfg = PlannerConfig(limits=workload.limits, **PROFILES.get(args.profile, {}))
    if getattr(args, "no_agent", False):
        cfg = cfg.with_(use_agent=False)
    if getattr(args, "runs", None):
        pass
    return cfg


def make_bench(args) -> Workbench:
    corpus = make_corpus(args)
    workload = WORKLOADS.get(args.workload, WORKLOADS["bird"])
    if getattr(args, "runs", None):
        workload = type(workload)(**{**workload.__dict__, "runs": args.runs})
    cache = Path(args.cache) if args.cache else None
    folds = 2 if args.profile == "smoke" else 4
    return Workbench(corpus, make_config(args), workload, cache,
                     reference_library(), inner_folds=folds)


def write(args, kind: str, result) -> None:
    text = report.render(kind, result)
    print(text)
    if args.out:
        out = Path(args.out)
        dump_json({"kind": kind, "result": result}, out)
        out.with_suffix(".txt").write_text(text + "\n")
        print(f"\nwritten to {out} and {out.with_suffix('.txt')}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_simulate(args) -> None:
    from .trace import SimulatedTrace, materialise, write_jsonl
    workload = WORKLOADS.get(args.workload, WORKLOADS["bird"])
    src = SimulatedTrace(n_questions=args.questions or workload.n_questions,
                         n_dbs=args.dbs or workload.n_dbs, seed=args.seed,
                         name=args.workload)
    grammar = Grammar()
    n = write_jsonl(args.out, materialise(src, src.questions(), grammar,
                                          args.max_states))
    print(f"wrote {n} questions to {args.out}")


def cmd_collect(args) -> None:
    from .executors.llm import tiers_from_env
    from .trace import Collector, load_bird
    questions = load_bird(args.dev_json, args.db_root, args.questions or None)
    collector = Collector(tiers_from_env(), Grammar(), max_states=args.max_states,
                          sample_rows=args.sample_rows, spend_cap_usd=args.cap)
    print(json.dumps(collector.collect(questions, args.out), indent=2))


def cmd_library(args) -> None:
    corpus = make_corpus(args)
    lib = reference_library()
    from .grammar import LIBRARY_ORDER
    rows = []
    for name in LIBRARY_ORDER:
        wf = lib[name]
        acc, cost = [], []
        for meta in corpus.questions:
            _, ok, c, _ = corpus.run(meta, State(), wf)
            acc.append(ok)
            cost.append(c)
        c = np.mean(cost, axis=0)
        rows.append({"name": name, "accuracy": 100 * float(np.mean(acc)),
                     "usd_per_1k": float(c[0]) * 1000, "tokens": float(c[1]),
                     "seconds": float(c[2])})
    write(args, "library", rows)


def cmd_verify(args) -> None:
    from .analysis import verify_identities
    res = verify_identities()
    print(json.dumps(res, indent=2))
    ok = all(v is True or (isinstance(v, dict) and v.get("within"))
             for v in res.values())
    sys.exit(0 if ok else 1)


def cmd_run(args) -> None:
    bench = make_bench(args)
    res = bench.run_method(args.method, args.budget, args.runs)
    from .eval.stats import accuracy
    acc, spread = accuracy(res["runs"])
    payload = {"method": args.method, "budget": args.budget, "accuracy": acc,
               "spread": spread, "details": res["details"]}
    print(f"{args.method} on {args.budget}: {acc:.2f} EX% (+-{spread:.2f})")
    if args.out:
        dump_json(payload, args.out)


def _exp():
    from .eval import experiments as x
    return {
        "main": ("main", lambda b, a: x.main_table(b, budgets=a.budgets, runs=a.runs)),
        "sweep": ("sweep", lambda b, a: x.budget_sweep(b, runs=1)),
        "ablation": ("ablation", lambda b, a: x.ablation(b, budgets=a.budgets,
                                                         runs=a.runs)),
        "sensitivity": ("sensitivity", lambda b, a: x.sensitivity(
            b, budget=(a.budgets or ["V4"])[0], runs=a.runs)),
        "online": ("online", lambda b, a: x.online_learning(
            b, days=a.days, day_n=a.day_questions)),
        "overhead": ("overhead", lambda b, a: x.overhead(b, budgets=a.budgets,
                                                          runs=1)),
        "frontier": ("frontier", lambda b, a: x.trace_frontier(b)),
        "worldmodel": ("world_model", lambda b, a: x.world_model_quality(b)),
        "plans": ("plans", lambda b, a: x.predicted_versus_replayed(
            b, budgets=a.budgets, runs=1)),
        "scenarios": ("scenarios", lambda b, a: x.scenarios(
            b, budget=(a.budgets or ["V4"])[0], runs=1)),
        "grid": ("grid", lambda b, a: x.budget_grid(b, runs=a.runs or 3)),
        "pooled": ("pooled", lambda b, a: x.pooled_budget(b, budgets=a.budgets,
                                                          runs=a.runs)),
        "orders": ("orders", lambda b, a: x.arrival_orders(b, budgets=a.budgets,
                                                           runs=a.runs)),
        "strata": ("strata", lambda b, a: x.strata(b, budgets=a.budgets,
                                                   runs=a.runs)),
        "perdb": ("perdb", lambda b, a: x.per_database(b, budgets=a.budgets,
                                                       runs=a.runs)),
        "calibration": ("calibration", lambda b, a: x.calibration(
            b, budget=(a.budgets or ["V4"])[0], runs=a.runs or 1)),
        "unprofiled": ("unprofiled", lambda b, a: x.unprofiled_trace(
            b, budget=(a.budgets or ["V4"])[0])),
        "scaling": ("scaling", lambda b, a: x.scaling(b)),
    }


EXPERIMENT_NAMES = ("main", "sweep", "grid", "pooled", "ablation", "sensitivity",
                    "online", "overhead", "scaling", "frontier", "worldmodel",
                    "calibration", "unprofiled", "plans", "scenarios", "orders",
                    "strata", "perdb")


def cmd_exp(args) -> None:
    kind, fn = _exp()[args.name]
    bench = make_bench(args)
    write(args, kind, fn(bench, args))


def cmd_report(args) -> None:
    print(report.render_file(args.path))


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="budgetsql",
                                description="Budget-constrained agentic planning "
                                            "for multi-stage Text-to-SQL")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, trace=True, out=True):
        if trace:
            sp.add_argument("--trace", help="recorded trace corpus (JSONL)")
        sp.add_argument("--questions", type=int, default=0,
                        help="0 takes the workload's own size")
        sp.add_argument("--dbs", type=int, default=0)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--max-states", dest="max_states", type=int, default=32)
        sp.add_argument("--workload", default="bird", choices=sorted(WORKLOADS))
        sp.add_argument("--profile", default="full", choices=sorted(PROFILES))
        sp.add_argument("--cache", default="runs/cache")
        if out:
            sp.add_argument("--out")

    sp = sub.add_parser("simulate", help="freeze a simulated trace corpus to disk")
    common(sp, trace=False, out=False)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_simulate)

    sp = sub.add_parser("collect", help="record a trace corpus by running the grammar")
    sp.add_argument("--dev-json", required=True)
    sp.add_argument("--db-root", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--questions", type=int, default=0)
    sp.add_argument("--max-states", dest="max_states", type=int, default=48)
    sp.add_argument("--sample-rows", dest="sample_rows", type=int, default=0)
    sp.add_argument("--cap", type=float, default=10.0, help="spend cap in USD")
    sp.set_defaults(func=cmd_collect)

    sp = sub.add_parser("library", help="accuracy and cost of the reference workflows")
    common(sp)
    sp.set_defaults(func=cmd_library)

    sp = sub.add_parser("verify", help="exact checks of the escalation algebra")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("run", help="run one method under one budget vector")
    common(sp)
    sp.add_argument("--method", default="budgetsql")
    sp.add_argument("--budget", default="V1", choices=sorted(BUDGETS))
    sp.add_argument("--runs", type=int, default=1)
    sp.add_argument("--no-agent", action="store_true")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("exp", help="run one of the comparisons")
    sp.add_argument("name", choices=sorted(EXPERIMENT_NAMES))
    common(sp)
    sp.add_argument("--budgets", nargs="*", default=None)
    sp.add_argument("--runs", type=int, default=None)
    sp.add_argument("--days", type=int, default=5)
    sp.add_argument("--day-questions", dest="day_questions", type=int, default=300)
    sp.add_argument("--no-agent", action="store_true")
    sp.set_defaults(func=cmd_exp)

    sp = sub.add_parser("report", help="render a saved result")
    sp.add_argument("path")
    sp.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
