from __future__ import annotations

import json
from pathlib import Path

from ..config import DIMS


def _row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                     for i, (c, w) in enumerate(zip(cells, widths)))


def table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
              for i in range(len(header))]
    lines = [_row(header, widths), "  ".join("-" * w for w in widths)]
    lines += [_row(r, widths) for r in rows]
    return "\n".join(lines)


def main_table(res: dict) -> str:
    budgets = list(res["budgets"])
    methods = res["methods"]
    header = ["method"] + budgets
    rows = []
    for m in methods:
        cells = [m]
        for b in budgets:
            d = res["budgets"][b]
            if m == "budgetsql":
                cells.append(f"{d['planner']['accuracy']:.2f}")
            else:
                cells.append(f"{d['baselines'][m]['accuracy']:.2f}")
        rows.append(cells)
    out = ["Accuracy (EX %) at matched budgets", table(header, rows), ""]
    lead_rows = []
    for b in budgets:
        d = res["budgets"][b]
        if not d["lead"]:
            continue
        ld = d["lead"]
        lead_rows.append([b, d["strongest_baseline"], f"{ld['mean']:+.2f}",
                          f"[{ld['ci'][0]:+.2f}, {ld['ci'][1]:+.2f}]",
                          f"{ld['wins']}/{ld['units']}"])
    out += ["Lead over the strongest baseline",
            table(["budget", "strongest", "lead", "95% CI", "units won"], lead_rows), ""]
    spend_rows = []
    for b in budgets:
        p = res["budgets"][b]["planner"]
        spend_rows.append([b] + [f"{p['utilisation'][k] * 100:.0f}%" for k in DIMS]
                          + [f"{p['binding_fraction'][k] * 100:.0f}%" for k in DIMS]
                          + [f"{p['planning_share']['usd'] * 100:.1f}%"])
    out += ["Spend, binding dimensions and planning share",
            table(["budget"] + [f"used {k}" for k in DIMS]
                  + [f"priced {k}" for k in DIMS] + ["planning"], spend_rows)]
    return "\n".join(out)


def ablation(res: dict) -> str:
    budgets = list(res["budgets"])
    arms = list(res["budgets"][budgets[0]]["arms"])
    rows = []
    for a in arms:
        cells = [a]
        for b in budgets:
            r = res["budgets"][b]["arms"][a]
            mark = "*" if r["excludes_zero"] else " "
            cells.append(f"{r['change']:+.2f}{mark}")
        rows.append(cells)
    head = ["arm"] + [f"{b} ({res['budgets'][b]['planner_accuracy']:.1f})"
                      for b in budgets]
    return ("Accuracy change when a design choice is replaced or removed\n"
            "(* 95% interval excludes zero; column headers: full planner's accuracy)\n"
            + table(head, rows))


def frontier(res: dict) -> str:
    rows = []
    for pair, d in res["pairs"].items():
        for label, r in d["rules"].items():
            rows.append([pair, label, f"{r['predicted']:+.2f}",
                         f"{r['realised']:+.2f}", f"{r['standard_error']:.2f}",
                         f"{d['ceiling_matched']:.2f}"])
    return ("Held-out gain of escalation rules by what the state holds\n"
            + table(["escalation", "state", "predicted", "held out", "s.e.",
                     "ceiling"], rows))


def world_model(res: dict) -> str:
    rows = []
    for opener, d in res["openers"].items():
        rows.append([opener, f"{d['auroc_before']:.3f}", f"{d['auroc_after']:.3f}",
                     f"{d['auroc_fix']:.3f}", f"{d['auroc_regression']:.3f}",
                     f"{d['auroc_continuation']:.3f}", f"{d['brier_after']:.3f}"])
    out = ["World-model quality",
           table(["opener", "AUROC before", "after", "fix", "regression",
                  "continuation", "Brier"], rows), ""]
    rows = [[o] + [f"{v:.2f}" for v in (d[k] for k in DIMS)]
            for o, d in res["cost_error"].items()]
    out += ["Median relative cost error",
            table(["opener"] + list(DIMS), rows)]
    return "\n".join(out)


def sensitivity(res: dict) -> str:
    out = [f"Sensitivity in {res['budget']}"]
    for key, rows in res["settings"].items():
        out.append(table([key, "accuracy", "turns", "planning share"],
                         [[str(r["value"]), f"{r['accuracy']:.2f}",
                           f"{r['turns']:.1f}",
                           f"{r['planning_share']['usd'] * 100:.1f}%"] for r in rows]))
        out.append("")
    return "\n".join(out)


def overhead(res: dict) -> str:
    rows = [[b, f"{d['planning_share']['usd'] * 100:.1f}%",
             f"{d['planning_share']['tok'] * 100:.1f}%",
             f"{d['planning_share']['sec'] * 100:.1f}%",
             f"{d['turns']:.1f}", f"{d['wall_seconds']:.1f}"]
            for b, d in res["budgets"].items()]
    out = ["Planning overhead",
           table(["budget", "money", "tokens", "time", "turns", "wall s"], rows)]
    if res.get("sessions"):
        out += ["", table(["sessions", "accuracy", "turns", "planning share"],
                          [[str(r["requested"]), f"{r['accuracy']:.2f}",
                            f"{r['turns']:.1f}",
                            f"{r['planning_share']['usd'] * 100:.1f}%"]
                           for r in res["sessions"]])]
    if res.get("batches"):
        out += ["", table(["batch", "accuracy", "turns", "planning share"],
                          [[str(r["n"]), f"{r['accuracy']:.2f}", f"{r['turns']}",
                            f"{r['planning_share']['usd'] * 100:.1f}%"]
                           for r in res["batches"]])]
    return "\n".join(out)


def online(res: dict) -> str:
    arms = ("frozen", "updated", "recalibrated", "labelled")
    rows = []
    for r in res["days"]:
        cells = [str(r["day"])]
        cells += [f"{r[a]['accuracy']:.2f}" if a in r else "-" for a in arms]
        cells += [f"{r.get(f'gain_{a}', 0.0):+.2f}" for a in arms[1:]]
        cells.append(f"{r['updated']['cancelled_per_1k']:.1f}")
        rows.append(cells)
    return ("Online cost updates, within-batch recalibration and late labels\n"
            + table(["day"] + list(arms) + [f"gain {a}" for a in arms[1:]]
                    + ["cancelled/1k"], rows))


def scenarios(res: dict) -> str:
    rows = [[k, f"{d['accuracy']:.2f}", d["strongest_baseline"],
             f"{d['baseline_accuracy']:.2f}", f"{d['lead']['mean']:+.2f}",
             f"[{d['lead']['ci'][0]:+.2f}, {d['lead']['ci'][1]:+.2f}]"]
            for k, d in res["configurations"].items()]
    return ("Open-weight agents and executors\n"
            + table(["configuration", "planner", "strongest", "baseline", "lead",
                     "95% CI"], rows))


def library(rows: list[dict]) -> str:
    return ("Reference workflows\n"
            + table(["workflow", "EX %", "$/1k", "tokens", "sec"],
                    [[r["name"], f"{r['accuracy']:.2f}", f"{r['usd_per_1k']:.3f}",
                      f"{r['tokens']:.0f}", f"{r['seconds']:.2f}"] for r in rows]))


def sweep(res: dict) -> str:
    money = res["money"]
    rows = [[m] + [f"{v:.2f}" for v in curve] for m, curve in res["curves"].items()]
    return ("Accuracy against the money budget, tokens and time unbounded\n"
            + table(["method"] + [f"{u * 1e3:g} m$" for u in money], rows))


def grid(res: dict) -> str:
    out = []
    for pair, d in res["pairs"].items():
        x, y = pair.split("x")
        rows = [[f"{c[x]:g}", f"{c[y]:g}", f"{c['accuracy']:.2f}",
                 c["strongest_baseline"], f"{c['lead']['mean']:+.2f}",
                 "*" if c["lead"]["excludes_zero"] else ""]
                for c in d["cells"]]
        out += [f"Lead when {x} crosses {y} "
                f"({d['excludes_zero']}/{d['n']} cells exclude zero)",
                table([x, y, "planner", "strongest", "lead", ""], rows), ""]
    return "\n".join(out)


def pooled(res: dict) -> str:
    rows = []
    for b, d in res["budgets"].items():
        ld = d["lead"] or {"mean": 0.0, "ci": [0.0, 0.0]}
        rows.append([b, f"{d['planner']['accuracy']:.2f}",
                     d["strongest_baseline"] or "-",
                     f"{ld['mean']:+.2f}",
                     f"[{ld['ci'][0]:+.2f}, {ld['ci'][1]:+.2f}]"])
    return (f"One budget for all {res['n_questions']} questions\n"
            + table(["budget", "planner", "strongest", "lead", "95% CI"], rows))


def orders(res: dict) -> str:
    rows = []
    for b, d in res["budgets"].items():
        for order, r in d.items():
            rows.append([b, order, f"{r['accuracy']:.2f}", r["strongest_baseline"],
                         f"{r['lead']['mean']:+.2f}",
                         f"{r['change_vs_random']:+.2f}"])
    return ("Lead under other arrival orders\n"
            + table(["budget", "order", "planner", "strongest", "lead",
                     "vs random"], rows))


def strata(res: dict) -> str:
    out = []
    for b, d in res["budgets"].items():
        rows = [[kind, label, f"{r['share'] * 100:.0f}%", f"{r['accuracy']:.2f}",
                 f"{r['baseline']:.2f}", f"{r['gain']:+.2f}"]
                for kind, labels in d.items() for label, r in labels.items()]
        out += [f"Accuracy by stratum in {b}",
                table(["stratum", "label", "share", "planner", "best baseline",
                       "gain"], rows), ""]
    return "\n".join(out)


def perdb(res: dict) -> str:
    out = []
    for b, d in res["budgets"].items():
        rows = [[g, f"{r['accuracy']:.2f}", r["strongest_baseline"],
                 f"{r['baseline_accuracy']:.2f}", f"{r['gain']:+.2f}"]
                for g, r in d["groups"].items()]
        out += [f"{b}: ahead on {d['ahead']}/{d['units']} budget groups",
                table(["group", "planner", "strongest", "baseline", "gain"],
                      rows), ""]
    return "\n".join(out)


def calibration(res: dict) -> str:
    rows = [[f"{r['lo']:.1f}-{r['hi']:.1f}", f"{r['predicted']:.3f}",
             f"{r['realised']:.3f}", str(r["n"])] for r in res["bins"]]
    return (f"Reliability of the selected continuations in {res['budget']} "
            f"(Brier {res['brier']:.3f}, ECE {res['ece']:.3f})\n"
            + table(["predicted", "mean predicted", "realised", "n"], rows))


def unprofiled(res: dict) -> str:
    rows = [[k, f"{r['auroc_full_trace']:.3f}", f"{r['auroc_thinned']:.3f}",
             f"{r['accuracy_full_trace']:.2f}", f"{r['accuracy_thinned']:.2f}",
             f"{r['drop']:+.2f}"]
            for k, r in res["arms"].items()]
    return (f"A corpus that traces one composition per question "
            f"({res['points_per_auroc']:.1f} points per AUROC)\n"
            + table(["model", "AUROC full", "AUROC thinned", "EX full",
                     "EX thinned", "drop"], rows))


def scaling(res: dict) -> str:
    rows = []
    for r in res["rows"]:
        whole = "-" if r["whole_s"] is None else (
            f"{r['whole_s']:.1f}" + ("" if r["whole_done"] else " (cut)"))
        rows.append([str(r["n"]), f"{r['score_s']:.1f}", str(r["groups"]),
                     f"{r['group_total_s']:.1f}", f"{r['group_ms']:.0f}", whole])
    return (f"Local CPU time of planning, groups of {res['group_size']} "
            f"questions, whole-batch limit {res['limit_s']:g}s\n"
            + table(["questions", "score s", "groups", "group LPs s",
                     "per group ms", "whole batch s"], rows))


RENDERERS = {
    "main": main_table, "ablation": ablation, "frontier": frontier,
    "world_model": world_model, "sensitivity": sensitivity, "overhead": overhead,
    "online": online, "scenarios": scenarios, "library": library, "sweep": sweep,
    "grid": grid, "pooled": pooled, "orders": orders, "strata": strata,
    "perdb": perdb, "calibration": calibration, "unprofiled": unprofiled,
    "scaling": scaling,
}


def render(kind: str, payload) -> str:
    fn = RENDERERS.get(kind)
    if fn is None:
        return json.dumps(payload, indent=2)
    return fn(payload)


def render_file(path: str | Path) -> str:
    obj = json.loads(Path(path).read_text())
    kind = obj.get("kind") or Path(path).stem
    return render(kind, obj.get("result", obj))
