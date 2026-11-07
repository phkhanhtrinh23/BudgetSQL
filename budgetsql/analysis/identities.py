"""Exact checks of the algebra the analysis rests on.

Each check is run in exact rational arithmetic over finite weighted traces or
finite grids, so a failure is a real failure and not a rounding artefact.  They
double as regression tests for the estimators in this package.
"""
from __future__ import annotations

import math
import random
from fractions import Fraction as F

Trace = list[tuple[F, int, int, F]]     # weight, keep correct, escalate correct, rule


def random_trace(n: int = 12, rng: random.Random | None = None) -> Trace:
    rng = rng or random.Random(0)
    rows: Trace = []
    for _ in range(n):
        w = F(rng.randint(1, 9))
        rows.append((w, rng.randint(0, 1), rng.randint(0, 1),
                     F(rng.randint(0, 8), 8)))
    total = sum(r[0] for r in rows)
    return [(w / total, a, b, s) for w, a, b, s in rows]


def rates(trace: Trace) -> tuple[F, F]:
    f = sum(w for w, a, b, _ in trace if a == 0 and b == 1)
    r = sum(w for w, a, b, _ in trace if a == 1 and b == 0)
    return f, r


def gain_over_random(trace: Trace) -> F:
    """Covariance form of a rule's gain over escalating at random."""
    mean_s = sum(w * s for w, _, _, s in trace)
    fix = [(w, F(1) if (a == 0 and b == 1) else F(0), s) for w, a, b, s in trace]
    reg = [(w, F(1) if (a == 1 and b == 0) else F(0), s) for w, a, b, s in trace]
    cov = lambda rows: (sum(w * e * s for w, e, s in rows)
                        - mean_s * sum(w * e for w, e, _ in rows))
    return cov(fix) - cov(reg)


def youden(trace: Trace, event) -> F | None:
    pos = sum(w for w, a, b, _ in trace if event(a, b))
    neg = sum(w for w, a, b, _ in trace if not event(a, b))
    if pos == 0 or neg == 0:
        return None
    hit = sum(w * s for w, a, b, s in trace if event(a, b)) / pos
    miss = sum(w * s for w, a, b, s in trace if not event(a, b)) / neg
    return hit - miss


def check_identity(trace: Trace) -> bool:
    """gamma(s) = f(1-f) J_fix - r(1-r) J_regression."""
    f, r = rates(trace)
    jf = youden(trace, lambda a, b: a == 0 and b == 1)
    jr = youden(trace, lambda a, b: a == 1 and b == 0)
    lhs = gain_over_random(trace)
    rhs = F(0)
    if jf is not None:
        rhs += f * (1 - f) * jf
    if jr is not None:
        rhs -= r * (1 - r) * jr
    return lhs == rhs


def check_ceiling(trace: Trace) -> bool:
    """No rule gains more than f(1-m+) over the best budget-matched mixture,
    and escalating exactly the fixes attains it."""
    f, r = rates(trace)
    m = f - r
    ceiling = f * (1 - max(m, F(0)))
    oracle = [(w, a, b, F(1) if (a == 0 and b == 1) else F(0)) for w, a, b, _ in trace]
    beta = sum(w * s for w, _, _, s in oracle)
    keep = sum(w * a for w, a, _, _ in trace)
    value = sum(w * (s * b + (1 - s) * a) for w, a, b, s in oracle)
    attained = value - keep - beta * max(m, F(0))
    gain = gain_over_random(trace)
    share = sum(w * s for w, _, _, s in trace)
    matched = gain + share * m - share * max(m, F(0))
    return attained == ceiling and matched <= ceiling


def check_regression_only_example() -> bool:
    """Two equally weighted questions, one a regression and one unaffected:
    escalating only the unaffected one gains a quarter over random escalation
    without any fix at all."""
    trace: Trace = [(F(1, 2), 1, 0, F(0)), (F(1, 2), 1, 1, F(1))]
    return gain_over_random(trace) == F(1, 4)


def check_detector_operating_point(p_keep: F, u: F, v: F, alpha: F,
                                   alpha_prime: F) -> bool:
    """A signal independent of the escalation's answer given the first one's
    gains p(1-p)(u + 1 - v)(alpha - alpha')."""
    rows: Trace = [
        (p_keep * v, 1, 1, alpha_prime),
        (p_keep * (1 - v), 1, 0, alpha_prime),
        ((1 - p_keep) * u, 0, 1, alpha),
        ((1 - p_keep) * (1 - u), 0, 0, alpha),
    ]
    expected = p_keep * (1 - p_keep) * (u + 1 - v) * (alpha - alpha_prime)
    return gain_over_random(rows) == expected


def check_information_term(n: int = 40, rng: random.Random | None = None) -> bool:
    """For two candidates, the per-question information gain is the smaller of
    the total positive and the total negative priced advantage."""
    rng = rng or random.Random(1)
    x = [F(rng.randint(-9, 9), 7) for _ in range(n)]
    gamma = (sum(max(v, F(0)) for v in x) - max(sum(v for v in x), F(0))) / n
    expect = min(sum(max(v, F(0)) for v in x), sum(max(-v, F(0)) for v in x)) / n
    return gamma == expect


def check_common_mode(skill_keep: float, skill_escalate: float,
                      grid: int = 400) -> dict:
    """Under a one-parameter difficulty model the escalation value keeps the
    sign of the skill gap in every state and never exceeds tanh(gap/4)."""
    ell = math.tanh(abs(skill_escalate - skill_keep) / 4.0)
    sign = 1 if skill_escalate > skill_keep else (-1 if skill_escalate < skill_keep else 0)
    worst = 0.0
    bad_sign = 0
    for i in range(grid + 1):
        theta = -8.0 + 16.0 * i / grid
        pk = 1.0 / (1.0 + math.exp(-(skill_keep - theta)))
        pe = 1.0 / (1.0 + math.exp(-(skill_escalate - theta)))
        delta = pe - pk
        worst = max(worst, abs(delta))
        if sign and delta * sign < -1e-15:
            bad_sign += 1
    return {"ell": ell, "max_abs_delta": worst, "within": worst <= ell + 1e-12,
            "sign_violations": bad_sign}


def run_all(seeds: int = 25) -> dict:
    """Run every check and report what passed."""
    out: dict[str, bool | dict] = {}
    ok_identity = ok_ceiling = True
    for s in range(seeds):
        trace = random_trace(12, random.Random(s))
        ok_identity &= check_identity(trace)
        ok_ceiling &= check_ceiling(trace)
    out["escalation_identity"] = ok_identity
    out["ceilings"] = ok_ceiling
    out["regression_only_example"] = check_regression_only_example()
    out["detector_operating_point"] = all(
        check_detector_operating_point(F(a, 10), F(b, 10), F(c, 10), F(d, 10), F(e, 10))
        for a in (3, 5, 7) for b in (2, 6) for c in (4, 8) for d in (9,) for e in (1,))
    out["information_term"] = check_information_term()
    cm = [check_common_mode(a, b) for a, b in
          ((0.0, 0.5), (0.5, 0.0), (-1.0, 1.0), (0.2, 0.2))]
    out["common_mode"] = {"within": all(c["within"] for c in cm),
                          "sign_violations": sum(c["sign_violations"] for c in cm)}
    return out
