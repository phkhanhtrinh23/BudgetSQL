"""Statistics for comparing two methods on the same questions.

Methods are run on the same question orders and replay the same recorded
outcomes, so their runs are paired.  A comparison's gain on a unit (a database,
or a fold of databases) is the mean over the paired runs of the difference in
accuracy on that unit, and the mean lead weights units by their questions.  The
standard error adds the bootstrap variance over units to the variance of the
answers on which the two methods differ, so it is never smaller than the paired
standard error.  A lead's interval uses Student's t on the units; a contrast
between two arms of the same method uses the paired standard error alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


def t_quantile(units: int, level: float = 0.95) -> float:
    """The two-sided quantile of Student's t on ``units`` units."""
    df = max(1, units - 1)
    return float(sps.t.ppf(0.5 + level / 2.0, df))


@dataclass
class Lead:
    mean: float
    standard_error: float
    lo: float
    hi: float
    wins: int
    units: int

    @property
    def excludes_zero(self) -> bool:
        return self.lo > 0.0 or self.hi < 0.0

    @property
    def half_width(self) -> float:
        return 0.5 * (self.hi - self.lo)

    def as_dict(self) -> dict:
        return {"mean": self.mean, "se": self.standard_error,
                "ci": [self.lo, self.hi], "wins": self.wins, "units": self.units,
                "excludes_zero": self.excludes_zero}


def bootstrap_units(gains: dict[str, float], weights: dict[str, float],
                    resamples: int = 4000, seed: int = 0) -> float:
    units = sorted(gains)
    if len(units) < 2:
        return 0.0
    g = np.array([gains[u] for u in units])
    w = np.array([weights.get(u, 1.0) for u in units], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(units), size=(resamples, len(units)))
    means = (g[idx] * w[idx]).sum(axis=1) / np.maximum(w[idx].sum(axis=1), 1e-12)
    return float(np.std(means, ddof=1))


def paired_se(differ: float, n: int) -> float:
    """The paired standard error of a gain when the two methods differ on a
    ``differ`` share of the questions."""
    if n <= 0:
        return 0.0
    return float(np.sqrt(max(0.0, differ) / n))


def lead(a_runs: list[dict[str, bool]], b_runs: list[dict[str, bool]],
         unit_of: dict[str, str], resamples: int = 4000, seed: int = 0) -> Lead:
    """The lead of ``a`` over ``b`` in accuracy points.

    Each element of ``a_runs``/``b_runs`` maps a question to whether that run
    answered it correctly.
    """
    assert len(a_runs) == len(b_runs) and a_runs, "runs must be paired"
    qids = sorted(set(unit_of))
    units = sorted(set(unit_of.values()))
    per_unit: dict[str, list[float]] = {u: [] for u in units}
    weights: dict[str, float] = {u: 0.0 for u in units}
    differ = 0.0
    total = 0
    for ra, rb in zip(a_runs, b_runs):
        by_unit: dict[str, list[float]] = {u: [] for u in units}
        for q in qids:
            u = unit_of[q]
            da = 1.0 if ra.get(q) else 0.0
            db = 1.0 if rb.get(q) else 0.0
            by_unit[u].append(da - db)
            differ += 1.0 if da != db else 0.0
            total += 1
        for u, vals in by_unit.items():
            if vals:
                per_unit[u].append(float(np.mean(vals)))
    gains = {u: float(np.mean(v)) * 100.0 for u, v in per_unit.items() if v}
    for q in qids:
        weights[unit_of[q]] = weights.get(unit_of[q], 0.0) + 1.0
    wsum = sum(weights.get(u, 0.0) for u in gains) or 1.0
    mean = sum(gains[u] * weights.get(u, 0.0) for u in gains) / wsum
    se_units = bootstrap_units(gains, weights, resamples, seed)
    se_paired = paired_se(differ / max(1, total), len(qids)) * 100.0
    se = float(np.sqrt(se_units ** 2 + se_paired ** 2))
    wins = sum(1 for u in gains if gains[u] > 0)
    half = t_quantile(len(gains)) * se
    return Lead(mean, se, mean - half, mean + half, wins, len(gains))


def accuracy(runs: list[dict[str, bool]]) -> tuple[float, float]:
    """Mean accuracy in points over the runs, with its spread."""
    means = [100.0 * np.mean([1.0 if v else 0.0 for v in r.values()]) for r in runs if r]
    if not means:
        return 0.0, 0.0
    return float(np.mean(means)), float(np.std(means))


def paired_contrast(a_runs: list[dict[str, bool]], b_runs: list[dict[str, bool]]
                    ) -> Lead:
    """A contrast between two arms of the same method on the same units and
    orders, which uses the paired standard error alone."""
    qids = sorted(set().union(*[set(r) for r in a_runs + b_runs]))
    diffs, differ = [], 0.0
    for ra, rb in zip(a_runs, b_runs):
        d = []
        for q in qids:
            da = 1.0 if ra.get(q) else 0.0
            db = 1.0 if rb.get(q) else 0.0
            d.append(da - db)
            differ += 1.0 if da != db else 0.0
        diffs.append(float(np.mean(d)))
    mean = float(np.mean(diffs)) * 100.0
    se = paired_se(differ / max(1, len(qids) * len(a_runs)), len(qids)) * 100.0
    half = 1.96 * se
    return Lead(mean, se, mean - half, mean + half,
                sum(1 for d in diffs if d > 0), len(diffs))
