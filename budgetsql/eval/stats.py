"""Statistics for comparing two methods on the same questions.

Methods are run on the same question orders and replay the same recorded
outcomes, so their runs are paired.  A comparison's gain on a unit (a database,
or a fold of databases) is the mean over the paired runs of the difference in
accuracy on that unit, and the mean lead weights units by their questions.  A
lead's standard error adds the bootstrap variance over units and the variance
of the modelled effects, and its interval uses Student's t on the
units.  A contrast between two arms of the same method (ablation, online) uses
the paired standard error, plus the effects' variance when only one arm
carries them.

A run maps each question to its correctness; a relaxation (the oracle) maps it
to a fractional value in [0, 1].
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


def _v(x) -> float:
    return float(x) if x is not None else 0.0


def lead(a_runs: list[dict[str, bool]], b_runs: list[dict[str, bool]],
         unit_of: dict[str, str], resamples: int = 4000, seed: int = 0,
         offset: float = 0.0, offset_var: float = 0.0) -> Lead:
    """The lead of ``a`` over ``b`` in accuracy points.

    Each element of ``a_runs``/``b_runs`` maps a question to whether that run
    answered it correctly.  ``offset`` (points) and ``offset_var`` (points
    squared) are the modelled effects ``a`` carries beyond its replay minus
    those ``b`` carries; the offset shifts every unit's gain and its variance
    enters the standard error.
    """
    assert len(a_runs) == len(b_runs) and a_runs, "runs must be paired"
    qids = sorted(set(unit_of))
    units = sorted(set(unit_of.values()))
    per_unit: dict[str, list[float]] = {u: [] for u in units}
    weights: dict[str, float] = {u: 0.0 for u in units}
    for ra, rb in zip(a_runs, b_runs):
        by_unit: dict[str, list[float]] = {u: [] for u in units}
        for q in qids:
            u = unit_of[q]
            da = _v(ra.get(q))
            db = _v(rb.get(q))
            by_unit[u].append(da - db)
        for u, vals in by_unit.items():
            if vals:
                per_unit[u].append(float(np.mean(vals)))
    gains = {u: float(np.mean(v)) * 100.0 + float(offset)
             for u, v in per_unit.items() if v}
    for q in qids:
        weights[unit_of[q]] = weights.get(unit_of[q], 0.0) + 1.0
    wsum = sum(weights.get(u, 0.0) for u in gains) or 1.0
    mean = sum(gains[u] * weights.get(u, 0.0) for u in gains) / wsum
    se_units = bootstrap_units(gains, weights, resamples, seed)
    se = float(np.sqrt(se_units ** 2 + max(0.0, offset_var)))
    wins = sum(1 for u in gains if gains[u] > 0)
    half = t_quantile(len(gains)) * se
    return Lead(mean, se, mean - half, mean + half, wins, len(gains))


def accuracy(runs: list[dict[str, bool]], offset: float = 0.0) -> tuple[float, float]:
    """Mean accuracy in points over the runs, with its spread; ``offset`` adds
    a modelled effect in points."""
    means = [100.0 * np.mean([_v(v) for v in r.values()]) for r in runs if r]
    if not means:
        return 0.0, 0.0
    return float(np.mean(means)) + float(offset), float(np.std(means))


def paired_contrast(a_runs: list[dict[str, bool]], b_runs: list[dict[str, bool]],
                    offset: float = 0.0, offset_var: float = 0.0) -> Lead:
    """A contrast between two arms of the same method on the same units and
    orders, which uses the paired standard error alone, plus the variance of
    the modelled effects that differ between the arms."""
    qids = sorted(set().union(*[set(r) for r in a_runs + b_runs]))
    diffs, differ = [], 0.0
    for ra, rb in zip(a_runs, b_runs):
        d = []
        for q in qids:
            da = _v(ra.get(q))
            db = _v(rb.get(q))
            d.append(da - db)
            differ += abs(da - db)
        diffs.append(float(np.mean(d)))
    mean = float(np.mean(diffs)) * 100.0 + float(offset)
    se = paired_se(differ / max(1, len(qids) * len(a_runs)), len(qids)) * 100.0
    se = float(np.sqrt(se ** 2 + max(0.0, offset_var)))
    half = 1.96 * se
    return Lead(mean, se, mean - half, mean + half,
                sum(1 for d in diffs if d > 0), len(diffs))
