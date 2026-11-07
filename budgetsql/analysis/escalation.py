"""What a world model can add to a single escalation.

The decision is whether to keep one configuration's answer or to take a larger
one's.  A fix is a question the escalation gets right that the first one got
wrong; a regression is the reverse.  A rule gains over escalating at random by
escalating the fixes and by not escalating the regressions, and either term
alone is enough --- which is why a model that predicts the change is worth more
than one that predicts correctness.

Everything here is measured from a trace that already ran both configurations on
every question, so the ceilings are known before any predictor exists.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EscalationStats:
    """Fix and regression rates of one escalation."""

    fix_rate: float
    regression_rate: float

    @property
    def m(self) -> float:
        """The accuracy change from escalating every question."""
        return self.fix_rate - self.regression_rate

    @property
    def ceiling_matched(self) -> float:
        """The most any rule can gain over the best budget-matched mixture."""
        return self.fix_rate * (1.0 - max(self.m, 0.0))

    @property
    def ceiling_unconstrained(self) -> float:
        """The most any rule can gain over the best mixture at any share."""
        return self.fix_rate - max(self.m, 0.0)

    def ceiling_at_share(self, beta: float) -> float:
        """The most a rule of share ``beta`` can gain over random escalation."""
        return (min(beta, self.fix_rate)
                - max(beta - (1.0 - self.regression_rate), 0.0)
                - beta * self.m)

    def as_dict(self) -> dict:
        return {"fix_rate": self.fix_rate, "regression_rate": self.regression_rate,
                "m": self.m, "ceiling_matched": self.ceiling_matched,
                "ceiling_unconstrained": self.ceiling_unconstrained}


def stats(y_keep: np.ndarray, y_escalate: np.ndarray,
          weights: np.ndarray | None = None) -> EscalationStats:
    keep = np.asarray(y_keep, dtype=float)
    esc = np.asarray(y_escalate, dtype=float)
    w = np.ones_like(keep) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    fix = float(np.sum(w * (1 - keep) * esc))
    reg = float(np.sum(w * keep * (1 - esc)))
    return EscalationStats(fix, reg)


def youden(rule: np.ndarray, event: np.ndarray,
           weights: np.ndarray | None = None) -> float:
    """How well a rule separates an event at its operating point."""
    s = np.asarray(rule, dtype=float)
    e = np.asarray(event, dtype=float)
    w = np.ones_like(s) if weights is None else np.asarray(weights, dtype=float)
    pos, neg = w * e, w * (1 - e)
    if pos.sum() <= 0 or neg.sum() <= 0:
        return 0.0
    return float((s * pos).sum() / pos.sum() - (s * neg).sum() / neg.sum())


def gain_over_random(rule: np.ndarray, y_keep: np.ndarray, y_escalate: np.ndarray,
                     weights: np.ndarray | None = None) -> float:
    """A rule's gain over escalating at random with the same share."""
    s = np.asarray(rule, dtype=float)
    keep = np.asarray(y_keep, dtype=float)
    esc = np.asarray(y_escalate, dtype=float)
    w = np.ones_like(s) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    fix = (1 - keep) * esc
    reg = keep * (1 - esc)
    mean_s = float((w * s).sum())
    cov_f = float((w * s * fix).sum()) - mean_s * float((w * fix).sum())
    cov_r = float((w * s * reg).sum()) - mean_s * float((w * reg).sum())
    return cov_f - cov_r


def identity_terms(rule: np.ndarray, y_keep: np.ndarray, y_escalate: np.ndarray,
                   weights: np.ndarray | None = None) -> dict:
    """The gain written as one term per event."""
    st = stats(y_keep, y_escalate, weights)
    keep = np.asarray(y_keep, dtype=float)
    esc = np.asarray(y_escalate, dtype=float)
    fix = (1 - keep) * esc
    reg = keep * (1 - esc)
    jf = youden(rule, fix, weights)
    jr = youden(rule, reg, weights)
    f, r = st.fix_rate, st.regression_rate
    return {"fix_term": f * (1 - f) * jf, "regression_term": r * (1 - r) * jr,
            "youden_fix": jf, "youden_regression": jr,
            "gain": f * (1 - f) * jf - r * (1 - r) * jr}


def frontier(bins: np.ndarray, y_keep: np.ndarray, y_escalate: np.ndarray,
             train: np.ndarray, test: np.ndarray, share: float | None = None
             ) -> dict:
    """The held-out gain of the rule that escalates the bins worth escalating.

    The value of each bin is estimated on the training rows; the rule escalates
    the bins whose estimate beats the budget-matched mixture and is scored on the
    held-out rows, which estimates the frontier from below.
    """
    bins = np.asarray(bins)
    keep = np.asarray(y_keep, dtype=float)
    esc = np.asarray(y_escalate, dtype=float)
    st_train = stats(keep[train], esc[train])
    threshold = max(st_train.m, 0.0)
    rule = np.zeros(len(bins), dtype=float)
    predicted = 0.0
    for b in np.unique(bins[train]):
        sel = train[bins[train] == b]
        if len(sel) == 0:
            continue
        delta = float(np.mean(esc[sel] - keep[sel]))
        if delta > threshold:
            rule[bins == b] = 1.0
            predicted += len(sel) / len(train) * (delta - threshold)
    beta = float(np.mean(rule[test]))
    if share is not None and beta > share > 0:
        rule = rule * (share / beta)
    st_test = stats(keep[test], esc[test])
    realised = (float(np.mean(rule[test] * esc[test] + (1 - rule[test]) * keep[test]))
                - float(np.mean(keep[test])) - float(np.mean(rule[test])) * max(st_test.m, 0.0))
    n = max(1, len(test))
    se = float(np.std(rule[test] * (esc[test] - keep[test])) / np.sqrt(n))
    return {"predicted": predicted, "realised": realised, "share": beta,
            "standard_error": se, "ceiling": st_test.ceiling_matched}


def logistic_difficulty_bound(skill_keep: float, skill_escalate: float) -> dict:
    """Under a one-parameter difficulty model, a difficulty both configurations
    share cannot change which of them is better, and it bounds how much the
    escalation value can vary between states."""
    gap = skill_escalate - skill_keep
    ell = float(np.tanh(abs(gap) / 4.0))
    return {"gap": gap, "ell": ell, "sign": int(np.sign(gap))}


def frontier_bound(m: float, ell: float) -> float:
    """What the frontier can be once the escalation value is bounded by ``ell``."""
    if m <= 0 or ell <= 0:
        return 0.0
    return float(m * max(0.0, 1.0 - m / ell))


def information_term(priced_gains: np.ndarray) -> float:
    """For two candidates, the gain from giving each question its own choice
    rather than one mixture: the smaller of the total positive and the total
    negative priced advantage."""
    x = np.asarray(priced_gains, dtype=float)
    return float(min(np.mean(np.maximum(x, 0.0)), np.mean(np.maximum(-x, 0.0))))


def paired_standard_error(differ: float, n: int) -> float:
    """The paired standard error of a gain between two plans."""
    if n <= 0:
        return 0.0
    return float(np.sqrt(max(0.0, differ) / n))
