"""When calling the agent can still pay for itself.

After a re-solve, the dual prices say what a unit of each dimension is worth and
what each question is already worth.  Scoring every composition the grammar
still allows from each open question's state gives the most any new candidate
could add; if that total is below what a turn of the agent would cost at those
prices, no turn that only adds candidates can raise the program's value once the
turn is charged, so the agent is not called.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features import QuestionMeta
from ..grammar import Grammar, State
from ..worldmodel.model import StageWorldModel


@dataclass
class Certificate:
    psi: float
    turn_price: float
    call_agent: bool
    per_question: dict[str, float]

    def as_dict(self) -> dict:
        return {"psi": self.psi, "turn_price": self.turn_price,
                "call_agent": self.call_agent}


def best_priced_value(model: StageWorldModel, meta: QuestionMeta, state: State,
                      grammar: Grammar, duals: np.ndarray, s_held: float,
                      depth: int = 2, memo: dict | None = None) -> float:
    """Backward induction over the grammar: the best priced value of any
    composition from this state, stop included.

    Because a rollout's cost is linear in the stage costs, the maximum over
    every composition needs no enumeration.
    """
    memo = {} if memo is None else memo
    key = (meta.qid, state.key, depth, round(s_held, 6))
    got = memo.get(key)
    if got is not None:
        return got
    best = float(s_held)
    if depth > 0:
        actions = grammar.actions(state)
        if actions:
            model.predict_batch([(meta, state, a) for a in actions])
        for a in actions:
            pred = model.predict(meta, state, a)
            res = pred.resolve(s_held)
            val = -float(duals @ pred.mu)
            for j, obs in enumerate(res.labels):
                w = float(res.p_obs[j])
                if w <= 0:
                    continue
                val += w * best_priced_value(model, meta, state.append(a, obs), grammar,
                                             duals, float(res.s_after[j]),
                                             depth - 1, memo)
            best = max(best, val)
    memo[key] = best
    return best


def certify(model: StageWorldModel, grammar: Grammar,
            open_questions: list[tuple[QuestionMeta, State, float]],
            duals: np.ndarray, omega: dict[str, float],
            turn_worst_case: np.ndarray, depth: int = 2) -> Certificate:
    """Compute the certificate and the decision it implies."""
    memo: dict = {}
    per_q: dict[str, float] = {}
    psi = 0.0
    for meta, state, s_held in open_questions:
        best = best_priced_value(model, meta, state, grammar, duals, s_held,
                                 depth=depth, memo=memo)
        gap = best - float(omega.get(meta.qid, 0.0))
        if gap > 0:
            per_q[meta.qid] = gap
            psi += gap
    price = float(duals @ np.asarray(turn_worst_case, dtype=float))
    return Certificate(psi, price, psi > price, per_q)
