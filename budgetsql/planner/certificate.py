"""When calling the agent can still pay for itself (Eq. 7).

After a re-solve, the dual prices say what a unit of each dimension is worth
(lambda) and what each open question is already worth (omega).  Scoring every
composition the grammar still admits from each open question's state gives the
most any new candidate could add,

    Psi = sum_i ( max_{w in W(G)} p-tilde_iw - lambda' c-hat_iw - omega_i )^+ .

If Psi is at most what a turn would cost at those prices, lambda' rho_turn, no
turn that only adds candidates can raise the program's value once the turn is
charged (Prop. 2(i)), so the agent is not called.

The values are the stacked values p-tilde the program allocates.  The stacks
are linear in the rollout's success, so ``scale * p-hat - lambda' c-hat`` is
maximised over every composition of the grammar exactly, by backward induction
over the observation tree; the direct head's share, which reads a whole
workflow, is bounded by its largest value over every workflow description the
grammar allows from the state (``StackedValues.bound_forms``).  The search
expands the tree level by level for many questions at once, so each level costs
one batch of head evaluations, and it skips an action whose most optimistic
value --- certain success at its own expected cost --- cannot beat stopping
where it starts.  Both keep Psi an upper bound of Eq. 7's, so a failed test is
still a proof.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..features import QuestionMeta
from ..grammar import Grammar, State
from ..worldmodel.model import StageWorldModel

#: Questions searched per batch of head evaluations; the search stops between
#: chunks once Psi exceeds what the test needs.
CHUNK = 64


@dataclass
class Certificate:
    psi: float
    turn_price: float
    call_agent: bool
    per_question: dict[str, float] = field(default_factory=dict)
    complete: bool = True        # False when the search stopped early

    def as_dict(self) -> dict:
        return {"psi": self.psi, "turn_price": self.turn_price,
                "call_agent": self.call_agent, "complete": self.complete}


@dataclass
class _Node:
    q: int
    state: State
    s: float
    depth: int
    options: list = field(default_factory=list)   # (price, [(p, child)])
    value: float = 0.0


def best_priced_values(model: StageWorldModel, grammar: Grammar,
                       items: list[tuple[QuestionMeta, State, float]],
                       duals: np.ndarray, scales: list[float],
                       depth: int | None = None) -> list[float]:
    """For each ``(meta, state, s_held)``, the maximum over every composition
    the grammar admits from ``state``, stop included, of
    ``scale * p-hat(w) - duals' c-hat(w)``.  ``depth`` caps the stages searched
    (``None``: the whole grammar)."""
    duals = np.asarray(duals, dtype=float)
    roots = [_Node(i, st, float(s), 0) for i, (_, st, s) in enumerate(items)]
    levels: list[list[_Node]] = [roots]
    while levels[-1]:
        frontier = [nd for nd in levels[-1] if depth is None or nd.depth < depth]
        acts = {id(nd): grammar.actions(nd.state) for nd in frontier}
        need = [(items[nd.q][0], nd.state, a) for nd in frontier for a in acts[id(nd)]]
        if need:
            model.predict_batch(need)
        nxt: list[_Node] = []
        for nd in frontier:
            scale = scales[nd.q]
            optimistic = max(scale, 0.0)
            stop = scale * nd.s
            for a in acts[id(nd)]:
                pred = model.predict(items[nd.q][0], nd.state, a)
                price = float(duals @ pred.mu)
                if optimistic - price <= stop + 1e-12:
                    continue        # cannot beat stopping here, whatever follows
                res = pred.resolve(nd.s)
                kids = []
                for j, obs in enumerate(res.labels):
                    w = float(res.p_obs[j])
                    if w <= 0.0:
                        continue
                    keep = bool(res.keep_new[j]) if model.cfg.use_select else True
                    child = _Node(nd.q, model.child(nd.state, a, obs, keep,
                                                    meta=items[nd.q][0]),
                                  res.s_given(j, keep), nd.depth + 1)
                    kids.append((w, child))
                    nxt.append(child)
                nd.options.append((price, kids))
        levels.append(nxt)
    for level in reversed(levels):
        for nd in level:
            best = scales[nd.q] * nd.s
            for price, kids in nd.options:
                best = max(best, -price + sum(w * c.value for w, c in kids))
            nd.value = best
    return [r.value for r in roots]


def certify(model: StageWorldModel, values, grammar: Grammar,
            open_questions: list[tuple[QuestionMeta, State, float]],
            duals: np.ndarray, omega: dict[str, float],
            turn_worst_case: np.ndarray, depth: int | None = None,
            need: float | None = None) -> Certificate:
    """Compute Psi and the test Psi > lambda' rho_turn.

    ``values`` are the stacked values the program allocates (``None`` scores
    with the raw rollout).  ``need`` lets the search stop once Psi exceeds it;
    by default it is the turn's price."""
    price = float(np.asarray(duals, dtype=float) @ np.asarray(turn_worst_case, dtype=float))
    need = price if need is None else float(need)
    per_q: dict[str, float] = {}
    psi = 0.0
    complete = True
    for start in range(0, len(open_questions), CHUNK):
        chunk = open_questions[start:start + CHUNK]
        metas = [m for m, _, _ in chunk]
        states = [st for _, st, _ in chunk]
        helds = [float(s) for _, _, s in chunk]
        if values is None:
            forms = [(float(np.clip(h, 0.0, 1.0)), 0.0, 1.0, 0.0) for h in helds]
        else:
            forms = values.bound_forms(metas, states, helds, grammar)
        best = best_priced_values(model, grammar, chunk, duals,
                                  [f[2] for f in forms], depth)
        for meta, (stop, const, _, bonus), b in zip(metas, forms, best):
            top = max(stop, const + bonus + b)
            gap = top - float(omega.get(meta.qid, 0.0))
            if gap > 0:
                per_q[meta.qid] = gap
                psi += gap
        if psi > need and start + CHUNK < len(open_questions):
            complete = False
            break
    return Certificate(psi, price, psi > price, per_q, complete)
