from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..features import QuestionMeta
from ..grammar import Grammar, State
from ..worldmodel.model import StageWorldModel

CHUNK = 64


@dataclass
class Certificate:
    psi: float
    turn_price: float
    call_agent: bool
    per_question: dict[str, float] = field(default_factory=dict)
    complete: bool = True

    def as_dict(self) -> dict:
        return {"psi": self.psi, "turn_price": self.turn_price,
                "call_agent": self.call_agent, "complete": self.complete}


@dataclass
class _Node:
    q: int
    state: State
    s: float
    depth: int
    options: list = field(default_factory=list)
    value: float = 0.0


def best_priced_values(model: StageWorldModel, grammar: Grammar,
                       items: list[tuple[QuestionMeta, State, float]],
                       duals: np.ndarray, scales: list[float],
                       depth: int | None = None) -> list[float]:
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
                    continue
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
