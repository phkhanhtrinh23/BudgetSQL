from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import zeros
from ..grammar import State, Workflow, parse_chain
from ..trace.corpus import TraceCorpus

DEFAULT_CONTINUATIONS = ("stop", "rep:L", "rep:M", "esc:M",
                         "esc:L", "ver:M,esc:L")


@dataclass
class Profile:
    acc: float = 0.0
    cost: np.ndarray = field(default_factory=zeros)
    n: int = 0

    def update(self, ok: bool, cost: np.ndarray) -> None:
        self.n += 1
        w = 1.0 / self.n
        self.acc += w * (float(ok) - self.acc)
        self.cost = self.cost + w * (np.asarray(cost, dtype=float) - self.cost)


def profile_library(corpus: TraceCorpus, dbs: list[str],
                    library: dict[str, Workflow]) -> dict[str, Profile]:
    out = {name: Profile() for name in library}
    metas = [m for d in dbs for m in corpus.questions_of(d)]
    for name, wf in library.items():
        for meta in metas:
            _, ok, cost, _ = corpus.run(meta, State(), wf)
            out[name].update(ok, cost)
    return out


def profile_continuations(corpus: TraceCorpus, dbs: list[str],
                          openers: dict[str, Workflow],
                          specs: tuple[str, ...] = DEFAULT_CONTINUATIONS
                          ) -> dict[tuple[str, str, str], Profile]:
    out: dict[tuple[str, str, str], Profile] = {}
    metas = [m for d in dbs for m in corpus.questions_of(d)]
    chains = {s: (None if s == "stop" else parse_chain(s)) for s in specs}
    for oname, opener in openers.items():
        for meta in metas:
            state, ok, cost, _ = corpus.run(meta, State(), opener)
            obs = state.answer_obs or "-"
            for spec, chain in chains.items():
                key = (oname, obs, spec)
                p = out.setdefault(key, Profile())
                if chain is None:
                    p.update(ok, zeros())
                    continue
                if not corpus.grammar.is_wellformed(chain, state):
                    continue
                _, ok2, cost2, _ = corpus.run(meta, state, chain, held=ok)
                p.update(ok2, cost2)
    return out


def lookup(table: dict[tuple[str, str, str], Profile], opener: str, obs: str,
           spec: str) -> Profile | None:
    got = table.get((opener, obs, spec))
    if got is not None and got.n > 0:
        return got
    pooled = Profile()
    for (o, _, s), p in table.items():
        if o == opener and s == spec and p.n:
            pooled.acc += p.acc * p.n
            pooled.cost = pooled.cost + p.cost * p.n
            pooled.n += p.n
    if pooled.n:
        pooled.acc /= pooled.n
        pooled.cost = pooled.cost / pooled.n
        return pooled
    return None
