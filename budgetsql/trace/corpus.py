"""The trace corpus: a trace source plus the training rows the heads read."""
from __future__ import annotations

import random
from dataclasses import dataclass
import numpy as np

from ..features import Observables, QuestionMeta, phi
from ..grammar import ANSWER_PRODUCING, Grammar, Stage, State, Workflow
from .base import StageOutcome, TraceSource


@dataclass
class TrainingRow:
    """One (state, stage, outcome) triple of the corpus."""

    meta: QuestionMeta
    state: State
    stage: Stage
    outcome: StageOutcome
    held_before: bool | None

    @property
    def db(self) -> str:
        return self.meta.db

    def features(self, use_question: bool = True, use_execution: bool = True) -> np.ndarray:
        return phi(self.meta, self.state, self.state_observables(),
                   use_question=use_question, use_execution=use_execution)

    def state_observables(self) -> Observables:
        return Observables()


class TraceCorpus:
    """A trace source with the sampling the world model trains on.

    ``training_rows`` walks the branches of the grammar breadth first and keeps
    at most ``max_states`` states per question, so the corpus covers every stage
    type at every depth without enumerating the whole tree.
    """

    def __init__(self, source: TraceSource, grammar: Grammar | None = None,
                 max_states: int = 48, seed: int = 0):
        self.source = source
        self.grammar = grammar or Grammar()
        self.max_states = max_states
        self.seed = seed
        self._questions = source.questions()
        self._by_db: dict[str, list[QuestionMeta]] = {}
        for m in self._questions:
            self._by_db.setdefault(m.db, []).append(m)

    # ---- the workload ----
    @property
    def questions(self) -> list[QuestionMeta]:
        return self._questions

    @property
    def dbs(self) -> list[str]:
        return sorted(self._by_db)

    def questions_of(self, db: str) -> list[QuestionMeta]:
        return list(self._by_db.get(db, []))

    def outcome(self, meta: QuestionMeta, state: State, stage: Stage) -> StageOutcome | None:
        return self.source.outcome(meta, state, stage)

    # ---- branch-faithful replay of one workflow ----
    def run(self, meta: QuestionMeta, state: State, wf: Workflow,
            select=None, held: bool | None = None
            ) -> tuple[State, bool, np.ndarray, list[StageOutcome]]:
        """Execute ``wf`` from ``state`` on the recorded branches.

        ``select(state, stage, outcome)`` decides whether a replacing stage's
        answer is kept; the default keeps it.  Returns the state reached, the
        correctness of the answer held, the cost and the outcomes.
        """
        cost = np.zeros(3)
        outs: list[StageOutcome] = []
        node = wf
        while not node.is_stop:
            stage = node.stage
            out = self.source.outcome(meta, state, stage)
            if out is None:
                break
            out.held_before = held
            cost = cost + out.cost
            outs.append(out)
            if stage.kind in ANSWER_PRODUCING:
                keep = True if select is None else bool(select(state, stage, out))
                if keep or held is None:
                    held = out.answer_correct
            state = state.append(stage, out.obs)
            node = node.child(out.obs)
        return state, bool(held), cost, outs

    # ---- training rows ----
    def training_rows(self, dbs: list[str] | None = None) -> list[TrainingRow]:
        rng = random.Random(self.seed)
        rows: list[TrainingRow] = []
        metas = [m for m in self._questions if dbs is None or m.db in dbs]
        for meta in metas:
            rows.extend(self._rows_for(meta, rng))
        return rows

    def _rows_for(self, meta: QuestionMeta, rng: random.Random) -> list[TrainingRow]:
        rows: list[TrainingRow] = []
        frontier: list[tuple[State, bool | None]] = [(State(), None)]
        seen = 0
        while frontier and seen < self.max_states:
            state, held = frontier.pop(0)
            seen += 1
            actions = self.grammar.actions(state)
            rng.shuffle(actions)
            for stage in actions[: 4 if len(state) else len(actions)]:
                out = self.source.outcome(meta, state, stage)
                if out is None:
                    continue
                out.held_before = held
                rows.append(TrainingRow(meta, state, stage, out, held))
                nxt_held = out.answer_correct if stage.kind in ANSWER_PRODUCING else held
                nxt = state.append(stage, out.obs)
                if seen + len(frontier) < self.max_states:
                    frontier.append((nxt, nxt_held))
        return rows

    def __len__(self) -> int:
        return len(self._questions)


def split_dbs(dbs: list[str], folds: int, seed: int = 0) -> list[list[str]]:
    """Group databases into folds, used where a single database's share of the
    budget could not pay for a top-tier call."""
    rng = random.Random(seed)
    order = list(dbs)
    rng.shuffle(order)
    out: list[list[str]] = [[] for _ in range(folds)]
    for i, db in enumerate(order):
        out[i % folds].append(db)
    return [f for f in out if f]
