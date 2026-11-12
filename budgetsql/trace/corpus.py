from __future__ import annotations

import random
from dataclasses import dataclass
import numpy as np

from ..features import QuestionMeta, phi
from ..grammar import ANSWER_PRODUCING, REPLACING, Grammar, Stage, State, Workflow
from .base import StageOutcome, TraceSource


@dataclass
class TrainingRow:
    meta: QuestionMeta
    state: State
    stage: Stage
    outcome: StageOutcome
    held_before: bool | None

    @property
    def db(self) -> str:
        return self.meta.db

    def features(self, use_question: bool = True, use_execution: bool = True) -> np.ndarray:
        return phi(self.meta, self.state, use_question=use_question,
                   use_execution=use_execution)


class TraceCorpus:
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

    def run(self, meta: QuestionMeta, state: State, wf: Workflow,
            select=None, held: bool | None = None
            ) -> tuple[State, bool, np.ndarray, list[StageOutcome]]:
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
            keep = True
            if stage.kind in REPLACING and state.has_answer and select is not None:
                keep = bool(select(state, stage, out))
            elif select is not None and hasattr(select, "observe"):
                select.observe(state, stage, out)
            if stage.kind in ANSWER_PRODUCING and (keep or held is None):
                held = out.answer_correct
            state = state.append(stage, out.obs, keep=keep, values=out)
            node = node.child(out.obs)
        return state, bool(held), cost, outs

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
                if seen + len(frontier) < self.max_states:
                    frontier.append((state.append(stage, out.obs, values=out), nxt_held))
                if (stage.kind in REPLACING and state.has_answer
                        and seen + len(frontier) < self.max_states):
                    frontier.append((state.append(stage, out.obs, keep=False, values=out),
                                     held))
        return rows

    def __len__(self) -> int:
        return len(self._questions)


def split_dbs(dbs: list[str], folds: int, seed: int = 0) -> list[list[str]]:
    rng = random.Random(seed)
    order = list(dbs)
    rng.shuffle(order)
    out: list[list[str]] = [[] for _ in range(folds)]
    for i, db in enumerate(order):
        out[i % folds].append(db)
    return [f for f in out if f]
