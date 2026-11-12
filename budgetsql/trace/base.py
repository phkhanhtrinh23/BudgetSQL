from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..config import DIMS, zeros
from ..features import QuestionMeta
from ..grammar import Stage, State


@dataclass
class StageOutcome:
    obs: str
    cost: np.ndarray = field(default_factory=zeros)
    worst_case: np.ndarray = field(default_factory=zeros)
    answer_correct: bool | None = None
    held_before: bool | None = None
    n_distinct: float = 1.0
    completion_tokens: float = 0.0
    equal_to_held: bool = False
    latency: float = 0.0

    @property
    def is_fix(self) -> bool:
        return (self.answer_correct is True) and (self.held_before is False)

    @property
    def is_regression(self) -> bool:
        return (self.answer_correct is False) and (self.held_before is True)

    def to_json(self) -> dict:
        return {
            "obs": self.obs,
            "cost": {k: float(v) for k, v in zip(DIMS, self.cost)},
            "worst_case": {k: float(v) for k, v in zip(DIMS, self.worst_case)},
            "answer_correct": self.answer_correct,
            "held_before": self.held_before,
            "n_distinct": self.n_distinct,
            "completion_tokens": self.completion_tokens,
            "equal_to_held": self.equal_to_held,
            "latency": self.latency,
        }

    @staticmethod
    def from_json(d: dict) -> "StageOutcome":
        def v(key, default=0.0):
            x = d.get(key) or {}
            return np.array([float(x.get(k, default)) for k in DIMS])

        return StageOutcome(
            obs=d["obs"],
            cost=v("cost"),
            worst_case=v("worst_case"),
            answer_correct=d.get("answer_correct"),
            held_before=d.get("held_before"),
            n_distinct=float(d.get("n_distinct", 1.0)),
            completion_tokens=float(d.get("completion_tokens", 0.0)),
            equal_to_held=bool(d.get("equal_to_held", False)),
            latency=float(d.get("latency", 0.0)),
        )


class TraceSource(Protocol):
    def questions(self) -> list[QuestionMeta]:
        ...

    def outcome(self, meta: QuestionMeta, state: State, stage: Stage) -> StageOutcome | None:
        ...
