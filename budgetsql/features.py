"""The fixed feature summary ``phi(h)`` the world model's heads read."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import TIERS
from .grammar import CTX_LEVELS, EXEC_STATUS, Observables, Stage, State

STAGE_KINDS = ("context", "generate", "execute", "repair", "verify", "escalate")


@dataclass
class QuestionMeta:
    """What is known about a question before anything runs."""

    qid: str
    db: str
    n_question_tokens: int = 0
    n_evidence_tokens: int = 0
    n_schema_tokens: int = 0
    n_tables: int = 0
    prompt_tokens: dict[str, int] = field(default_factory=dict)
    difficulty: str = ""

    def prompt_at(self, level: str) -> int:
        if level in self.prompt_tokens:
            return self.prompt_tokens[level]
        base = self.n_schema_tokens + self.n_question_tokens + self.n_evidence_tokens
        shrink = {"full": 1.0, "prune1": 0.59, "prune2": 0.42}.get(level, 1.0)
        return int(base * shrink)


def _log1p(x: float) -> float:
    return math.log1p(max(0.0, float(x)))


QUESTION_COLUMNS = (
    ["q.log_question_tok", "q.log_evidence_tok", "q.log_schema_tok", "q.n_tables"]
    + [f"q.log_prompt_{lv}" for lv in CTX_LEVELS]
)

HISTORY_COLUMNS = (
    ["h.n_llm_stages", "h.n_calls", "h.depth"]
    + [f"h.n_{t}" for t in TIERS]
    + [f"h.ctx_{lv}" for lv in CTX_LEVELS]
    + [f"h.ans_kind_{k}" for k in STAGE_KINDS]
    + [f"h.ans_tier_{t}" for t in TIERS]
    + ["h.ans_samples", "h.ans_cap", "h.has_answer", "h.executed"]
    + [f"h.status_{s}" for s in EXEC_STATUS]
    + ["h.rows_none", "h.rows_one", "h.rows_many",
       "h.agree", "h.split", "h.n_distinct", "h.log_completion_tok",
       "h.n_earlier_answers", "h.frac_earlier_equal", "h.verified", "h.verify_accept"]
)

COLUMNS = QUESTION_COLUMNS + HISTORY_COLUMNS
N_FEATURES = len(COLUMNS)
IDX = {c: i for i, c in enumerate(COLUMNS)}
QUESTION_SLICE = slice(0, len(QUESTION_COLUMNS))
HISTORY_SLICE = slice(len(QUESTION_COLUMNS), N_FEATURES)


def phi(meta: QuestionMeta, state: State, obs: Observables | None = None,
        use_question: bool = True, use_execution: bool = True) -> np.ndarray:
    """Feature vector of a state: the question part, fixed before anything
    runs, and the history part, including the held answer's observables (the
    agreement of its samples, its completion's length and the agreement of
    earlier answers with it).  ``use_question`` / ``use_execution`` drop one
    part of the summary for the world-model input ablations."""
    x = np.zeros(N_FEATURES, dtype=float)
    obs = obs or state.observables

    if use_question:
        x[IDX["q.log_question_tok"]] = _log1p(meta.n_question_tokens)
        x[IDX["q.log_evidence_tok"]] = _log1p(meta.n_evidence_tokens)
        x[IDX["q.log_schema_tok"]] = _log1p(meta.n_schema_tokens)
        x[IDX["q.n_tables"]] = meta.n_tables
        for lv in CTX_LEVELS:
            x[IDX[f"q.log_prompt_{lv}"]] = _log1p(meta.prompt_at(lv))

    if not use_execution:
        return x

    x[IDX["h.depth"]] = len(state)
    x[IDX["h.n_llm_stages"]] = sum(1 for s, _ in state.history if s.llm_calls)
    x[IDX["h.n_calls"]] = state.llm_calls
    for s, _ in state.history:
        if s.tier:
            x[IDX[f"h.n_{s.tier}"]] += 1
    x[IDX[f"h.ctx_{state.ctx}"]] = 1.0

    a = state.answer_stage
    if a is not None:
        x[IDX["h.has_answer"]] = 1.0
        x[IDX[f"h.ans_kind_{a.kind}"]] = 1.0
        if a.tier:
            x[IDX[f"h.ans_tier_{a.tier}"]] = 1.0
        x[IDX["h.ans_samples"]] = a.samples
        x[IDX["h.ans_cap"]] = a.cap / 1000.0
    if state.executed and state.answer_obs:
        x[IDX["h.executed"]] = 1.0
        o = state.answer_obs
        status = o.split(".")
        head = ".".join(status[:2]) if status[0] == "valid" and len(status) > 1 else status[0]
        if head in EXEC_STATUS:
            x[IDX[f"h.status_{head}"]] = 1.0
        elif o.startswith("valid"):
            x[IDX["h.status_valid.one"]] = 1.0
        if o.startswith("valid.match") or o.startswith("valid.diff"):
            x[IDX["h.status_valid.one"]] = 1.0
        x[IDX["h.rows_none"]] = 1.0 if o.startswith("empty") else 0.0
        x[IDX["h.rows_one"]] = 1.0 if "valid.one" in o else 0.0
        x[IDX["h.rows_many"]] = 1.0 if "valid.many" in o else 0.0
        x[IDX["h.agree"]] = 1.0 if o.endswith("agree") or o.endswith("match") else 0.0
        x[IDX["h.split"]] = 1.0 if o.endswith("split") or o.endswith("diff") else 0.0
        x[IDX["h.n_distinct"]] = obs.n_distinct
        x[IDX["h.log_completion_tok"]] = _log1p(obs.completion_tokens)

    for s, o in state.history:
        if s.kind == "verify":
            x[IDX["h.verified"]] = 1.0
            x[IDX["h.verify_accept"]] = 1.0 if o == "accept" else 0.0

    n_earlier = max(0, state.n_answers - 1)
    x[IDX["h.n_earlier_answers"]] = n_earlier
    x[IDX["h.frac_earlier_equal"]] = obs.earlier_equal / n_earlier if n_earlier else 0.0
    return x


def differential_mask() -> np.ndarray:
    """Columns the fix-and-regression heads lose without differential features:
    the agreement among samples and between answers, and the new answer's
    execution."""
    keep = np.ones(N_FEATURES, dtype=bool)
    for c in ("h.agree", "h.split", "h.n_distinct", "h.frac_earlier_equal",
              "h.n_earlier_answers"):
        keep[IDX[c]] = False
    for s in EXEC_STATUS:
        keep[IDX[f"h.status_{s}"]] = False
    for c in ("h.rows_none", "h.rows_one", "h.rows_many"):
        keep[IDX[c]] = False
    return keep


def stage_descriptor(stage: Stage) -> np.ndarray:
    """A stage's own descriptor, appended to ``phi(h)`` when one head serves
    several stages of a type."""
    v = np.zeros(len(STAGE_KINDS) + len(TIERS) + 2 + len(CTX_LEVELS), dtype=float)
    v[STAGE_KINDS.index(stage.kind)] = 1.0
    if stage.tier:
        v[len(STAGE_KINDS) + TIERS.index(stage.tier)] = 1.0
    v[len(STAGE_KINDS) + len(TIERS)] = stage.samples
    v[len(STAGE_KINDS) + len(TIERS) + 1] = stage.cap / 1000.0
    v[len(STAGE_KINDS) + len(TIERS) + 2 + CTX_LEVELS.index(stage.ctx)] = 1.0
    return v


DESCRIPTOR_WIDTH = len(STAGE_KINDS) + len(TIERS) + 2 + len(CTX_LEVELS)

