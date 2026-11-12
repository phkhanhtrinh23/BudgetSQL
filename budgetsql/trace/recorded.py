from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator

from ..features import QuestionMeta
from ..grammar import REPLACING, Stage, State
from .base import StageOutcome

BRANCH_SEP = ";"
PAIR_SEP = "="


def branch_key(state: State) -> str:
    return BRANCH_SEP.join(f"{sid}{PAIR_SEP}{obs}" for sid, obs in state.key)


def _open(path: str | Path, mode: str):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t")
    return open(path, mode)


def write_jsonl(path: str | Path, questions: Iterable[tuple[QuestionMeta, dict]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _open(path, "w") as fh:
        for meta, branches in questions:
            obj = {
                "qid": meta.qid,
                "db": meta.db,
                "difficulty": meta.difficulty,
                "n_question_tokens": meta.n_question_tokens,
                "n_evidence_tokens": meta.n_evidence_tokens,
                "n_schema_tokens": meta.n_schema_tokens,
                "n_tables": meta.n_tables,
                "prompt_tokens": meta.prompt_tokens,
                "branches": {
                    key: {sid: out.to_json() for sid, out in stages.items()}
                    for key, stages in branches.items()
                },
            }
            fh.write(json.dumps(obj) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with _open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


class RecordedTrace:
    """Replays stage outcomes recorded by a collection run."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._meta: list[QuestionMeta] = []
        self._branches: dict[str, dict[str, dict[str, StageOutcome]]] = {}
        for obj in read_jsonl(self.path):
            meta = QuestionMeta(
                qid=obj["qid"], db=obj["db"],
                n_question_tokens=obj.get("n_question_tokens", 0),
                n_evidence_tokens=obj.get("n_evidence_tokens", 0),
                n_schema_tokens=obj.get("n_schema_tokens", 0),
                n_tables=obj.get("n_tables", 0),
                prompt_tokens=obj.get("prompt_tokens") or {},
                difficulty=obj.get("difficulty", ""),
            )
            self._meta.append(meta)
            self._branches[meta.qid] = {
                key: {sid: StageOutcome.from_json(d) for sid, d in stages.items()}
                for key, stages in obj.get("branches", {}).items()
            }

    def questions(self) -> list[QuestionMeta]:
        return list(self._meta)

    def outcome(self, meta: QuestionMeta, state: State, stage: Stage) -> StageOutcome | None:
        stages = self._branches.get(meta.qid, {}).get(branch_key(state))
        if not stages:
            return None
        got = stages.get(stage.id)
        if got is None:
            return None
        return StageOutcome.from_json(got.to_json())

    def coverage(self) -> dict[str, float]:
        n_branch = sum(len(b) for b in self._branches.values())
        n_stage = sum(len(s) for b in self._branches.values() for s in b.values())
        return {"questions": len(self._meta), "branches": n_branch, "stages": n_stage}


def materialise(source, questions: Iterable[QuestionMeta], grammar,
                max_states: int = 64) -> Iterator[tuple[QuestionMeta, dict]]:
    for meta in questions:
        branches: dict[str, dict[str, StageOutcome]] = {}
        frontier = [State()]
        seen = 0
        while frontier and seen < max_states:
            state = frontier.pop(0)
            seen += 1
            key = branch_key(state)
            here = branches.setdefault(key, {})
            for stage in grammar.actions(state):
                out = source.outcome(meta, state, stage)
                if out is None:
                    continue
                here[stage.id] = out
                if seen + len(frontier) < max_states:
                    frontier.append(state.append(stage, out.obs))
                if (stage.kind in REPLACING and state.has_answer
                        and seen + len(frontier) < max_states):
                    frontier.append(state.append(stage, out.obs, keep=False))
        yield meta, branches
