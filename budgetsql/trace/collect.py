from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import vec
from ..executors import prompts
from ..executors.llm import ChatClient, extract_sql
from ..executors.schema import Schema, prune, read_schema, render
from ..executors.sql import ExecResult, execute, matches
from ..features import QuestionMeta
from ..grammar import REPLACING, Grammar, Stage, State
from .base import StageOutcome
from .recorded import branch_key, write_jsonl


@dataclass
class Question:
    qid: str
    db: str
    db_path: str
    question: str
    evidence: str = ""
    reference_sql: str = ""
    difficulty: str = ""


def load_bird(dev_json: str | Path, db_root: str | Path,
              limit: int | None = None) -> list[Question]:
    items = json.loads(Path(dev_json).read_text())
    out: list[Question] = []
    for i, it in enumerate(items):
        db = it.get("db_id") or it.get("db")
        path = Path(db_root) / db / f"{db}.sqlite"
        if not path.exists():
            alt = list(Path(db_root).glob(f"**/{db}.sqlite"))
            if not alt:
                continue
            path = alt[0]
        out.append(Question(
            qid=str(it.get("question_id", i)), db=db, db_path=str(path),
            question=it.get("question", ""), evidence=it.get("evidence", "") or "",
            reference_sql=it.get("SQL") or it.get("query") or "",
            difficulty=it.get("difficulty", "") or ""))
        if limit and len(out) >= limit:
            break
    return out


@dataclass
class _Answer:
    sql: str
    result: ExecResult
    correct: bool


@dataclass
class Collector:
    tiers: dict[str, ChatClient]
    grammar: Grammar = field(default_factory=Grammar)
    max_states: int = 48
    sample_rows: int = 0
    sql_timeout: float = 10.0
    spend_cap_usd: float = 10.0
    verbose: bool = True

    def __post_init__(self) -> None:
        self.spent = 0.0
        self._schemas: dict[str, Schema] = {}
        self._samples: dict[tuple, dict] = {}

    def schema(self, q: Question) -> Schema:
        got = self._schemas.get(q.db_path)
        if got is None:
            got = read_schema(q.db_path, self.sample_rows)
            self._schemas[q.db_path] = got
        return got

    def schema_text(self, q: Question, level: str) -> str:
        s = self.schema(q)
        if level != "full":
            s = prune(s, q.question, q.evidence, level)
        return render(s, self.sample_rows)

    def meta(self, q: Question) -> QuestionMeta:
        s = self.schema(q)
        full = render(s, self.sample_rows)
        tok = lambda t: max(1, len(t) // 4)
        return QuestionMeta(
            qid=q.qid, db=q.db,
            n_question_tokens=tok(q.question), n_evidence_tokens=tok(q.evidence),
            n_schema_tokens=tok(full), n_tables=s.n_tables,
            prompt_tokens={lv: tok(self.schema_text(q, lv)) + tok(q.question)
                                + tok(q.evidence) for lv in ("full", "prune1", "prune2")},
            difficulty=q.difficulty)

    def reference(self, q: Question) -> ExecResult:
        return execute(q.db_path, q.reference_sql, self.sql_timeout)

    def _sample(self, q: Question, tier: str, ctx: str, cap: int, k: int) -> dict:
        key = (q.qid, tier, ctx, cap, k)
        got = self._samples.get(key)
        if got is not None:
            return got
        client = self.tiers[tier]
        msgs = prompts.generate(self.schema_text(q, ctx), q.question, q.evidence,
                                self.sample_rows)
        reply = client.chat(msgs, max_tokens=cap, temperature=0.0 if k == 0 else 0.8,
                            seed=k)
        self.spent += reply["usd"]
        out = {"sql": extract_sql(reply["text"]), **reply}
        self._samples[key] = out
        return out

    def _executed(self, q: Question, draw: dict) -> ExecResult:
        got = draw.get("exec")
        if got is None:
            got = execute(q.db_path, draw["sql"], self.sql_timeout)
            draw["exec"] = got
        return got

    def run_stage(self, q: Question, ref: ExecResult, state: State, stage: Stage,
                  held: _Answer | None) -> tuple[StageOutcome, _Answer | None]:
        if stage.kind == "context":
            return StageOutcome("ok", vec(0, 0, 0.05), vec(0, 0, 0.2)), held

        if stage.kind == "generate":
            used = sum(s.samples for s, _ in state.history if s.tier == stage.tier)
            draws = [self._sample(q, stage.tier, stage.ctx, stage.cap, used + i)
                     for i in range(stage.samples)]
            answers = []
            for d in draws:
                res = self._executed(q, d)
                answers.append(_Answer(d["sql"], res, matches(res, ref)))
            pick = _vote(answers)
            cost = vec(sum(d["usd"] for d in draws),
                       sum(d["prompt_tokens"] + d["completion_tokens"] for d in draws),
                       sum(d["latency"] for d in draws))
            worst = _worst(draws, stage)
            out = StageOutcome("ok", cost, worst, answer_correct=pick.correct,
                               n_distinct=float(len({_key(a) for a in answers})),
                               completion_tokens=float(np.mean(
                                   [d["completion_tokens"] for d in draws])),
                               latency=float(cost[2]))
            return out, pick

        if stage.kind == "execute":
            if held is None:
                return StageOutcome("error", vec(0, 0, 0.0), vec(0, 0, 0.0)), held
            status = held.result.status
            obs = status
            seconds = held.result.seconds
            n_distinct = 1.0
            if stage.samples > 1:
                used = sum(s.samples for s, _ in state.history
                           if s.tier == state.answer_stage.tier) - stage.samples
                draws = [self._sample(q, state.answer_stage.tier,
                                      state.answer_stage.ctx,
                                      state.answer_stage.cap, used + i)
                         for i in range(stage.samples)]
                results = [self._executed(q, d) for d in draws]
                from ..executors.sql import result_key
                keys = {result_key(r.rows) for r in results}
                n_distinct = float(len(keys))
                obs = f"{status}.{'agree' if len(keys) == 1 else 'split'}"
                seconds = sum(r.seconds for r in results)
            return StageOutcome(obs, vec(0, 0, seconds),
                                vec(0, 0, self.sql_timeout * stage.samples),
                                n_distinct=n_distinct), held

        if stage.kind == "verify":
            if held is None:
                return StageOutcome("doubt", vec(0, 0, 0.0), vec(0, 0, 0.0)), held
            client = self.tiers[stage.tier]
            reply = client.chat(prompts.verify(q.question, held.sql,
                                               held.result.preview()),
                                max_tokens=stage.cap)
            self.spent += reply["usd"]
            obs = "accept" if "ACCEPT" in (reply["text"] or "").upper() else "doubt"
            cost = vec(reply["usd"], reply["prompt_tokens"] + reply["completion_tokens"],
                       reply["latency"])
            return StageOutcome(obs, cost, _worst([reply], stage),
                                completion_tokens=float(reply["completion_tokens"])), held

        if stage.kind in ("repair", "escalate"):
            client = self.tiers[stage.tier]
            if stage.kind == "repair" and held is not None:
                msgs = prompts.repair(self.schema_text(q, state.ctx), q.question,
                                      held.sql, held.result.preview(), q.evidence)
            else:
                msgs = prompts.generate(self.schema_text(q, state.ctx), q.question,
                                        q.evidence, self.sample_rows)
            reply = client.chat(msgs, max_tokens=stage.cap)
            self.spent += reply["usd"]
            sql = extract_sql(reply["text"])
            res = execute(q.db_path, sql, self.sql_timeout)
            new = _Answer(sql, res, matches(res, ref))
            same = held is not None and _key(new) == _key(held)
            if not res.ok:
                obs = "error"
            elif res.n_rows == 0:
                obs = "empty"
            else:
                obs = "valid.match" if same else "valid.diff"
            cost = vec(reply["usd"],
                       reply["prompt_tokens"] + reply["completion_tokens"],
                       reply["latency"] + res.seconds)
            out = StageOutcome(obs, cost, _worst([reply], stage),
                               answer_correct=new.correct,
                               held_before=None if held is None else held.correct,
                               completion_tokens=float(reply["completion_tokens"]),
                               equal_to_held=same, latency=float(cost[2]))
            return out, new
        raise ValueError(f"cannot run stage {stage.id}")

    def collect_question(self, q: Question) -> tuple[QuestionMeta, dict]:
        ref = self.reference(q)
        meta = self.meta(q)
        branches: dict[str, dict[str, StageOutcome]] = {}
        frontier: list[tuple[State, _Answer | None]] = [(State(), None)]
        seen = 0
        while frontier and seen < self.max_states:
            state, held = frontier.pop(0)
            seen += 1
            here = branches.setdefault(branch_key(state), {})
            for stage in self.grammar.actions(state):
                if self.spent >= self.spend_cap_usd:
                    return meta, branches
                if stage.id in here:
                    continue
                out, new_held = self.run_stage(q, ref, state, stage, held)
                here[stage.id] = out
                if seen + len(frontier) < self.max_states:
                    frontier.append((state.append(stage, out.obs), new_held))
                if (stage.kind in REPLACING and held is not None
                        and seen + len(frontier) < self.max_states):
                    frontier.append((state.append(stage, out.obs, keep=False), held))
        return meta, branches

    def collect(self, questions: list[Question], out_path: str | Path,
                resume: bool = True) -> dict:
        out_path = Path(out_path)
        done: set[str] = set()
        if resume and out_path.exists():
            for line in out_path.read_text().splitlines():
                if line.strip():
                    done.add(json.loads(line)["qid"])
        started = time.time()
        records: list[tuple[QuestionMeta, dict]] = []
        for i, q in enumerate(questions):
            if q.qid in done:
                continue
            if self.spent >= self.spend_cap_usd:
                break
            records.append(self.collect_question(q))
            if self.verbose and (i % 10 == 0):
                print(f"[{i + 1}/{len(questions)}] spent ${self.spent:.3f} "
                      f"{time.time() - started:.0f}s", flush=True)
            if len(records) >= 25:
                _append(out_path, records)
                records = []
        if records:
            _append(out_path, records)
        return {"questions": len(questions), "written": len(questions) - len(done),
                "spent_usd": self.spent, "seconds": time.time() - started}


def _append(path: Path, records: list) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    write_jsonl(tmp, records)
    with open(path, "a") as fh, open(tmp) as src:
        fh.write(src.read())
    tmp.unlink(missing_ok=True)


def _key(a: _Answer) -> tuple:
    from ..executors.sql import result_key
    return result_key(a.result.rows)


def _vote(answers: list[_Answer]) -> _Answer:
    counts: dict[tuple, int] = {}
    for a in answers:
        counts[_key(a)] = counts.get(_key(a), 0) + 1
    best = max(counts.values())
    for a in answers:
        if counts[_key(a)] == best:
            return a
    return answers[0]


def _worst(replies: list[dict], stage: Stage) -> np.ndarray:
    from ..config import PRICES
    price = PRICES.get(stage.tier)
    prompt = float(np.mean([r.get("prompt_tokens", 0) for r in replies]) or 0.0)
    n = max(1, stage.samples)
    if price is None:
        return vec(0, 0, 0)
    return vec((prompt * price.inp + stage.cap * price.out) * 1e-6 * n,
               (prompt + stage.cap) * n, 10.0 * n)
