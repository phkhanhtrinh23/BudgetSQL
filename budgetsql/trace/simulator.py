"""A branch-faithful trace simulator.

The simulator stands in for a recorded corpus wherever one is not available: it
draws, per question, a shared difficulty and one independent error term per tier
and per sample, so that the three properties the planner relies on are present
in the same shape as in a measured trace --- stochastic outcomes, dependence
between tiers, and an execution status that carries information about a tier's
own error beyond difficulty.  Samples are drawn in a fixed seeded order, so a
cheaper composition is exactly the prefix of a longer one and every branch keeps
its own prompt, sample and cost.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np

from ..config import PRICES, TIERS, vec
from ..features import QuestionMeta
from ..grammar import Stage, State
from .base import StageOutcome


@dataclass(frozen=True)
class SimulatorParams:
    """Tier skill, own-error scale and malformed-output cut, with the per-tier
    prices and latencies of the executors."""

    theta_sd: float = 3.4
    #: How much of a question's difficulty its surface form reveals.  A little:
    #: question and schema length say much less about the answer than one
    #: executed stage does.
    theta_from_text: float = 0.25
    skill: dict[str, float] = field(default_factory=lambda: {"S": -0.711, "M": 0.158, "L": 0.247})
    own_error: dict[str, float] = field(default_factory=lambda: {"S": 2.8, "M": 1.0, "L": 1.0})
    #: How much of a tier's own error a question carries across its samples.
    #: Samples of one model on one question fail together far more often than
    #: independent draws would, which is what makes a vote cheap but weak.
    sample_corr: dict[str, float] = field(default_factory=lambda: {"S": 0.85, "M": 0.88, "L": 0.88})
    malformed_cut: dict[str, float] = field(default_factory=lambda: {"S": 1.3, "M": 1.5, "L": 1.6})
    latency: dict[str, float] = field(default_factory=lambda: {"S": 1.69, "M": 1.54, "L": 1.42})
    repair_cut: float = 1.20
    escalate_cut: float = 0.25
    verify_accept_if_correct: float = 0.78
    verify_accept_if_wrong: float = 0.40
    agree_if_wrong: float = 0.33
    completion_tokens: float = 92.0
    completion_sd: float = 0.45
    cache_discount: float = 0.85
    sql_seconds: float = 0.02
    schema_tokens: tuple[int, int] = (200, 1240)
    questions_per_db: tuple[int, int] = (80, 230)


def _seed(*parts) -> int:
    h = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "big")


@dataclass
class _Latent:
    theta: float
    z: dict[str, list[float]]
    u: dict[str, list[float]]
    wrong_id: dict[str, list[int]]


class SimulatedTrace:
    """A trace source over simulated questions."""

    MAX_SAMPLES = 8

    def __init__(self, n_questions: int = 1530, n_dbs: int = 11,
                 params: SimulatorParams | None = None, seed: int = 0,
                 name: str = "bird"):
        self.p = params or SimulatorParams()
        self.seed = seed
        self.name = name
        self._latent: dict[str, _Latent] = {}
        self._questions = self._make_questions(n_questions, n_dbs)

    # ---- workload ----
    def _make_questions(self, n: int, n_dbs: int) -> list[QuestionMeta]:
        rng = np.random.default_rng(_seed(self.seed, self.name, "workload") % (2**32))
        lo, hi = self.p.schema_tokens
        sizes = rng.integers(lo, hi, size=n_dbs)
        tables = rng.integers(3, 18, size=n_dbs)
        weights = rng.uniform(*self.p.questions_per_db, size=n_dbs)
        counts = np.maximum(1, np.round(weights / weights.sum() * n).astype(int))
        while counts.sum() > n:
            counts[int(np.argmax(counts))] -= 1
        while counts.sum() < n:
            counts[int(np.argmin(counts))] += 1
        out: list[QuestionMeta] = []
        for d in range(n_dbs):
            db = f"db_{d:02d}"
            for i in range(int(counts[d])):
                qid = f"{db}_q{i:04d}"
                qr = np.random.default_rng(_seed(self.seed, qid) % (2**32))
                qt = int(qr.integers(8, 42))
                et = int(qr.integers(0, 30))
                st = int(sizes[d] * qr.uniform(0.85, 1.15))
                meta = QuestionMeta(
                    qid=qid, db=db, n_question_tokens=qt, n_evidence_tokens=et,
                    n_schema_tokens=st, n_tables=int(tables[d]),
                    prompt_tokens={"full": st + qt + et,
                                   "prune1": int((st * 0.52) + qt + et),
                                   "prune2": int((st * 0.33) + qt + et)},
                    difficulty=("simple", "moderate", "challenging")[int(qr.integers(0, 3))],
                )
                out.append(meta)
        return out

    def questions(self) -> list[QuestionMeta]:
        return list(self._questions)

    # ---- latents ----
    def _text_difficulty(self, meta: QuestionMeta) -> float:
        """A standardised score from the question's surface form: longer
        questions over wider schemas are a little harder."""
        lo, hi = self.p.schema_tokens
        mid, half = (lo + hi) / 2.0, max(1.0, (hi - lo) / 2.0)
        schema = (meta.n_schema_tokens - mid) / half
        length = (meta.n_question_tokens - 25.0) / 17.0
        tables = (meta.n_tables - 10.0) / 7.0
        return float(np.clip((schema + length + tables) / 3.0, -2.5, 2.5))

    def _own_error(self, rng) -> dict[str, list[float]]:
        """A tier's error on a question: a shared part plus one draw per sample."""
        out: dict[str, list[float]] = {}
        for t in TIERS:
            r = self.p.sample_corr[t]
            sd = self.p.own_error[t]
            shared = rng.normal(0.0, sd * math.sqrt(r))
            per = rng.normal(0.0, sd * math.sqrt(1.0 - r), self.MAX_SAMPLES)
            out[t] = [float(shared + e) for e in per]
        return out

    def _lat(self, meta: QuestionMeta) -> _Latent:
        got = self._latent.get(meta.qid)
        if got is not None:
            return got
        rng = np.random.default_rng(_seed(self.seed, "latent", meta.qid) % (2**32))
        rho = self.p.theta_from_text
        text = self._text_difficulty(meta)
        theta = self.p.theta_sd * (rho * text + math.sqrt(max(0.0, 1 - rho * rho))
                                   * float(rng.normal()))
        lat = _Latent(
            theta=theta,
            z=self._own_error(rng),
            u={t: list(rng.uniform(0.0, 1.0, self.MAX_SAMPLES)) for t in TIERS},
            wrong_id={t: list(rng.integers(1, 11, self.MAX_SAMPLES)) for t in TIERS},
        )
        self._latent[meta.qid] = lat
        return lat

    def _draw(self, meta: QuestionMeta, tier: str, k: int, cut_bonus: float = 0.0,
              ctx: str = "full") -> tuple[bool, bool, int, float]:
        """One sample of a tier: (correct, malformed, result id, score).

        ``cut_bonus`` widens the malformed cut without making the answer more
        likely to be right: a repair that is shown the error message usually
        returns executable SQL, which need not be the right SQL.
        """
        lat = self._lat(meta)
        k = k % self.MAX_SAMPLES
        penalty = {"full": 0.0, "prune1": 0.18, "prune2": 0.42}.get(ctx, 0.0)
        score = self.p.skill[tier] - lat.theta + lat.z[tier][k] - penalty
        malformed = score < -(self.p.malformed_cut[tier] + cut_bonus)
        correct = (score > 0.0) and not malformed
        rid = 0 if correct else int(lat.wrong_id[tier][k])
        return correct, malformed, rid, score

    def _cut_for(self, kind: str) -> float:
        if kind == "repair":
            return self.p.repair_cut
        if kind == "escalate":
            return self.p.escalate_cut
        return 0.0

    def _used(self, state: State, tier: str) -> int:
        """Samples of a tier already drawn, which fixes the next sample index."""
        return sum(s.samples for s, _ in state.history if s.tier == tier)

    def _used_before(self, state: State, held: Stage) -> int:
        """Samples of the held answer's tier drawn before the stage produced it."""
        last = state.answer_index
        if last < 0:
            for i, (s, _) in enumerate(state.history):
                if s == held:
                    last = i
        if last < 0:
            return 0
        return sum(s.samples for s, _ in state.history[:last] if s.tier == held.tier)

    # ---- costs ----
    def _cost(self, meta: QuestionMeta, state: State, stage: Stage,
              path_seed: int) -> tuple[np.ndarray, np.ndarray, float]:
        if stage.kind == "context":
            return vec(0.0, 0.0, 0.05), vec(0.0, 0.0, 0.2), 0.0
        if stage.kind == "execute":
            sec = self.p.sql_seconds * stage.samples
            return vec(0.0, 0.0, sec), vec(0.0, 0.0, self.p.sql_seconds * 4 * stage.samples), 0.0
        rng = np.random.default_rng(path_seed % (2**32))
        prompt = float(meta.prompt_at(state.ctx if stage.kind != "generate" else stage.ctx))
        if stage.kind == "repair":
            prompt = prompt * 0.35 + 130.0
        elif stage.kind == "verify":
            prompt = prompt * 0.45 + 90.0
        mean = self.p.completion_tokens if stage.kind != "verify" else 24.0
        comp = float(np.clip(rng.lognormal(math.log(mean), self.p.completion_sd),
                             4.0, stage.cap))
        price = PRICES[stage.tier]
        usd = (prompt * price.inp * self.p.cache_discount + comp * price.out) * 1e-6 * stage.samples
        tok = (prompt + comp) * stage.samples
        lat = self.p.latency[stage.tier] * (0.6 if stage.kind == "verify" else 1.0)
        sec = float(max(0.05, rng.normal(lat, 0.18))) * stage.samples
        cost = vec(usd, tok, sec)
        worst = vec((prompt * price.inp + stage.cap * price.out) * 1e-6 * stage.samples,
                    (prompt + stage.cap) * stage.samples,
                    10.0 * stage.samples)
        return cost, worst, comp

    # ---- the trace source ----
    def outcome(self, meta: QuestionMeta, state: State, stage: Stage) -> StageOutcome | None:
        path = _seed(self.seed, meta.qid, state.key, stage.id)
        cost, worst, comp = self._cost(meta, state, stage, path)
        rng = np.random.default_rng(path % (2**32))

        if stage.kind == "context":
            return StageOutcome("ok", cost, worst)

        if stage.kind == "generate":
            k0 = self._used(state, stage.tier)
            draws = [self._draw(meta, stage.tier, k0 + i, ctx=stage.ctx)
                     for i in range(stage.samples)]
            ids = [d[2] for d in draws]
            correct = _majority(ids) == 0
            return StageOutcome("ok", cost, worst, answer_correct=correct,
                                n_distinct=float(len(set(ids))), completion_tokens=comp,
                                latency=float(cost[2]),
                                equal_to_held=False)

        if stage.kind == "execute":
            held = state.answer_stage
            if held is None:
                return None
            if held.tier is None:
                return None
            k0 = self._used_before(state, held)
            cut = self._cut_for(held.kind)
            draws = [self._draw(meta, held.tier, k0 + i, cut_bonus=cut, ctx=held.ctx)
                     for i in range(held.samples)]
            ids = [d[2] for d in draws]
            maj = _majority(ids)
            malformed = draws[ids.index(maj)][1]
            u = float(rng.uniform())
            if malformed:
                status = "error" if u < 0.62 else "empty"
            elif maj == 0:
                status = "valid.many" if u < 0.58 else "valid.one"
            else:
                status = "valid.many" if u < 0.46 else ("valid.one" if u < 0.9 else "empty")
            obs = status
            if stage.samples > 1:
                agree = len(set(ids)) == 1
                obs = f"{status}.{'agree' if agree else 'split'}"
            return StageOutcome(obs, cost, worst,
                                n_distinct=float(len(set(ids))), completion_tokens=comp)

        if stage.kind == "verify":
            correct = self._held_correct(meta, state)
            p_acc = (self.p.verify_accept_if_correct if correct
                     else self.p.verify_accept_if_wrong)
            obs = "accept" if rng.uniform() < p_acc else "doubt"
            return StageOutcome(obs, cost, worst, n_distinct=1.0, completion_tokens=comp)

        if stage.kind in ("repair", "escalate"):
            k0 = self._used(state, stage.tier)
            cut = self._cut_for(stage.kind)
            correct, malformed, rid, _ = self._draw(meta, stage.tier, k0, cut_bonus=cut,
                                                    ctx=state.ctx)
            held_correct = self._held_correct(meta, state)
            held_id = 0 if held_correct else self._held_id(meta, state)
            same = rid == held_id
            u = float(rng.uniform())
            if malformed:
                obs = "error" if u < 0.62 else "empty"
            elif same:
                obs = "valid.match"
            else:
                obs = "valid.diff" if u < 0.93 else "empty"
            return StageOutcome(obs, cost, worst, answer_correct=correct,
                                held_before=held_correct, n_distinct=1.0 + (0 if same else 1),
                                completion_tokens=comp, equal_to_held=same,
                                latency=float(cost[2]))
        return None

    # ---- what the held answer is, replayed from the state ----
    def _held_draws(self, meta: QuestionMeta, state: State):
        held = state.answer_stage
        if held is None or held.tier is None:
            return []
        used_before = self._used_before(state, held)
        cut = self._cut_for(held.kind)
        return [self._draw(meta, held.tier, used_before + i, cut_bonus=cut, ctx=held.ctx)
                for i in range(held.samples)]

    def _held_correct(self, meta: QuestionMeta, state: State) -> bool:
        draws = self._held_draws(meta, state)
        if not draws:
            return False
        return _majority([d[2] for d in draws]) == 0

    def _held_id(self, meta: QuestionMeta, state: State) -> int:
        draws = self._held_draws(meta, state)
        if not draws:
            return -1
        return _majority([d[2] for d in draws])


def _majority(ids: list[int]) -> int:
    if not ids:
        return -1
    counts: dict[int, int] = {}
    for i in ids:
        counts[i] = counts.get(i, 0) + 1
    best = max(counts.values())
    for i in ids:
        if counts[i] == best:
            return i
    return ids[0]
