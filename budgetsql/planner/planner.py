"""The planner: plan a batch, run it under the meter, re-plan when it pays.

A cheap first stage runs on every question, planned rather than fixed.  The
world model then reads what each first stage returned and predicts which further
stages would fix the held answer, which would break it, and what each would
cost.  The agent composes candidates from a fixed-size digest, the allocation
program splits the budget among them, and its dual prices order the
continuations, decide when the agent is worth calling again and size the
reserve.  One meter charges every call, the agent's own included.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config import DIMS, PRICES, PlannerConfig, zeros
from ..features import QuestionMeta
from ..grammar import (LIBRARY_ORDER, STOP, Grammar, State, Workflow, branch,
                       parse_chain, reference_library, shape)
from ..trace.corpus import TraceCorpus
from ..worldmodel.model import StageWorldModel
from ..worldmodel.stacking import StackedValues
from . import reserve as reserve_mod
from .allocation import Allocation, Candidate, greedy, solve
from .certificate import certify
from .digest import CellCandidate, Digest, DigestCell, merge_cells
from .execute import ExecutionContext, first_block, run_workflow
from .meter import Meter

#: The compositions the opening solve chooses a first stage from.
OPENING_CANDIDATES = tuple(LIBRARY_ORDER)

@dataclass
class QuestionRecord:
    meta: QuestionMeta
    state: State = field(default_factory=State)
    held_correct: bool = False
    s_held: float = 0.0
    closed: bool = False
    first_obs: str = "-"
    spend: np.ndarray = field(default_factory=zeros)
    predicted: np.ndarray = field(default_factory=zeros)
    n_stages: int = 0
    cell: str = ""
    planned: Workflow | None = None     # what the last allocation chose for it
    blocked: set[str] = field(default_factory=set)   # candidates the meter refused

    @property
    def qid(self) -> str:
        return self.meta.qid

    @property
    def db(self) -> str:
        return self.meta.db

    def drift(self) -> np.ndarray:
        p = np.maximum(self.predicted, 1e-12)
        return (self.spend - self.predicted) / p


@dataclass
class RunResult:
    accuracy: float
    correct: dict[str, bool]
    spend: np.ndarray
    budget: np.ndarray
    meter: dict
    turns: int = 0
    sessions: int = 0
    planning_share: dict[str, float] = field(default_factory=dict)
    duals_history: list[np.ndarray] = field(default_factory=list)
    predicted_accuracy: float = 0.0
    wall_seconds: float = 0.0
    cancelled: int = 0
    refused: int = 0
    n_questions: int = 0

    def binding_fraction(self) -> dict[str, float]:
        if not self.duals_history:
            return {k: 0.0 for k in DIMS}
        arr = np.stack(self.duals_history)
        return {k: float((arr[:, i] > 0).mean()) for i, k in enumerate(DIMS)}

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "predicted_accuracy": self.predicted_accuracy,
            "n_questions": self.n_questions,
            "spend": {k: float(v) for k, v in zip(DIMS, self.spend)},
            "budget": {k: float(v) for k, v in zip(DIMS, self.budget)},
            "utilisation": {k: float(s / b) if b else 0.0
                            for k, s, b in zip(DIMS, self.spend, self.budget)},
            "planning_share": self.planning_share,
            "binding_fraction": self.binding_fraction(),
            "turns": self.turns, "sessions": self.sessions,
            "cancelled": self.cancelled, "refused": self.refused,
            "wall_seconds": self.wall_seconds,
        }


class BudgetSQL:
    """Plans and runs one budget group."""

    def __init__(self, values: StackedValues, corpus: TraceCorpus,
                 grammar: Grammar | None = None, cfg: PlannerConfig | None = None,
                 agent=None, library: dict[str, Workflow] | None = None):
        self.values = values
        self.model: StageWorldModel = values.model
        self.corpus = corpus
        self.grammar = grammar or corpus.grammar
        self.cfg = cfg or values.cfg
        self.agent = agent
        self.library = library or reference_library()
        self.meter: Meter | None = None
        self.records: dict[str, QuestionRecord] = {}
        self._cell_candidates: dict[str, dict[str, Workflow]] = {}
        self._turns = 0
        self._sessions = 0
        self._duals: list[np.ndarray] = []
        self._last_alloc: Allocation | None = None

    # ------------------------------------------------------------------ #
    # running a budget group
    # ------------------------------------------------------------------ #
    def run(self, metas: list[QuestionMeta], budget: np.ndarray,
            meter: Meter | None = None) -> RunResult:
        t0 = time.time()
        cfg = self.cfg
        self.meter = meter or Meter(budget, planning_cap=cfg.planning_cap)
        self.records = {m.qid: QuestionRecord(m) for m in metas}
        self._cell_candidates = {}
        self._turns = self._sessions = 0
        self._duals = []

        opening = self._opening_plan()
        predicted = opening.value / max(1, len(metas))
        if cfg.plan_whole_workflows:
            self._run_whole_workflows(opening)
            return self._collect(predicted, t0)
        self._run_first_stages(opening)

        if cfg.use_agent and self.agent is not None:
            self._session(opening_session=True)

        while True:
            alloc = self._solve_continuations()
            self._last_alloc = alloc
            if alloc is None:
                break
            order = reserve_mod.run_order(alloc)[: cfg.replan_block]
            if not order:
                break
            self._run_block(order)
            if cfg.use_agent and self.agent is not None and self._flagged_cells():
                self._session()
            if not cfg.periodic_resolves and not (cfg.use_agent and self.agent):
                break

        return self._collect(predicted, t0)

    def _collect(self, predicted: float, t0: float) -> RunResult:
        correct = {q: r.held_correct for q, r in self.records.items()}
        stats = self.meter.stats
        share = {k: float(p / b) if b else 0.0
                 for k, p, b in zip(DIMS, stats.planning_spend, self.meter.budget)}
        return RunResult(
            accuracy=float(np.mean([1.0 if v else 0.0 for v in correct.values()])) if correct else 0.0,
            correct=correct, spend=stats.spend.copy(), budget=self.meter.budget.copy(),
            meter=self.meter.snapshot(), turns=self._turns, sessions=self._sessions,
            planning_share=share, duals_history=self._duals,
            predicted_accuracy=predicted, wall_seconds=time.time() - t0,
            cancelled=stats.cancelled, refused=stats.refused,
            n_questions=len(self.records),
        )

    # ------------------------------------------------------------------ #
    # planning
    # ------------------------------------------------------------------ #
    def _turn_worst_case(self) -> np.ndarray:
        """What one turn of the agent may cost: its prompt is known and its
        completion is capped, so the worst case is known before the turn."""
        price = PRICES["agent"]
        prompt = float(self.cfg.digest_tokens)
        comp = 400.0
        return np.array([(prompt * price.inp + comp * price.out) * 1e-6,
                         prompt + comp, 6.0])

    def _planning_reserve(self) -> np.ndarray:
        """The budget held back for planning: the worst case of one turn, for as
        long as the certificate can still call the agent."""
        if not self._agent_available():
            return zeros()
        return self._turn_worst_case()

    def _agent_available(self) -> bool:
        if not (self.cfg.use_agent and self.agent is not None):
            return False
        if self._turns >= self.cfg.turns_per_session * self.cfg.max_sessions:
            return False
        cap = self.cfg.planning_cap * self.meter.budget[0]
        return bool(self.meter.stats.planning_spend[0] + self._turn_worst_case()[0]
                    <= cap + 1e-12)

    def _opening_plan(self) -> Allocation:
        """One solve from the empty state over compositions that begin with a
        single call, scored on the question part of the state."""
        names = [n for n in OPENING_CANDIDATES if n in self.library]
        wfs = [self.library[n] for n in names]
        recs = list(self.records.values())
        cands: list[Candidate] = []
        items = [(r.meta, r.state, wf) for wf in wfs for r in recs]
        rolls = self.model.rollout_many(items, [0.0] * len(items))
        vals = self.values.values([i[0] for i in items], [i[1] for i in items],
                                  [i[2] for i in items], rolls,
                                  [0.0] * len(items))
        k = 0
        for wf in wfs:
            for r in recs:
                roll = rolls[k]
                cands.append(Candidate(r.qid, wf.name or shape(wf), float(vals[k]),
                                       roll.cost, roll.var, roll.worst_case,
                                       is_stop=False, payload=wf))
                k += 1
        for r in recs:
            cands.append(Candidate(r.qid, "stop", 0.0, zeros(), zeros(), zeros(),
                                   is_stop=True, payload=None))
        alloc = self._solve(cands, reserve=self._planning_reserve())
        self._duals.append(alloc.duals.copy())
        return alloc

    def _solve(self, cands: list[Candidate], reserve: np.ndarray | None = None,
               rho: np.ndarray | None = None) -> Allocation:
        rhs = self.meter.remaining.copy()
        if reserve is not None:
            rhs = rhs - np.asarray(reserve, dtype=float)
        if rho is not None:
            rhs = rhs - np.asarray(rho, dtype=float)
        rhs = np.maximum(rhs, 0.0)
        if self.cfg.use_allocation_lp:
            return solve(cands, rhs, priced_dims=self.cfg.priced_dims)
        return greedy(cands, rhs)

    def _run_first_stages(self, alloc: Allocation) -> None:
        chosen = alloc.chosen()
        for qid, cand in chosen.items():
            rec = self.records[qid]
            wf: Workflow | None = cand.payload
            if wf is None or wf.is_stop:
                rec.closed = True
                continue
            report = self._execute(rec, first_block(wf))
            rec.first_obs = rec.state.answer_obs or "-"
            if not report.made_progress and not rec.state.has_answer:
                rec.closed = True
        self._assign_cells()

    def _run_whole_workflows(self, alloc: Allocation) -> None:
        """Give every question its complete workflow from predictions made
        before anything runs."""
        for qid, cand in alloc.chosen().items():
            rec = self.records[qid]
            wf: Workflow | None = cand.payload
            if wf is None or wf.is_stop:
                rec.closed = True
                continue
            self._execute(rec, wf)
            rec.closed = True

    def _scoring_state(self, rec: QuestionRecord) -> tuple[State, float]:
        """The state a candidate is scored in.  By default this is the state the
        question is really in; scoring before the first stage replaces the
        observation it returned by the one the model expected and keeps the
        prior correctness."""
        if not self.cfg.score_from_empty or not rec.state.history:
            return rec.state, rec.s_held
        first_stage, _ = rec.state.history[0]
        pred = self.model.predict(rec.meta, State(), first_stage)
        res = pred.resolve(0.0)
        j = int(np.argmax(res.p_obs))
        blind = State().append(first_stage, res.labels[j])
        for stage, obs in rec.state.history[1:]:
            blind = blind.append(stage, obs)
        return blind, float(res.s_after[j])

    def _candidates(self, rec: QuestionRecord) -> list[Workflow]:
        """What a question may run next: the one-stage edits of the candidate
        the last allocation chose for it, and whatever the agent registered for
        its cell."""
        out: dict[str, Workflow] = {}
        for wf in self.grammar.one_stage_edits(rec.state, rec.planned or STOP):
            out[shape(wf)] = wf
        # a verification only pays together with what it would trigger
        for a in self.grammar.actions(rec.state):
            if a.kind == "verify":
                for tier in ("M", "L"):
                    chain = parse_chain(f"ver:{a.tier},esc:{tier}")
                    if self.grammar.is_wellformed(chain, rec.state):
                        out[shape(chain)] = chain
        for wf in self._cell_candidates.get(rec.cell, {}).values():
            if self.grammar.is_wellformed(wf, rec.state):
                out[shape(wf)] = wf
        keep = [wf for wf in out.values()
                if not wf.is_stop and shape(wf) not in rec.blocked]
        return keep[: self.cfg.max_candidates]

    def _solve_continuations(self, reserve: np.ndarray | None = None) -> Allocation | None:
        open_recs = [r for r in self.records.values() if not r.closed]
        if not open_recs:
            return None
        items: list[tuple[QuestionMeta, State, Workflow]] = []
        owners: list[QuestionRecord] = []
        scoring: list[float] = []
        for r in open_recs:
            state, s_held = self._scoring_state(r)
            # stop is a candidate like any other, and it is valued by the same
            # calibration, so keeping the held answer and paying for another
            # stage are compared on one scale
            for wf in [STOP] + self._candidates(r):
                items.append((r.meta, state, wf))
                owners.append(r)
                scoring.append(s_held)
        if not items:
            return None
        rolls = self.model.rollout_many(items, scoring)
        vals = self.values.values([i[0] for i in items], [i[1] for i in items],
                                  [i[2] for i in items], rolls, scoring)
        cands: list[Candidate] = []
        for (meta, state, wf), r, roll, v in zip(items, owners, rolls, vals):
            cands.append(Candidate(r.qid, shape(wf), float(v), roll.cost, roll.var,
                                   roll.worst_case, is_stop=wf.is_stop,
                                   payload=None if wf.is_stop else wf))

        res = reserve if reserve is not None else self._planning_reserve()
        alloc = self._solve(cands, reserve=res)
        rho = reserve_mod.reserve_for(alloc, self.cfg.replan_block, alloc.duals,
                                      delta=self.cfg.reserve_delta,
                                      z=self.cfg.reserve_z)
        if np.any(rho > 0):
            alloc = self._solve(cands, reserve=res, rho=rho)
        for qid, cand in alloc.chosen().items():
            rec = self.records.get(qid)
            if rec is not None:
                rec.planned = cand.payload
        self._duals.append(alloc.duals.copy())
        return alloc

    def _run_block(self, order: list[Candidate]) -> None:
        touched: set[str] = set()
        for cand in order:
            rec = self.records.get(cand.qid)
            if rec is None or rec.closed:
                continue
            wf: Workflow | None = cand.payload
            if wf is None or wf.is_stop:
                rec.closed = True
                continue
            report = self._execute(rec, wf)
            touched.add(rec.qid)
            if not report.made_progress:
                # the meter would not pay for this one: drop it and let the next
                # solve choose among what is left
                rec.blocked.add(shape(wf))
                rec.planned = None
                if not self._candidates(rec):
                    rec.closed = True
        self._assign_cells()

    # ------------------------------------------------------------------ #
    # executing stages under the meter
    # ------------------------------------------------------------------ #
    def _execute(self, rec: QuestionRecord, wf: Workflow):
        report = run_workflow(self._ctx(), rec, wf)
        if report.exhausted or not self.grammar.actions(rec.state):
            rec.closed = True
        return report

    def _ctx(self) -> ExecutionContext:
        return ExecutionContext(self.corpus, self.meter, self.cfg, self.grammar,
                                self.model)

    # ------------------------------------------------------------------ #
    # digest cells
    # ------------------------------------------------------------------ #
    def _assign_cells(self) -> None:
        open_recs = [r for r in self.records.values() if not r.closed]
        if not open_recs:
            return
        s = np.array([r.s_held for r in open_recs])
        c = np.array([float(r.spend[0]) for r in open_recs])
        deciles = np.linspace(0.1, 0.9, 9)
        s_edges = (np.unique(np.quantile(s, deciles)) if len(s) > len(deciles)
                   else np.array([0.33, 0.66]))
        c_edges = (np.unique(np.quantile(c, deciles)) if len(c) > len(deciles)
                   else np.array([0.0]))
        for r in open_recs:
            si = int(np.searchsorted(s_edges, r.s_held))
            ci = int(np.searchsorted(c_edges, float(r.spend[0])))
            r.cell = f"{r.db}|{_obs_bucket(r.first_obs)}|s{si}c{ci}"

    def digest(self) -> Digest:
        open_recs = [r for r in self.records.values() if not r.closed]
        groups: dict[str, list[QuestionRecord]] = {}
        for r in open_recs:
            groups.setdefault(r.cell, []).append(r)
        cells: list[DigestCell] = []
        for cid, recs in groups.items():
            s = np.array([r.s_held for r in recs])
            drift = np.mean([r.drift() for r in recs], axis=0) if recs else zeros()
            cell = DigestCell(
                cid=cid, db=recs[0].db, obs=_obs_bucket(recs[0].first_obs),
                qids=[r.qid for r in recs], s_mean=float(s.mean()),
                s_spread=float(s.std()),
                cost_mean=np.mean([r.spend for r in recs], axis=0),
                drift=drift,
                flagged=bool(np.max(np.abs(drift)) > self.cfg.drift_threshold),
            )
            cell.candidates = self._cell_summary(recs, cid)
            cell.uncertain = any(c.uncertain for c in cell.candidates)
            cells.append(cell)
        cells = merge_cells(cells, self.cfg.max_digest_cells)
        duals = self._last_alloc.duals if self._last_alloc is not None else zeros()
        stats = self.meter.stats
        return Digest(cells, self.meter.remaining, stats.spend, stats.planning_spend,
                      duals, self.meter.budget, n_open=len(open_recs),
                      turns_used=self._turns,
                      turns_left=max(0, self.cfg.turns_per_session - self._turns))

    def _cell_summary(self, recs: list[QuestionRecord], cid: str) -> list[CellCandidate]:
        sample = recs[: min(8, len(recs))]
        wfs: dict[str, Workflow] = {}
        for wf in self._candidates(sample[0]):
            wfs[shape(wf)] = wf
        for wid, wf in self._cell_candidates.get(cid, {}).items():
            wfs[wid] = wf
        out: list[CellCandidate] = []
        for wid, wf in list(wfs.items())[:6]:
            items = [(r.meta, r.state, wf) for r in sample
                     if self.grammar.is_wellformed(wf, r.state)]
            if not items:
                continue
            rolls = self.model.rollout_many(items, [r.s_held for r in sample[:len(items)]])
            out.append(CellCandidate(
                wid, float(np.mean([r.p for r in rolls])),
                np.mean([r.cost for r in rolls], axis=0),
                float(np.std([r.p for r in rolls])),
                any(r.uncertain for r in rolls)))
        return sorted(out, key=lambda c: -c.p)

    def _flagged_cells(self) -> list[str]:
        """Which cells have drifted far enough from their prediction to be worth
        a look.  This reads the records only, so it costs nothing."""
        groups: dict[str, list[QuestionRecord]] = {}
        for r in self.records.values():
            if not r.closed:
                groups.setdefault(r.cell, []).append(r)
        out = []
        for cid, recs in groups.items():
            drift = np.mean([r.drift() for r in recs], axis=0)
            if float(np.max(np.abs(drift))) > self.cfg.drift_threshold:
                out.append(cid)
        return out

    # ------------------------------------------------------------------ #
    # agent sessions
    # ------------------------------------------------------------------ #
    def _session(self, opening_session: bool = False) -> None:
        from .tools import ToolBox   # imported here: the tools bind to a planner

        self._sessions += 1
        box = ToolBox(self)
        for _ in range(self.cfg.turns_per_session):
            if not self._agent_available():
                break
            if not opening_session and not self._certificate().call_agent:
                break
            digest = self.digest()
            if not digest.cells:
                break
            wc = self._turn_worst_case()
            if self.meter.charge(wc, wc, label="agent turn", planning=True) is None:
                break
            self._turns += 1
            try:
                self.agent.turn(digest, box)
            except Exception:
                break
            opening_session = False

    def _certificate(self):
        alloc = self._last_alloc
        duals = alloc.duals if alloc is not None else zeros()
        omega = {}
        if alloc is not None:
            omega = {q: float(w) for q, w in zip(alloc.qids, alloc.omega)}
        open_recs = [r for r in self.records.values() if not r.closed]
        sample = open_recs[: 200]
        scale = len(open_recs) / max(1, len(sample))
        cert = certify(self.model, self.grammar,
                       [(r.meta, r.state, r.s_held) for r in sample],
                       duals, omega, self._turn_worst_case())
        cert.psi *= scale
        cert.call_agent = cert.psi > cert.turn_price
        return cert

    # ---- used by the tools ----
    def register(self, cell: str, wf: Workflow) -> bool:
        if not isinstance(wf, Workflow) or wf.is_stop:
            return False
        recs = [r for r in self.records.values() if r.cell == cell and not r.closed]
        if not recs:
            return False
        if not any(self.grammar.is_wellformed(wf, r.state) for r in recs):
            return False
        self._cell_candidates.setdefault(cell, {})[shape(wf)] = wf
        return True

    def probe(self, cell: str, stage, n: int) -> dict:
        """Run a stage on a few questions of a cell, to observe a cell whose
        predictions spread widely or rest on few training questions."""
        recs = [r for r in self.records.values() if r.cell == cell and not r.closed]
        picked = recs[: max(0, int(n))]
        before = self.meter.stats.spend.copy()
        seen: dict[str, int] = {}
        wf = branch(stage, {o: None for o in stage.observations()})
        for r in picked:
            if not self.grammar.is_wellformed(wf, r.state):
                continue
            self._execute(r, wf)
            seen[r.state.answer_obs or "-"] = seen.get(r.state.answer_obs or "-", 0) + 1
        return {"probed": len(picked), "observations": seen,
                "cost": (self.meter.stats.spend - before).tolist()}


def _obs_bucket(obs: str) -> str:
    if obs in ("-", None):
        return "-"
    head = obs.split(".")[0]
    tail = obs.split(".")[-1]
    return head if tail in (head, "") else f"{head}.{tail}"


