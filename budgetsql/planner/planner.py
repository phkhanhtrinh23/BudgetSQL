"""The planner of one budget group: plan it, run it under the meter, re-solve.

A cheap first stage runs on every question, planned rather than fixed.  The
world model then reads what each first stage returned and predicts which further
stages would fix the held answer, which would break it, and what each would
cost.  The allocation program splits the group's budget among the candidates,
and its dual prices order the continuations, size the reserve and price the
certificate.  One meter charges every call, the agent's own included.

The agent is central to the batch rather than to a group: one digest covers
every group and its turns are charged to the groups in proportion to their
questions.  :class:`budgetsql.planner.batch.BatchPlanner` drives the groups'
planners through Alg. 1 and holds the agent's sessions.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config import DIMS, PRICES, PlannerConfig, zeros
from ..features import QuestionMeta
from ..grammar import (LIBRARY_ORDER, STOP, Grammar, State, Workflow, branch,
                       compact, parse_chain, reference_library, shape,
                       single_call_first)
from ..trace.corpus import TraceCorpus
from ..worldmodel.model import StageWorldModel
from ..worldmodel.stacking import StackedValues
from . import reserve as reserve_mod
from .allocation import Allocation, Candidate, greedy, solve
from .certificate import Certificate, certify
from .digest import CellCandidate, Digest, DigestCell
from .execute import ExecutionContext, first_block, pace, run_workflow
from .meter import Meter

#: The compositions the opening solve chooses a first stage from: those that
#: begin with a single call (Sec. IV-C).
OPENING_CANDIDATES = tuple(n for n, wf in reference_library().items()
                           if n in LIBRARY_ORDER and single_call_first(wf))


def remainder(wf: Workflow | None, state: State) -> Workflow:
    """What is left of a composition once a question has run its stages: its
    sub-workflow after the observed branch, or stop if the question left it."""
    node = wf or STOP
    for stage, obs in state.history:
        if node.is_stop or node.stage.id != stage.id:
            return STOP
        node = node.child(obs)
    return node

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
    opening: Workflow | None = None     # the composition its first stage came from
    planned: Workflow | None = None     # what the last allocation chose for it
    #: candidates the agent registered for the cell the question was in when
    #: it registered them; cells are re-formed after every block, so the
    #: candidates stay with the question
    registered: dict[str, Workflow] = field(default_factory=dict)
    #: whether its cell has been flagged, so its candidates' one-stage edits
    #: are candidates too
    edited: bool = False
    #: the stacked value p-tilde of the last continuation it ran, when chosen
    selected_value: float | None = None
    blocked: set[str] = field(default_factory=set)   # candidates the meter refused
    #: Observations of the stages run since the question joined a digest cell:
    #: (stage id, predicted distribution, index observed).
    obs_log: list[tuple[str, np.ndarray, int]] = field(default_factory=list)

    @property
    def qid(self) -> str:
        return self.meta.qid

    @property
    def db(self) -> str:
        return self.meta.db

    def drift(self) -> np.ndarray:
        p = np.maximum(self.predicted, 1e-12)
        return (self.spend - self.predicted) / p

    def observe(self, stage, resolved, j: int) -> None:
        """Record what a stage observed against what was predicted, once the
        question is in a cell; the digest flags a cell whose shares shift."""
        if self.cell and len(resolved.labels) > 1:
            self.obs_log.append((stage.id, np.asarray(resolved.p_obs, dtype=float),
                                 int(j)))


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
    tier_money: dict[str, float] = field(default_factory=dict)

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
            "tier_money": self.tier_money,
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
        self._duals: list[np.ndarray] = []
        self._last_alloc: Allocation | None = None
        self._predicted = 0.0
        self._t0 = 0.0
        #: the coordinator of the batch this group belongs to, and the share of
        #: each agent turn the group pays (its share of the batch's questions)
        self.batch = None
        self.turn_share = 1.0

    # ------------------------------------------------------------------ #
    # running a budget group
    # ------------------------------------------------------------------ #
    def run(self, metas: list[QuestionMeta], budget: np.ndarray,
            meter: Meter | None = None) -> RunResult:
        """Plan and run this group as a batch of its own."""
        from .batch import BatchPlanner
        return BatchPlanner([self], self.cfg, self.agent).run([(metas, budget, meter)])[0]

    def start(self, metas: list[QuestionMeta], budget: np.ndarray,
              meter: Meter | None = None) -> None:
        """Alg. 1, line 2: solve Eq. 6 from h0, run the planned first stages and
        observe them."""
        self._t0 = time.time()
        cfg = self.cfg
        self.meter = meter or Meter(budget, planning_cap=cfg.planning_cap)
        self.records = {m.qid: QuestionRecord(m) for m in metas}
        self._duals = []
        self._last_alloc = None

        opening = self._opening_plan()
        self._predicted = opening.value / max(1, len(metas))
        if cfg.plan_whole_workflows:
            self._run_whole_workflows(opening)
            return
        self._run_first_stages(opening)
        if cfg.recalibrate_success:
            self._recalibrate()

    def resolve(self) -> Allocation | None:
        """Re-solve Eq. 6 over the open questions (Alg. 1, line 7)."""
        alloc = self._solve_continuations()
        self._last_alloc = alloc
        return alloc

    def run_next_block(self, whole_plan: bool = False) -> bool:
        """Run the next K planned continuations of the last solve, or its whole
        plan; whether anything ran."""
        alloc = self._last_alloc
        if alloc is None:
            return False
        order = reserve_mod.run_order(alloc)
        if not whole_plan:
            order = order[: self.cfg.replan_block]
        if not order:
            return False
        self._run_block(order)
        return True

    def finish(self, turns: int = 0, sessions: int = 0) -> RunResult:
        correct = {q: r.held_correct for q, r in self.records.items()}
        stats = self.meter.stats
        share = {k: float(p / b) if b else 0.0
                 for k, p, b in zip(DIMS, stats.planning_spend, self.meter.budget)}
        return RunResult(
            accuracy=float(np.mean([1.0 if v else 0.0 for v in correct.values()])) if correct else 0.0,
            correct=correct, spend=stats.spend.copy(), budget=self.meter.budget.copy(),
            meter=self.meter.snapshot(), turns=turns, sessions=sessions,
            planning_share=share, duals_history=self._duals,
            predicted_accuracy=self._predicted, wall_seconds=time.time() - self._t0,
            cancelled=stats.cancelled, refused=stats.refused,
            n_questions=len(self.records),
            tier_money={t: float(v[0]) for t, v in stats.tier_spend.items()},
        )

    @property
    def is_open(self) -> bool:
        return any(not r.closed for r in self.records.values())

    # ------------------------------------------------------------------ #
    # planning
    # ------------------------------------------------------------------ #
    def turn_worst_case(self) -> np.ndarray:
        """What one agent turn may cost the whole batch: its prompt is the
        digest and its completion and time are capped, so rho_turn is known
        before the turn."""
        from .agent import PROMPT_OVERHEAD_TOKENS
        price = PRICES["agent"]
        # the fixed part of the prompt (system, tool spec, message framing),
        # plus the digest, whose size is capped
        prompt = float(PROMPT_OVERHEAD_TOKENS + self.cfg.digest_tokens
                       + self.cfg.agent_result_tokens)
        comp = float(self.cfg.agent_completion_tokens)
        return np.array([(prompt * price.inp + comp * price.out) * 1e-6,
                         prompt + comp, float(self.cfg.agent_timeout_s)])

    def turn_charge(self) -> np.ndarray:
        """This group's share of one turn's worst case."""
        return self.turn_share * self.turn_worst_case()

    def _planning_reserve(self) -> np.ndarray:
        """The planning reserve r: this group's share of rho_turn, held for as
        long as the agent can still be called."""
        if self.batch is None or self.cfg.free_turns or not self.batch.agent_available():
            return zeros()
        return self.turn_charge()

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
        """Each question starts with the first stage of its composition in the
        extreme point (of the larger share if split), in arrival order."""
        chosen = alloc.chosen()
        recs = list(self.records.values())
        cheap = self._cheap_cost(recs) if self.cfg.pacing else None
        for i, rec in enumerate(recs):
            cand = chosen.get(rec.qid)
            wf: Workflow | None = None if cand is None else cand.payload
            if wf is None or wf.is_stop:
                rec.closed = True
                continue
            rec.opening = wf
            block = first_block(wf)
            if cheap is not None:
                expected = self.model.rollout(rec.meta, rec.state, block, 0.0).cost
                if not pace(self.meter, expected, len(recs) - i - 1, cheap):
                    block = self._cheapest_block(rec) or block
            report = self._execute(rec, block)
            rec.first_obs = rec.state.answer_obs or "-"
            if not report.made_progress and not rec.state.has_answer:
                rec.closed = True
        self._assign_cells()

    def _cheapest_block(self, rec: QuestionRecord) -> Workflow | None:
        actions = [a for a in self.grammar.actions(rec.state) if a.llm_calls]
        if not actions:
            return None
        a = min(actions, key=lambda s: (s.samples, {"S": 0, "M": 1, "L": 2}.get(s.tier, 3)))
        return first_block(branch(a, {o: None for o in a.observations()}))

    def _cheap_cost(self, recs: list[QuestionRecord]) -> np.ndarray:
        """The mean predicted cost of the cheapest first call."""
        blocks = [self._cheapest_block(r) for r in recs]
        items = [(r.meta, r.state, b) for r, b in zip(recs, blocks) if b is not None]
        if not items:
            return zeros()
        rolls = self.model.rollout_many(items, [0.0] * len(items))
        return np.mean([r.cost for r in rolls], axis=0)

    def _run_whole_workflows(self, alloc: Allocation) -> None:
        """Give every question its complete workflow from predictions made
        before anything runs."""
        chosen = alloc.chosen()
        for rec in self.records.values():
            cand = chosen.get(rec.qid)
            wf: Workflow | None = None if cand is None else cand.payload
            if wf is None or wf.is_stop:
                rec.closed = True
                continue
            self._execute(rec, wf)
            rec.closed = True

    def _recalibrate(self) -> None:
        """Within-batch recalibration (Sec. VI): each database's share of
        correct answers is read off its observed execution rates, and the held
        answers are rescored."""
        by_db: dict[str, list[QuestionRecord]] = {}
        for r in self.records.values():
            if r.state.history:
                by_db.setdefault(r.db, []).append(r)
        for db, recs in by_db.items():
            self.model.recalibrate(db, [(r.meta, r.state) for r in recs])
            helds = self.model.s_of_many([(r.meta, r.state) for r in recs])
            for r, s in zip(recs, helds):
                r.s_held = float(s)

    def _scoring_state(self, rec: QuestionRecord) -> tuple[State, float]:
        """The state a candidate is scored in.  By default this is the state the
        question is really in.  Scoring before the first stage (an ablation)
        replaces what each stage of the opening block returned by the
        observation the model expected, with imputed observables, keeps the
        rest of the history and its Select decisions, and rescores the held
        answer from that state."""
        if not self.cfg.score_from_empty or not rec.state.history:
            return rec.state, rec.s_held
        blind, s = State(), 0.0
        n_open = self._opening_length(rec)
        for i, (stage, obs) in enumerate(rec.state.history):
            if i < n_open:
                res = self.model.predict(rec.meta, blind, stage).resolve(s)
                j = int(np.argmax(res.p_obs))
                blind = self.model.child(blind, stage, res.labels[j], True, meta=rec.meta)
                s = float(res.s_after[j])
            else:
                blind = blind.append(stage, obs, keep=rec.state.kept(i),
                                     observables=rec.state.trail[i])
        return blind, float(self.model.s_of_state(rec.meta, blind))

    @staticmethod
    def _opening_length(rec: QuestionRecord) -> int:
        """How many stages of a question's history its opening block ran."""
        n = 0
        node = first_block(rec.opening) if rec.opening is not None else STOP
        for stage, obs in rec.state.history:
            if node is None or node.is_stop or node.stage.id != stage.id:
                break
            n += 1
            node = node.children.get(obs)
        return n

    def _candidates(self, rec: QuestionRecord) -> list[Workflow]:
        """W_i: what a question may run next.  The rest of the composition its
        first stage came from, the candidates the agent registered for its cell,
        and, once its cell has been flagged, the one-stage edits of those
        candidates (Sec. IV-C).  Stop is always added by the caller."""
        base: dict[str, Workflow] = {}
        rest = remainder(rec.opening, rec.state)
        if not rest.is_stop and self.grammar.is_wellformed(rest, rec.state):
            base[shape(rest)] = rest
        for wf in rec.registered.values():
            if self.grammar.is_wellformed(wf, rec.state):
                base[shape(wf)] = wf
        out = dict(base)
        if rec.edited:
            for wf in list(base.values()) + [rec.planned or STOP]:
                for edit in self.grammar.one_stage_edits(rec.state, wf):
                    out[shape(edit)] = edit
            # a verification only pays together with what it would trigger
            for a in self.grammar.actions(rec.state):
                if a.kind == "verify":
                    for tier in ("M", "L"):
                        chain = parse_chain(f"ver:{a.tier},esc:{tier}")
                        if self.grammar.is_wellformed(chain, rec.state):
                            out[shape(chain)] = chain
        keep = [wf for wf in out.values()
                if not wf.is_stop and shape(wf) not in rec.blocked]
        return keep[: self.cfg.max_candidates]

    def add_edits(self, cells: list[str]) -> None:
        """After a flag: the candidates of the questions now in the flagged
        cells are widened by their one-stage edits at the next solve."""
        flagged = set(cells)
        for r in self.records.values():
            if not r.closed and r.cell in flagged:
                r.edited = True

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
        if self.cfg.cost_reserve:
            # Eq. 8 over the block planned before the next re-solve
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
        for cand in order:
            rec = self.records.get(cand.qid)
            if rec is None or rec.closed:
                continue
            wf: Workflow | None = cand.payload
            if wf is None or wf.is_stop:
                rec.closed = True
                continue
            report = self._execute(rec, wf)
            if report.stages:
                rec.selected_value = float(cand.value)
            if not report.made_progress:
                # the meter would not pay for this one.  A refused continuation
                # waits when releasing the reservations in flight would make it
                # fit; run one at a time, nothing is in flight, so it is
                # removed from the candidates and the next solve chooses among
                # what is left
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
                                self.model, select=self.cfg.use_select)

    # ------------------------------------------------------------------ #
    # digest cells
    # ------------------------------------------------------------------ #
    def _assign_cells(self) -> None:
        """Cells by database, first-stage observation and deciles of predicted
        success and predicted cost."""
        open_recs = [r for r in self.records.values() if not r.closed]
        if not open_recs:
            return
        s = np.array([r.s_held for r in open_recs])
        c = np.array([float(r.predicted[0]) for r in open_recs])
        deciles = np.linspace(0.1, 0.9, 9)
        s_edges = (np.unique(np.quantile(s, deciles)) if len(s) > len(deciles)
                   else np.array([0.33, 0.66]))
        c_edges = (np.unique(np.quantile(c, deciles)) if len(c) > len(deciles)
                   else np.array([0.0]))
        for r in open_recs:
            si = int(np.searchsorted(s_edges, r.s_held))
            ci = int(np.searchsorted(c_edges, float(r.predicted[0])))
            r.cell = f"{r.db}|{_obs_bucket(r.first_obs)}|s{si}c{ci}"

    def _open_by_cell(self) -> dict[str, list[QuestionRecord]]:
        groups: dict[str, list[QuestionRecord]] = {}
        for r in self.records.values():
            if not r.closed:
                groups.setdefault(r.cell, []).append(r)
        return groups

    def _cell_drift(self, recs: list[QuestionRecord]) -> np.ndarray:
        return np.mean([r.drift() for r in recs], axis=0) if recs else zeros()

    def _cell_shift(self, recs: list[QuestionRecord]) -> float:
        """The largest total-variation distance, over the stages the cell's
        questions ran, between the observation shares seen and predicted."""
        by_stage: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
        for r in recs:
            for sid, p, j in r.obs_log:
                pred, seen, n = by_stage.get(sid, (np.zeros(len(p)), np.zeros(len(p)), 0))
                if len(pred) != len(p):
                    continue
                seen = seen.copy()
                seen[j] += 1.0
                by_stage[sid] = (pred + p, seen, n + 1)
        worst = 0.0
        for pred, seen, n in by_stage.values():
            if n:
                worst = max(worst, 0.5 * float(np.abs(seen / n - pred / n).sum()))
        return worst

    def _cell_flagged(self, recs: list[QuestionRecord]) -> bool:
        thr = self.cfg.drift_threshold
        drift = self._cell_drift(recs)
        return bool(float(np.max(np.abs(drift))) > thr or self._cell_shift(recs) > thr)

    def digest_cells(self) -> list[DigestCell]:
        """This group's cells, before the batch merges them to fit G_max."""
        cells: list[DigestCell] = []
        for cid, recs in self._open_by_cell().items():
            s = np.array([r.s_held for r in recs])
            cell = DigestCell(
                cid=cid, db=recs[0].db, obs=_obs_bucket(recs[0].first_obs),
                qids=[r.qid for r in recs], s_mean=float(s.mean()),
                s_spread=float(s.std()),
                cost_mean=np.mean([r.predicted for r in recs], axis=0),
                drift=self._cell_drift(recs),
                flagged=self._cell_flagged(recs),
                members=[cid],
            )
            cell.candidates = self._cell_summary(recs, cid)
            cell.uncertain = any(c.uncertain for c in cell.candidates)
            cells.append(cell)
        return cells

    def digest(self) -> Digest:
        """The digest of this group alone; a batch reads the digest of every
        group (``BatchPlanner.digest``)."""
        if self.batch is not None:
            return self.batch.digest()
        duals = self._last_alloc.duals if self._last_alloc is not None else zeros()
        stats = self.meter.stats
        return Digest(self.digest_cells(), self.meter.remaining, stats.spend,
                      stats.planning_spend, duals, self.meter.budget,
                      n_open=sum(1 for r in self.records.values() if not r.closed)
                      ).fit(self.cfg.digest_tokens, self.cfg.max_digest_cells)

    def _cell_summary(self, recs: list[QuestionRecord], cid: str) -> list[CellCandidate]:
        sample = recs[: min(8, len(recs))]
        wfs: dict[str, Workflow] = {}
        for wf in self._candidates(sample[0]):
            wfs[shape(wf)] = wf
        for r in recs:
            for wid, wf in r.registered.items():
                wfs[wid] = wf
        out: list[CellCandidate] = []
        for wid, wf in list(wfs.items())[:6]:
            pairs = [(r, (r.meta, r.state, wf)) for r in sample
                     if self.grammar.is_wellformed(wf, r.state)]
            if not pairs:
                continue
            rolls = self.model.rollout_many([it for _, it in pairs],
                                            [r.s_held for r, _ in pairs])
            out.append(CellCandidate(
                compact(wf), float(np.mean([r.p for r in rolls])),
                np.mean([r.cost for r in rolls], axis=0),
                float(np.std([r.p for r in rolls])),
                any(r.uncertain for r in rolls)))
        return sorted(out, key=lambda c: -c.p)

    def flagged_cells(self) -> list[str]:
        """Which cells have drifted far enough from their prediction, in cost
        or in observation shares, to be worth a look.  This reads the records
        only, so it costs nothing."""
        return [cid for cid, recs in self._open_by_cell().items()
                if self._cell_flagged(recs)]

    # ------------------------------------------------------------------ #
    # the certificate
    # ------------------------------------------------------------------ #
    def certificate(self, need: float | None = None) -> Certificate:
        """Eq. 7 for this group at the duals of its last solve, with the price
        of its share of one turn.  ``need`` stops the search once Psi exceeds
        it, since the test only compares."""
        alloc = self._last_alloc
        duals = alloc.duals if alloc is not None else zeros()
        omega = {}
        if alloc is not None:
            omega = {q: float(w) for q, w in zip(alloc.qids, alloc.omega)}
        open_recs = [r for r in self.records.values() if not r.closed]
        return certify(self.model, self.values, self.grammar,
                       [(r.meta, r.state, r.s_held) for r in open_recs],
                       duals, omega, self.turn_charge(),
                       depth=self.cfg.certificate_depth, need=need)

    # ---- used by the tools ----
    def owns(self, cell: str) -> bool:
        return any(r.cell == cell for r in self.records.values())

    def register(self, cell: str, wf: Workflow) -> bool:
        if not isinstance(wf, Workflow) or wf.is_stop:
            return False
        recs = [r for r in self.records.values() if r.cell == cell and not r.closed]
        if not recs:
            return False
        ok = False
        for r in recs:
            if self.grammar.is_wellformed(wf, r.state):
                r.registered[shape(wf)] = wf
                ok = True
        return ok

    def probe(self, cell: str, stage, n: int) -> dict:
        """Run a stage on a few questions of a cell, to observe a cell whose
        predictions spread widely or rest on few training questions."""
        recs = [r for r in self.records.values() if r.cell == cell and not r.closed]
        picked = recs[: max(0, int(n))]
        before = self.meter.stats.spend.copy()
        seen: dict[str, int] = {}
        wf = branch(stage, {o: None for o in stage.observations()})
        probed = 0
        for r in picked:
            if not self.grammar.is_wellformed(wf, r.state):
                continue
            self._execute(r, wf)
            probed += 1
            seen[r.state.answer_obs or "-"] = seen.get(r.state.answer_obs or "-", 0) + 1
        return {"probed": probed, "observations": seen,
                "cost": (self.meter.stats.spend - before).tolist()}


def _obs_bucket(obs: str) -> str:
    if obs in ("-", None):
        return "-"
    head = obs.split(".")[0]
    tail = obs.split(".")[-1]
    return head if tail in (head, "") else f"{head}.{tail}"


