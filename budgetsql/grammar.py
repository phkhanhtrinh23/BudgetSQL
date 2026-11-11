"""The stage grammar, question states and workflows composed from it.

A workflow is a decision tree over stages whose branches are the observations of
``execute`` and ``verify``.  The grammar, not a fixed list, defines what a
deployment can run; a workflow calls an LLM at most ``MAX_LLM_CALLS`` times on
any branch.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping

from .config import TIERS

MAX_LLM_CALLS = 4

#: Context levels: the full schema, or a schema pruned by retrieval at two levels.
CTX_LEVELS: tuple[str, ...] = ("full", "prune1", "prune2")

#: Observation alphabets. ``execute`` reports the execution status with a
#: row-count bucket and, for several samples, the agreement among them.
EXEC_STATUS: tuple[str, ...] = ("valid.one", "valid.many", "empty", "error")
AGREEMENT: tuple[str, ...] = ("agree", "split")
VERIFY_OBS: tuple[str, ...] = ("accept", "doubt")
#: A replacing stage observes the new answer's execution and its agreement with
#: the held answer.
REPLACE_OBS: tuple[str, ...] = ("valid.match", "valid.diff", "empty", "error")

OBSERVE_ONLY = ("execute", "verify")
REPLACING = ("repair", "escalate")
ANSWER_PRODUCING = ("generate",) + REPLACING


@dataclass(frozen=True, order=True)
class Stage:
    kind: str
    tier: str | None = None
    samples: int = 1
    ctx: str = "full"
    cap: int = 600

    @property
    def id(self) -> str:
        if self.kind == "context":
            return f"ctx:{self.ctx}"
        if self.kind == "execute":
            return f"exec:{self.samples}"
        s = f"{self.kind}:{self.tier}"
        if self.samples > 1:
            s += f"x{self.samples}"
        if self.kind == "generate" and self.ctx != "full":
            s += f"@{self.ctx}"
        return s

    @property
    def llm_calls(self) -> int:
        return 0 if self.kind in ("context", "execute") else self.samples

    def observations(self) -> tuple[str, ...]:
        if self.kind == "context":
            return ("ok",)
        if self.kind == "execute":
            if self.samples > 1:
                return tuple(f"{s}.{a}" for s in EXEC_STATUS for a in AGREEMENT)
            return EXEC_STATUS
        if self.kind == "verify":
            return VERIFY_OBS
        if self.kind in REPLACING:
            return REPLACE_OBS
        return ("ok",)

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.id


# --------------------------------------------------------------------------- #
# states
# --------------------------------------------------------------------------- #
#: Marks, in a branch key, a replacing stage whose new answer Select rejected.
REJECTED = "~rejected"


@dataclass(frozen=True)
class Observables:
    """The numeric part of what the executed stages revealed about the held
    answer: how many distinct results its samples gave, how long its producing
    call's completion was, and how many earlier answers agree with it (with
    ``n_earlier`` the answers produced besides it).  In an imagined rollout
    they are imputed by their training mean given the stage and observation."""

    n_distinct: float = 1.0
    completion_tokens: float = 0.0
    earlier_equal: float = 0.0

    def after(self, stage: "Stage", kept: bool, values) -> "Observables":
        """The observables once ``stage`` ran and returned ``values`` (any
        object with ``n_distinct``, ``completion_tokens`` and
        ``equal_to_held``), with Select's decision ``kept``."""
        if values is None:
            return self
        nd = float(getattr(values, "n_distinct", 1.0))
        comp = float(getattr(values, "completion_tokens", 0.0))
        eq = float(getattr(values, "equal_to_held", 0.0))
        if stage.kind == "generate":
            return Observables(nd, comp, 0.0)
        if stage.kind == "execute":
            return Observables(nd, self.completion_tokens, self.earlier_equal)
        if stage.kind in REPLACING:
            if kept:
                # the answer it replaced, and those that agreed with it, now
                # agree with the new one exactly when the two are equal
                return Observables(nd, comp, (self.earlier_equal + 1.0) * eq)
            # a rejected answer is one more answer the held one is compared to
            return Observables(self.n_distinct, self.completion_tokens,
                               self.earlier_equal + eq)
        return self


@dataclass(frozen=True)
class State:
    """What the planner knows about one question: the stages run and what they
    returned.  ``phi`` in :mod:`budgetsql.features` summarises it for the
    world model's heads.

    The held answer is the one Select kept: a replacing stage whose new answer
    is rejected leaves the earlier answer, its producing stage and its
    observation held.  Which replacing stages were rejected is part of the
    branch, because every later stage --- a verification, a comparison of a
    new answer with the held one --- reads the answer held.
    """

    history: tuple[tuple[Stage, str], ...] = ()
    ctx: str = "full"
    answer_stage: Stage | None = None
    answer_obs: str | None = None
    n_answers: int = 0
    #: history indices of replacing stages whose answer Select rejected
    rejected: tuple[int, ...] = ()
    #: history index of the stage that produced the held answer
    answer_index: int = -1
    #: the observables after each stage, so a prefix can be replayed exactly
    trail: tuple[Observables, ...] = ()

    @property
    def observables(self) -> Observables:
        return self.trail[-1] if self.trail else Observables()

    def append(self, stage: Stage, obs: str, keep: bool = True, values=None,
               observables: Observables | None = None) -> "State":
        """The state after ``stage`` observed ``obs``.  For a replacing stage,
        ``keep`` is Select's decision on its new answer.  ``values`` is what the
        stage returned (a trace outcome, or the imputed means of a rollout);
        ``observables`` sets the result directly, as a replay does."""
        i = len(self.history)
        hist = self.history + ((stage, obs),)
        ctx = stage.ctx if stage.kind == "context" else self.ctx
        kept = keep or self.answer_stage is None or stage.kind not in REPLACING
        ob = observables if observables is not None else \
            self.observables.after(stage, kept, values)
        trail = self.trail + (ob,)
        if stage.kind in ANSWER_PRODUCING:
            if kept:
                return State(hist, ctx, stage, obs, self.n_answers + 1,
                             self.rejected, i, trail)
            return State(hist, ctx, self.answer_stage, self.answer_obs,
                         self.n_answers + 1, self.rejected + (i,), self.answer_index,
                         trail)
        if stage.kind == "execute" and self.answer_stage is not None:
            # execution is what reveals the held answer's status
            return State(hist, ctx, self.answer_stage, obs, self.n_answers,
                         self.rejected, self.answer_index, trail)
        return State(hist, ctx, self.answer_stage, self.answer_obs, self.n_answers,
                     self.rejected, self.answer_index, trail)

    def kept(self, i: int) -> bool:
        """Whether the answer of the ``i``-th stage was kept by Select."""
        return i not in self.rejected

    def replay(self, upto: int | None = None) -> "State":
        """The state after the first ``upto`` stages, with the same decisions
        and the same observables."""
        out = State()
        for i, (stage, obs) in enumerate(self.history[:upto]):
            out = out.append(stage, obs, keep=self.kept(i),
                             observables=self.trail[i] if i < len(self.trail) else None)
        return out

    def before_last(self) -> "State":
        """The state as it stood before the last stage ran."""
        return self.replay(len(self.history) - 1)

    # ---- bookkeeping the grammar needs ----
    @property
    def llm_calls(self) -> int:
        return sum(s.llm_calls for s, _ in self.history)

    @property
    def tiers_used(self) -> tuple[str, ...]:
        return tuple(sorted({s.tier for s, _ in self.history if s.tier}))

    @property
    def has_answer(self) -> bool:
        return self.answer_stage is not None

    @property
    def executed(self) -> bool:
        """Whether the held answer has been executed."""
        return self.answer_obs is not None and (
            self.answer_stage is not None
            and (self.answer_stage.kind in REPLACING or any(
                s.kind == "execute" for s, _ in self.history))
        )

    @property
    def last_status(self) -> str | None:
        if self.answer_obs is None:
            return None
        return self.answer_obs.split(".")[0] if self.answer_obs != "ok" else None

    @property
    def key(self) -> tuple[tuple[str, str], ...]:
        """Branch key into the trace corpus; a rejected answer is marked."""
        return tuple((s.id, o if i not in self.rejected else o + REJECTED)
                     for i, (s, o) in enumerate(self.history))

    def __len__(self) -> int:
        return len(self.history)


# --------------------------------------------------------------------------- #
# workflows
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Workflow:
    """A decision tree over stages.  ``children`` maps an observation of
    ``stage`` to the sub-workflow that follows it; a missing or ``None`` child
    stops and keeps the answer held at that point."""

    stage: Stage | None = None
    children: Mapping[str, "Workflow | None"] = field(default_factory=dict)
    name: str = ""

    @property
    def is_stop(self) -> bool:
        return self.stage is None

    def child(self, obs: str) -> "Workflow":
        nxt = self.children.get(obs)
        return nxt if nxt is not None else STOP

    def stages(self) -> Iterable[Stage]:
        """Every stage of the tree once, in the order it can first run.  A
        sub-workflow shared by several observations is listed once."""
        seen: set[int] = set()
        stack = [self]
        while stack:
            node = stack.pop(0)
            if node is None or node.stage is None or id(node) in seen:
                continue
            seen.add(id(node))
            yield node.stage
            stack.extend(node.children.values())

    def max_llm_calls(self) -> int:
        if self.stage is None:
            return 0
        below = max((c.max_llm_calls() for c in self.children.values() if c), default=0)
        return self.stage.llm_calls + below

    def depth(self) -> int:
        if self.stage is None:
            return 0
        return 1 + max((c.depth() for c in self.children.values() if c), default=0)

    def rename(self, name: str) -> "Workflow":
        return replace(self, name=name)

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.name or (self.stage.id if self.stage else "stop")


STOP = Workflow(name="stop")


def seq(*stages: Stage, name: str = "") -> Workflow:
    """A workflow that runs the stages in order, on every observation."""
    node: Workflow | None = None
    for st in reversed(stages):
        node = Workflow(st, {o: node for o in st.observations()})
    return (node or STOP).rename(name)


def branch(stage: Stage, mapping: Mapping[str, Workflow | None], name: str = "") -> Workflow:
    children = {o: mapping.get(o) for o in stage.observations()}
    return Workflow(stage, children, name)


def gen(tier: str, samples: int = 1, ctx: str = "full", cap: int = 600) -> Stage:
    return Stage("generate", tier, samples, ctx, cap)


def execute(samples: int = 1) -> Stage:
    return Stage("execute", None, samples)


def repair(tier: str, cap: int = 600) -> Stage:
    return Stage("repair", tier, 1, "full", cap)


def verify(tier: str, cap: int = 64) -> Stage:
    return Stage("verify", tier, 1, "full", cap)


def escalate(tier: str, cap: int = 600) -> Stage:
    return Stage("escalate", tier, 1, "full", cap)


def context(level: str) -> Stage:
    return Stage("context", None, 1, level)


# --------------------------------------------------------------------------- #
# the grammar
# --------------------------------------------------------------------------- #
class Grammar:
    """Which stages may follow a state, and which workflows that admits."""

    def __init__(self, tiers: tuple[str, ...] = TIERS,
                 max_llm_calls: int = MAX_LLM_CALLS,
                 sample_counts: tuple[int, ...] = (1, 3),
                 ctx_levels: tuple[str, ...] = CTX_LEVELS,
                 cap: int = 600):
        self.tiers = tiers
        self.max_llm_calls = max_llm_calls
        self.sample_counts = sample_counts
        self.ctx_levels = ctx_levels
        self.cap = cap
        self._rank = {t: i for i, t in enumerate(tiers)}

    # ---- legal next stages ----
    def actions(self, state: State) -> list[Stage]:
        out: list[Stage] = []
        left = self.max_llm_calls - state.llm_calls
        if not state.has_answer:
            if not state.history:
                for lv in self.ctx_levels[1:]:
                    out.append(context(lv))
            for t in self.tiers:
                for k in self.sample_counts:
                    if k <= left:
                        out.append(gen(t, k, state.ctx, self.cap))
            return out
        if not state.executed:
            return [execute(state.answer_stage.samples)]
        if left <= 0:
            return []
        status = state.last_status
        if status in ("empty", "error"):
            for t in self.tiers:
                out.append(repair(t, self.cap))
        held = state.answer_stage.tier
        for t in self.tiers:
            if held is None or self._rank[t] > self._rank[held]:
                out.append(escalate(t, self.cap))
        if left >= 2 and not any(s.kind == "verify" for s, _ in state.history):
            for t in self.tiers:
                out.append(verify(t))
        return out

    def is_wellformed(self, wf: Workflow, state: State | None = None) -> bool:
        state = state or State()
        if wf.is_stop:
            return True
        legal = {s.id for s in self.actions(state)}
        if wf.stage.id not in legal:
            return False
        for obs in wf.stage.observations():
            sub = wf.children.get(obs)
            if sub is None:
                continue
            if not self.is_wellformed(sub, state.append(wf.stage, obs)):
                return False
        return wf.max_llm_calls() <= self.max_llm_calls

    def one_stage_edits(self, state: State, wf: Workflow) -> list[Workflow]:
        """One tier up or down, one stage added or removed."""
        out: list[Workflow] = []
        if not wf.is_stop:
            out.append(STOP)
            for obs in wf.stage.observations():
                sub = wf.children.get(obs)
                if sub is not None and not sub.is_stop:
                    out.append(branch(wf.stage, {o: STOP for o in wf.stage.observations()}))
                    break
            st = wf.stage
            if st.tier:
                i = self._rank[st.tier]
                for j in (i - 1, i + 1):
                    if 0 <= j < len(self.tiers):
                        edit = replace(st, tier=self.tiers[j])
                        cand = Workflow(edit, {o: wf.children.get(o) for o in edit.observations()})
                        if self.is_wellformed(cand, state):
                            out.append(cand)
        for a in self.actions(state):
            cand = branch(a, {o: wf if not wf.is_stop else None for o in a.observations()})
            if self.is_wellformed(cand, state):
                out.append(cand)
        seen, uniq = set(), []
        for c in out:
            k = shape(c)
            if k not in seen:
                seen.add(k)
                uniq.append(c)
        return uniq


def token(stage: Stage) -> str:
    """A stage in the agent's compact syntax (``parse_stage`` reads it back)."""
    if stage.kind == "context":
        return f"ctx:{stage.ctx}"
    if stage.kind == "execute":
        return "exec" if stage.samples == 1 else f"exec:{stage.samples}"
    short = {"generate": "gen", "repair": "rep", "verify": "ver",
             "escalate": "esc"}[stage.kind]
    t = f"{short}:{stage.tier}" + (f"x{stage.samples}" if stage.samples > 1 else "")
    if stage.kind == "generate" and stage.ctx != "full":
        t += f"@{stage.ctx}"
    return t


def compact(wf: Workflow) -> str:
    """A short text for a workflow: a chain ``ver:M,esc:L`` when every
    observation continues the same way, otherwise the branches that continue,
    ``exec[error>rep:L;empty>rep:L]``."""
    if wf.is_stop:
        return "stop"
    subs = {o: wf.child(o) for o in wf.stage.observations()}
    keys = {shape(w) for w in subs.values()}
    head = token(wf.stage)
    if len(keys) == 1:
        rest = next(iter(subs.values()))
        return head if rest.is_stop else f"{head},{compact(rest)}"
    parts = [f"{o}>{compact(w)}" for o, w in subs.items() if not w.is_stop]
    return f"{head}[{';'.join(parts)}]"


def single_call_first(wf: Workflow) -> bool:
    """Whether a composition's first call is a single call; a Context stage
    before it calls nothing."""
    node = wf
    while node is not None and not node.is_stop:
        if node.stage.llm_calls:
            return node.stage.llm_calls == 1
        node = next(iter(node.children.values()), None)
    return False


def shape(wf: Workflow) -> str:
    """A canonical string for a workflow, used to de-duplicate candidates."""
    if wf.is_stop:
        return "."
    inner = ",".join(f"{o}>{shape(wf.child(o))}" for o in wf.stage.observations())
    return f"{wf.stage.id}({inner})"


# --------------------------------------------------------------------------- #
# a library of reference workflows
# --------------------------------------------------------------------------- #
def _after(stage: Stage, sub: Workflow | None) -> Workflow:
    """Run ``stage``, then ``sub`` whatever it observes."""
    return Workflow(stage, {o: sub for o in stage.observations()})


def _with_repair(tier: str, rep: str, name: str) -> Workflow:
    """One call of ``tier``, repaired by ``rep`` on an error or an empty result.

    The repair stage observes its own answer's execution, so no separate
    execution follows it.
    """
    fix = seq(repair(rep))
    e = Workflow(execute(1), {"empty": fix, "error": fix})
    return _after(gen(tier), e).rename(name)


def _vote(tier: str, name: str) -> Workflow:
    return seq(gen(tier, 3), execute(3), name=name)


def _cascade(low: str, high: str, name: str) -> Workflow:
    """Three samples of ``low``, then ``high`` when the samples split."""
    esc = seq(escalate(high))
    e = execute(3)
    e_node = Workflow(e, {o: (esc if o.endswith("split") else None) for o in e.observations()})
    return _after(gen(low, 3), e_node).rename(name)


def reference_library() -> dict[str, Workflow]:
    """Named workflows that are useful on their own and as a starting point for
    the agent: single calls, repairs, votes, cascades, a compressed context and
    a verified escalation."""
    lib: dict[str, Workflow] = {}
    for t in TIERS:
        lib[t] = seq(gen(t), execute(1), name=t)
        lib[f"{t}+R"] = _with_repair(t, t, f"{t}+R")
    lib["S+R_L"] = _with_repair("S", "L", "S+R_L")
    lib["M+R_L"] = _with_repair("M", "L", "M+R_L")
    lib["Sx3"] = _vote("S", "Sx3")
    lib["Mx3"] = _vote("M", "Mx3")
    lib["S->M"] = _cascade("S", "M", "S->M")
    lib["S->L"] = _cascade("S", "L", "S->L")
    lib["M->L"] = _cascade("M", "L", "M->L")

    lib["M_c"] = seq(context("prune1"), gen("M", 1, "prune1"), execute(1), name="M_c")

    fix = seq(repair("L"))
    e_node = Workflow(execute(1), {"empty": fix, "error": fix})
    lib["M_c+R_L"] = Workflow(
        context("prune1"),
        {"ok": _after(gen("M", 1, "prune1"), e_node)},
        "M_c+R_L",
    )

    esc = seq(escalate("L"))
    v_node = Workflow(verify("M"), {"accept": None, "doubt": esc})
    lib["M->V->L"] = _after(gen("M"), _after(execute(1), v_node)).rename("M->V->L")
    return lib


LIBRARY_ORDER = ["S", "M", "L", "S+R", "M+R", "L+R", "S+R_L", "M+R_L",
                 "Sx3", "Mx3", "S->M", "S->L", "M->L", "M_c", "M_c+R_L", "M->V->L"]


# --------------------------------------------------------------------------- #
# a compact text form for composing workflows
# --------------------------------------------------------------------------- #
def parse_stage(token: str) -> Stage:
    """``gen:M``, ``gen:Mx3``, ``gen:M@prune1``, ``exec``, ``exec:3``,
    ``rep:L``, ``ver:M``, ``esc:L``, ``ctx:prune1``."""
    token = token.strip()
    kind, _, rest = token.partition(":")
    kind = {"gen": "generate", "exec": "execute", "rep": "repair",
            "ver": "verify", "esc": "escalate", "ctx": "context"}.get(kind, kind)
    if kind == "context":
        if rest not in CTX_LEVELS:
            raise ValueError(f"unknown context level {rest!r}")
        return context(rest)
    if kind == "execute":
        return execute(int(rest) if rest else 1)
    tier, _, ctx = rest.partition("@")
    samples = 1
    if "x" in tier:
        tier, _, k = tier.partition("x")
        samples = int(k)
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    if kind == "generate":
        return gen(tier, samples, ctx or "full")
    if kind == "repair":
        return repair(tier)
    if kind == "verify":
        return verify(tier)
    if kind == "escalate":
        return escalate(tier)
    raise ValueError(f"unknown stage {token!r}")


def parse_chain(spec: str) -> Workflow:
    """A comma-separated chain of stages, run in order whatever each observes."""
    stages = [parse_stage(t) for t in spec.split(",") if t.strip()]
    if not stages:
        return STOP
    return seq(*stages, name=spec.strip())


def _split_top(text: str, sep: str) -> list[str]:
    """Split at ``sep`` outside brackets."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def parse_workflow(text: str) -> Workflow:
    """Read the compact syntax ``compact`` writes: a chain ``ver:M,esc:L``, or
    a stage whose observations continue differently,
    ``exec[error>rep:L;empty>rep:L]`` (observations not listed stop)."""
    text = text.strip()
    if text in ("", "stop"):
        return STOP
    parts = _split_top(text, ",")
    head, rest = parts[0].strip(), ",".join(parts[1:]).strip()
    if "[" in head:
        if not head.endswith("]") or rest:
            raise ValueError(f"a branching stage ends its workflow: {text!r}")
        i = head.index("[")
        stage = parse_stage(head[:i])
        mapping: dict[str, Workflow | None] = {}
        for part in _split_top(head[i + 1:-1], ";"):
            if not part.strip():
                continue
            obs, _, sub = part.partition(">")
            obs = obs.strip()
            if obs not in stage.observations():
                raise ValueError(f"{stage.id} cannot observe {obs!r}")
            mapping[obs] = parse_workflow(sub)
        return branch(stage, mapping, name=text)
    stage = parse_stage(head)
    sub = parse_workflow(rest) if rest else None
    return Workflow(stage, {o: sub for o in stage.observations()}, text)
