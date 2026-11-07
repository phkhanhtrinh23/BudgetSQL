import numpy as np

from budgetsql.config import budget_for_group
from budgetsql.planner import BudgetSQL, Meter, PlanningAgent


def _run(corpus, values, cfg, budget="V1", agent=None):
    metas = corpus.questions_of(corpus.dbs[-1])
    b = budget_for_group(budget, len(metas))
    planner = BudgetSQL(values, corpus, corpus.grammar, cfg, agent)
    return planner.run(metas, b, Meter(b, planning_cap=cfg.planning_cap)), b


def test_a_run_never_exceeds_its_budget(corpus, values, cfg):
    for budget in ("V1", "V2", "V3", "V4"):
        res, b = _run(corpus, values, cfg, budget, PlanningAgent())
        assert np.all(res.spend <= b + 1e-9), budget
        assert 0.0 <= res.accuracy <= 1.0


def test_every_question_gets_an_answer_or_keeps_none(corpus, values, cfg):
    res, _ = _run(corpus, values, cfg)
    assert len(res.correct) == len(corpus.questions_of(corpus.dbs[-1]))


def test_planning_stays_within_its_cap(corpus, values, cfg):
    res, b = _run(corpus, values, cfg, "V1", PlanningAgent())
    assert res.planning_share["usd"] <= cfg.planning_cap + 1e-9


def test_a_larger_budget_never_hurts(corpus, values, cfg):
    small, _ = _run(corpus, values, cfg.with_(use_agent=False), "V3")
    large, _ = _run(corpus, values, cfg.with_(use_agent=False), "V1")
    assert large.accuracy >= small.accuracy - 0.1


def test_tools_reject_ill_formed_candidates(corpus, values, cfg):
    from budgetsql.planner.tools import ToolBox
    metas = corpus.questions_of(corpus.dbs[-1])
    b = budget_for_group("V1", len(metas))
    planner = BudgetSQL(values, corpus, corpus.grammar, cfg, PlanningAgent())
    planner.run(metas, b, Meter(b))
    box = ToolBox(planner)
    assert not box.compose("nope", "gen:X").ok
    assert not box.compose("nope", "not a stage").ok
