import numpy as np
import pytest

from budgetsql.baselines import BASELINES, Oracle
from budgetsql.config import budget_for_group
from budgetsql.planner import Meter


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_every_baseline_runs_within_budget(corpus, cfg, library, name):
    metas = corpus.questions_of(corpus.dbs[-1])
    train = [d for d in corpus.dbs if d != corpus.dbs[-1]]
    method = BASELINES[name](name=name, cfg=cfg, library=library)
    method.prepare(corpus, train)
    b = budget_for_group("V4", len(metas))
    res = method.run(metas, b, Meter(b))
    assert np.all(res.spend <= b + 1e-9)
    assert len(res.correct) == len(metas)


def test_oracle_is_an_upper_reference(corpus, cfg, library):
    metas = corpus.questions_of(corpus.dbs[-1])
    oracle = Oracle(name="oracle", cfg=cfg, library=library)
    oracle.prepare(corpus, corpus.dbs)
    b = budget_for_group("V1", len(metas))
    res = oracle.run(metas, b, Meter(b))
    best = BASELINES["best_fixed_mixture"](name="m", cfg=cfg, library=library)
    best.prepare(corpus, [d for d in corpus.dbs if d != corpus.dbs[-1]])
    other = best.run(metas, b, Meter(b))
    assert res.accuracy >= other.accuracy - 1e-9


def test_best_fixed_workflow_picks_one_affordable_workflow(corpus, cfg, library):
    from budgetsql.baselines.fixed import ONE_SHOT, BestFixedWorkflow
    metas = corpus.questions_of(corpus.dbs[-1])
    method = BestFixedWorkflow(name="w", cfg=cfg, library=library)
    method.prepare(corpus, [d for d in corpus.dbs if d != corpus.dbs[-1]])
    b = budget_for_group("V4", len(metas))
    method.run(metas, b, Meter(b))
    assert method.choice in ONE_SHOT
    per_q = b / len(metas)
    assert np.all(method.profiles[method.choice].cost <= per_q + 1e-12)
