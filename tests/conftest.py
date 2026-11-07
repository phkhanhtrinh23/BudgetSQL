import pytest

from budgetsql.config import PlannerConfig
from budgetsql.grammar import Grammar, reference_library
from budgetsql.trace import SimulatedTrace, TraceCorpus


@pytest.fixture(scope="session")
def corpus():
    src = SimulatedTrace(n_questions=90, n_dbs=3, seed=7)
    return TraceCorpus(src, Grammar(), max_states=12, seed=7)


@pytest.fixture(scope="session")
def cfg():
    return PlannerConfig(n_estimators=20, num_leaves=6, turns_per_session=2,
                         replan_block=10)


@pytest.fixture(scope="session")
def library():
    return reference_library()


@pytest.fixture(scope="session")
def values(corpus, cfg, library):
    from budgetsql.worldmodel import CrossFitStacker
    stacker = CrossFitStacker(cfg, inner_folds=2, workflows=library,
                              grammar=corpus.grammar)
    return stacker.fit(corpus, corpus.dbs[-1])
