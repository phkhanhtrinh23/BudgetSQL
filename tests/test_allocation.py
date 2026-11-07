import numpy as np

from budgetsql.config import DIMS, vec
from budgetsql.planner.allocation import Candidate, greedy, solve
from budgetsql.planner.reserve import bernstein_reserve, run_order, theta


def _candidates():
    out = []
    for q in ("q1", "q2"):
        out.append(Candidate(q, "stop", 0.50, vec(), is_stop=True))
        out.append(Candidate(q, "cheap", 0.55, vec(1e-4, 100, 1.0)))
        out.append(Candidate(q, "rich", 0.70, vec(4e-4, 400, 4.0)))
    return out


def test_duals_satisfy_complementary_slackness():
    alloc = solve(_candidates(), vec(5e-4, 1e5, 1e3))
    assert alloc.price_of("usd") > 0
    assert alloc.binding() == ["usd"]
    for c, x in zip(alloc.candidates, alloc.x):
        if x > 1e-7:
            assert np.isclose(c.value - alloc.duals @ c.cost,
                              alloc.omega[alloc.qids.index(c.qid)], atol=1e-6)


def test_extreme_point_splits_few_questions():
    alloc = solve(_candidates(), vec(5e-4, 1e5, 1e3))
    assert len(alloc.split_questions()) <= len(DIMS)


def test_the_program_is_at_least_as_good_as_greedy():
    cands = _candidates()
    budget = vec(5e-4, 1e5, 1e3)
    assert solve(cands, budget).value >= greedy(cands, budget).value - 1e-9


def test_budget_is_respected_and_reserve_is_held_where_priced():
    alloc = solve(_candidates(), vec(5e-4, 1e5, 1e3))
    assert np.all(alloc.spend <= vec(5e-4, 1e5, 1e3) + 1e-9)
    assert theta(alloc) >= 1.0
    rho = bernstein_reserve(np.array([1e-8, 1.0, 1e-2]), np.array([1e-3, 500, 5]),
                            np.log(2.0))
    assert np.all(rho >= 0)
    assert len(run_order(alloc)) <= len(alloc.candidates)
