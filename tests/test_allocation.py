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


def test_reserve_z_is_proportional_to_the_predicted_deviation():
    import numpy as np
    from budgetsql.config import zeros
    from budgetsql.planner.allocation import Candidate, solve
    from budgetsql.planner.reserve import reserve_for
    cost = np.array([1.0, 10.0, 1.0])
    cands = [Candidate("q", "w", 0.8, cost, cost ** 2, cost * 2),
             Candidate("q", "stop", 0.2, zeros(), zeros(), zeros(), is_stop=True)]
    alloc = solve(cands, np.array([4.0, 40.0, 4.0]))
    one = reserve_for(alloc, 1, np.ones(3), z=1.0)
    two = reserve_for(alloc, 1, np.ones(3), z=2.0)
    assert np.allclose(two, 2.0 * one)
    assert np.allclose(reserve_for(alloc, 1, np.zeros(3), z=1.0), 0.0)
