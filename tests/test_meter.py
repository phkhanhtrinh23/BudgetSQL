import numpy as np
import pytest

from budgetsql.config import vec
from budgetsql.grammar import gen
from budgetsql.planner.meter import BudgetExceeded, Meter


def test_reservation_blocks_an_overrun():
    m = Meter(vec(1e-3, 1000, 10))
    r = m.reserve(vec(9e-4, 900, 9))
    assert r is not None
    assert m.reserve(vec(9e-4, 900, 9)) is None
    m.settle(r, vec(1e-4, 100, 1))
    assert m.reserve(vec(5e-4, 500, 5)) is not None


def test_spend_never_exceeds_the_budget():
    m = Meter(vec(1e-3, 1000, 10))
    for _ in range(50):
        m.charge(vec(2e-4, 200, 2), vec(2e-4, 200, 2))
    assert np.all(m.spend <= m.budget + 1e-12)
    assert m.stats.refused > 0


def test_a_call_is_charged_at_most_its_reservation():
    m = Meter(vec(1e-3, 1000, 10))
    r = m.reserve(vec(5e-4, 500, 5))
    charged = m.settle(r, vec(9e-4, 900, 9), cancelled=True)
    assert np.all(charged <= vec(5e-4, 500, 5) + 1e-12)
    with pytest.raises(BudgetExceeded):
        m.settle(r, vec(0, 0, 0))


def test_completion_limit_follows_the_remaining_budget():
    m = Meter(vec(1e-5, 400, 10))
    assert m.affordable_completion(gen("L"), prompt_tokens=100) < 600


def test_planning_cap_is_enforced():
    m = Meter(vec(1.0, 1e6, 1e4), planning_cap=0.01)
    assert m.charge(vec(0.005, 10, 1), planning=True) is not None
    assert m.charge(vec(0.02, 10, 1), planning=True) is None
