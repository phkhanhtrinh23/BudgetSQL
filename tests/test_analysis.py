import numpy as np

from budgetsql.analysis import (frontier_bound, gain_over_random, identity_terms,
                                logistic_difficulty_bound, stats, verify_identities)


def test_exact_identities_hold():
    res = verify_identities()
    assert res["escalation_identity"] and res["ceilings"]
    assert res["regression_only_example"] and res["detector_operating_point"]
    assert res["information_term"]
    assert res["common_mode"]["within"] and res["common_mode"]["sign_violations"] == 0


def test_identity_matches_the_covariance_form():
    rng = np.random.default_rng(0)
    keep = rng.integers(0, 2, 400)
    esc = rng.integers(0, 2, 400)
    rule = rng.random(400)
    terms = identity_terms(rule, keep, esc)
    assert np.isclose(terms["gain"], gain_over_random(rule, keep, esc), atol=1e-9)


def test_a_rule_cannot_beat_its_ceiling():
    rng = np.random.default_rng(1)
    keep = rng.integers(0, 2, 500)
    esc = rng.integers(0, 2, 500)
    st = stats(keep, esc)
    oracle = ((keep == 0) & (esc == 1)).astype(float)
    gain = gain_over_random(oracle, keep, esc)
    share = float(oracle.mean())
    over_matched = gain + share * st.m - share * max(st.m, 0.0)
    assert over_matched <= st.ceiling_matched + 1e-12


def test_shared_difficulty_leaves_no_room():
    b = logistic_difficulty_bound(0.5, 0.5)
    assert b["ell"] == 0.0
    assert frontier_bound(0.0, b["ell"]) == 0.0
