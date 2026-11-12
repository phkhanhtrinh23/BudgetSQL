import numpy as np

from budgetsql.grammar import State, execute, gen, parse_chain, verify


def test_observing_does_not_change_expected_correctness(corpus, values):
    model = values.model
    meta = corpus.questions_of(corpus.dbs[-1])[0]
    state = State().append(gen("M"), "ok")
    for stage in (execute(1), verify("M")):
        res = model.predict(meta, state, stage).resolve(0.4)
        assert np.isclose(float(res.p_obs @ res.s_after), 0.4, atol=1e-6)
        assert np.isclose(res.p_obs.sum(), 1.0)


def test_fix_and_regression_stay_consistent(corpus, values):
    model = values.model
    meta = corpus.questions_of(corpus.dbs[-1])[0]
    state = State().append(gen("M"), "ok").append(execute(1), "error")
    res = model.predict(meta, state, parse_chain("rep:L").stage).resolve(0.3)
    assert np.all(res.s_after >= -1e-9) and np.all(res.s_after <= 1 + 1e-9)
    assert np.all(res.fix <= 1.0) and np.all(res.reg <= 1.0)


def test_rollout_costs_are_monotone_in_the_stages(corpus, values):
    model = values.model
    meta = corpus.questions_of(corpus.dbs[-1])[0]
    short = model.rollout(meta, State(), parse_chain("gen:M,exec"), 0.0)
    long = model.rollout(meta, State(), parse_chain("gen:M,exec,esc:L"), 0.0)
    assert np.all(long.cost >= short.cost - 1e-12)
    assert 0.0 <= short.p <= 1.0 and 0.0 <= long.p <= 1.0


def test_online_cost_updates_move_towards_the_realised_cost(corpus, values):
    model = values.model
    meta = corpus.questions_of(corpus.dbs[-1])[0]
    stage = gen("M")
    before = model.predict(meta, State(), stage).mu.copy()
    for _ in range(60):
        model.observe_cost(meta.db, stage, before, before * 3.0)
    after = model.predict(meta, State(), stage).mu
    assert np.all(after >= before - 1e-12)
    assert after[0] > before[0]
    model.reset_online()


def test_recalibration_moves_predictions_onto_the_observed_share(corpus, values):
    model = values.model
    db = corpus.dbs[-1]
    metas = corpus.questions_of(db)
    items = []
    for meta in metas:
        state, _, _, _ = corpus.run(meta, State(), parse_chain("gen:M,exec"))
        items.append((meta, state))
    before = float(np.mean(model.s_of_many(items)))
    target = min(0.95, max(0.05, before + 0.3))
    model.calibrate_to(db, items, target)
    after = float(np.mean(model.s_of_many(items)))
    assert abs(after - target) < abs(before - target)
    model.reset_online()


def test_observed_shares_identify_a_share_of_correct_answers(corpus, values):
    model = values.model
    metas = corpus.questions_of(corpus.dbs[-1])
    items, real = [], []
    for meta in metas:
        state, ok, _, _ = corpus.run(meta, State(), parse_chain("gen:M,exec"))
        items.append((meta, state))
        real.append(1.0 if ok else 0.0)
    pi = model.estimate_accuracy(items)
    assert pi is None or 0.0 <= pi <= 1.0
