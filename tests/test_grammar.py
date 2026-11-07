from budgetsql.grammar import (LIBRARY_ORDER, MAX_LLM_CALLS, Grammar, State,
                               parse_chain, parse_stage, reference_library, shape)


def test_library_is_wellformed():
    g = Grammar()
    lib = reference_library()
    assert set(LIBRARY_ORDER) == set(lib)
    for name in LIBRARY_ORDER:
        wf = lib[name]
        assert g.is_wellformed(wf), name
        assert wf.max_llm_calls() <= MAX_LLM_CALLS, name


def test_call_budget_is_respected():
    g = Grammar()
    state = State()
    for _ in range(6):
        actions = g.actions(state)
        if not actions:
            break
        a = actions[0]
        state = state.append(a, a.observations()[0])
    assert state.llm_calls <= MAX_LLM_CALLS


def test_parse_round_trip():
    chain = parse_chain("gen:Mx3,exec:3,esc:L")
    stages = list(chain.stages())
    assert [s.id for s in stages] == ["generate:Mx3", "exec:3", "escalate:L"]
    assert parse_stage("gen:M@prune1").ctx == "prune1"


def test_shapes_distinguish_branches():
    a = parse_chain("esc:L")
    b = parse_chain("esc:M")
    assert shape(a) != shape(b)
