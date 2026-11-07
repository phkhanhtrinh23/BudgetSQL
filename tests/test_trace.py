import numpy as np

from budgetsql.grammar import Grammar, State, reference_library
from budgetsql.trace import RecordedTrace, SimulatedTrace, TraceCorpus, materialise, write_jsonl


def test_branches_are_reproducible(corpus):
    lib = reference_library()
    meta = corpus.questions[0]
    a = corpus.run(meta, State(), lib["M+R_L"])
    b = corpus.run(meta, State(), lib["M+R_L"])
    assert a[1] == b[1]
    assert np.allclose(a[2], b[2])


def test_a_cheaper_composition_is_a_prefix(corpus):
    lib = reference_library()
    meta = corpus.questions[0]
    one = corpus.run(meta, State(), lib["M"])
    both = corpus.run(meta, State(), lib["M+R_L"])
    assert one[0].history[0] == both[0].history[0]


def test_a_recorded_corpus_replays_the_same_outcomes(tmp_path):
    src = SimulatedTrace(n_questions=20, n_dbs=2, seed=3)
    grammar = Grammar()
    path = tmp_path / "trace.jsonl"
    write_jsonl(path, materialise(src, src.questions(), grammar, max_states=10))
    live = TraceCorpus(src, grammar, max_states=10)
    saved = TraceCorpus(RecordedTrace(path), grammar, max_states=10)
    lib = reference_library()
    for meta in saved.questions[:5]:
        a = live.run(meta, State(), lib["M"])
        b = saved.run(meta, State(), lib["M"])
        assert a[1] == b[1]
        assert np.allclose(a[2], b[2])


def test_training_rows_cover_every_stage_type(corpus):
    kinds = {row.stage.kind for row in corpus.training_rows(corpus.dbs[:1])}
    assert {"generate", "execute", "repair", "escalate"} <= kinds
