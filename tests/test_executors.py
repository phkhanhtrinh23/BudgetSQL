import json
import sqlite3

import numpy as np
import pytest

from budgetsql.config import PRICES
from budgetsql.executors import prompts
from budgetsql.executors.llm import extract_sql
from budgetsql.executors.schema import prune, read_schema, render
from budgetsql.executors.sql import execute, matches
from budgetsql.grammar import Grammar, State, execute as exec_stage, gen, repair
from budgetsql.trace.collect import Collector, Question, load_bird


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "shop.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE customers (customer_id INTEGER, name TEXT, city TEXT);
        CREATE TABLE orders (order_id INTEGER, customer_id INTEGER, total REAL);
        INSERT INTO customers VALUES (1, 'Ada', 'Lisbon'), (2, 'Bo', 'Porto');
        INSERT INTO orders VALUES (1, 1, 10.0), (2, 1, 5.0), (3, 2, 7.5);
        """
    )
    con.commit()
    con.close()
    return str(path)


def test_schema_is_read_and_pruned(db):
    schema = read_schema(db, sample_rows=1)
    assert set(schema.tables) == {"customers", "orders"}
    assert schema.n_tables == 2
    full = render(schema)
    pruned = render(prune(schema, "How many orders per customer?", level="prune2"))
    assert len(pruned) <= len(full)
    assert "orders" in pruned


def test_execution_reports_the_status_the_model_reads(db):
    ok = execute(db, "SELECT name FROM customers WHERE city = 'Lisbon'")
    assert ok.ok and ok.status == "valid.one"
    many = execute(db, "SELECT name FROM customers")
    assert many.status == "valid.many"
    assert execute(db, "SELECT name FROM customers WHERE city = 'Nowhere'").status == "empty"
    bad = execute(db, "SELECT nope FROM customers")
    assert bad.status == "error" and bad.preview()


def test_result_sets_match_regardless_of_row_order(db):
    a = execute(db, "SELECT name FROM customers ORDER BY name")
    b = execute(db, "SELECT name FROM customers ORDER BY name DESC")
    assert matches(a, b)
    assert not matches(a, execute(db, "SELECT city FROM customers"))


def test_sql_is_taken_from_the_last_fenced_block():
    text = "thinking\n```sql\nSELECT 1\n```\nbetter:\n```sql\nSELECT 2;\n```"
    assert extract_sql(text) == "SELECT 2"
    assert extract_sql("SELECT 3;") == "SELECT 3"


def test_prompts_carry_the_evidence():
    msgs = prompts.generate("TABLE t(a)", "how many?", "a is a count")
    assert "a is a count" in msgs[-1]["content"]
    assert prompts.repair("TABLE t(a)", "q", "SELECT", "no such column")[-1]["content"]
    assert prompts.verify("q", "SELECT 1", "[(1,)]")[-1]["content"]


class _StubClient:
    """Answers with a fixed query, priced like the tier it stands in for."""

    def __init__(self, sql: str, tier: str = "M"):
        self.sql = sql
        self.price = PRICES[tier]
        self.calls = 0

    def chat(self, messages, max_tokens=600, temperature=0.0, timeout=None,
             n=1, seed=None):
        self.calls += 1
        return {"text": f"```sql\n{self.sql}\n```", "texts": [], "prompt_tokens": 120,
                "completion_tokens": 20, "usd": self.price.usd(120, 20),
                "latency": 0.2, "ok": True, "finish": "stop"}


def test_collector_records_a_branch_with_its_cost(db):
    gold = "SELECT name FROM customers WHERE city = 'Lisbon'"
    tiers = {t: _StubClient(gold if t == "L" else "SELECT nope FROM customers", t)
             for t in ("S", "M", "L")}
    collector = Collector(tiers, Grammar(), max_states=6, spend_cap_usd=1.0,
                          verbose=False)
    q = Question("q0", "shop", db, "Who lives in Lisbon?", reference_sql=gold)
    ref = collector.reference(q)
    assert ref.ok

    state = State()
    out, held = collector.run_stage(q, ref, state, gen("M"), None)
    assert out.answer_correct is False and out.cost[0] > 0
    state = state.append(gen("M"), out.obs)

    out, held = collector.run_stage(q, ref, state, exec_stage(1), held)
    assert out.obs == "error"
    state = state.append(exec_stage(1), out.obs)

    out, held = collector.run_stage(q, ref, state, repair("L"), held)
    assert out.answer_correct is True and out.is_fix
    assert np.all(out.worst_case >= out.cost)


def test_a_collected_corpus_round_trips(db, tmp_path):
    gold = "SELECT name FROM customers"
    tiers = {t: _StubClient(gold, t) for t in ("S", "M", "L")}
    collector = Collector(tiers, Grammar(), max_states=4, spend_cap_usd=1.0,
                          verbose=False)
    q = Question("q0", "shop", db, "Who are the customers?", reference_sql=gold)
    out = tmp_path / "trace.jsonl"
    stats = collector.collect([q], out)
    assert stats["spent_usd"] > 0

    from budgetsql.grammar import reference_library
    from budgetsql.trace import RecordedTrace, TraceCorpus
    corpus = TraceCorpus(RecordedTrace(out), Grammar(), max_states=4)
    assert len(corpus.questions) == 1
    _, ok, cost, _ = corpus.run(corpus.questions[0], State(),
                                reference_library()["M"])
    assert ok and cost[0] > 0


def test_bird_items_are_located(tmp_path, db):
    root = tmp_path / "dbs" / "shop"
    root.mkdir(parents=True)
    (root / "shop.sqlite").write_bytes(open(db, "rb").read())
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps([{"question_id": 7, "db_id": "shop",
                                "question": "how many?", "evidence": "",
                                "SQL": "SELECT 1", "difficulty": "simple"}]))
    items = load_bird(dev, tmp_path / "dbs")
    assert len(items) == 1 and items[0].qid == "7" and items[0].db == "shop"
