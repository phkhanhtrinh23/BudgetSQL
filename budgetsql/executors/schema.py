"""Reading a SQLite schema and pruning it for the generator.

Pruning is a per-question decision, not a preprocessing step: it removes a large
share of the generator's prompt tokens and it fails on the questions whose
needed columns it drops, so which level to use is left to the plan.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

#: How much of each table's column list survives at each level.
LEVELS = {"full": 1.0, "prune1": 0.52, "prune2": 0.33}


@dataclass
class Column:
    table: str
    name: str
    type: str = ""


@dataclass
class Schema:
    tables: dict[str, list[Column]] = field(default_factory=dict)
    create_sql: dict[str, str] = field(default_factory=dict)
    sample_rows: dict[str, list] = field(default_factory=dict)

    @property
    def n_tables(self) -> int:
        return len(self.tables)

    def columns(self) -> list[Column]:
        return [c for cols in self.tables.values() for c in cols]


def read_schema(db_path: str, sample_rows: int = 0) -> Schema:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    out = Schema()
    try:
        names = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'")]
        for t in names:
            cols = [Column(t, r[1], r[2]) for r in con.execute(f'PRAGMA table_info("{t}")')]
            out.tables[t] = cols
            row = con.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (t,)).fetchone()
            out.create_sql[t] = (row[0] if row and row[0] else "") or ""
            if sample_rows:
                try:
                    out.sample_rows[t] = con.execute(
                        f'SELECT * FROM "{t}" LIMIT {int(sample_rows)}').fetchall()
                except Exception:
                    out.sample_rows[t] = []
    finally:
        con.close()
    return out


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z_0-9]+", text or "")
            if len(w) > 2}


def prune(schema: Schema, question: str, evidence: str = "",
          level: str = "prune1") -> Schema:
    """Keep the tables and columns most similar to the question and evidence.

    A table's score is how many of its name's and its columns' words the
    question uses; the highest-scoring tables are kept whole, the rest keep only
    their key-looking and matching columns.
    """
    if level == "full" or level not in LEVELS:
        return schema
    want = _tokens(question) | _tokens(evidence)
    keep_share = LEVELS[level]

    def score(table: str, cols: list[Column]) -> int:
        s = len(_tokens(table) & want) * 3
        for c in cols:
            s += len(_tokens(c.name) & want)
        return s

    ranked = sorted(schema.tables.items(), key=lambda kv: -score(*kv))
    n_keep = max(1, int(round(len(ranked) * max(keep_share, 0.2))))
    out = Schema()
    for i, (t, cols) in enumerate(ranked):
        if i < n_keep:
            kept = cols
        else:
            kept = [c for c in cols
                    if _tokens(c.name) & want
                    or c.name.lower().endswith("id")
                    or c.name.lower() == "name"]
            if not kept:
                continue
        out.tables[t] = kept
        out.create_sql[t] = schema.create_sql.get(t, "")
        if t in schema.sample_rows:
            out.sample_rows[t] = schema.sample_rows[t]
    return out


def render(schema: Schema, sample_rows: int = 0) -> str:
    """The schema as the generator reads it."""
    parts = []
    for t, cols in schema.tables.items():
        cols_txt = ", ".join(f"{c.name} {c.type}".strip() for c in cols)
        parts.append(f"TABLE {t}({cols_txt})")
        if sample_rows and schema.sample_rows.get(t):
            rows = schema.sample_rows[t][:sample_rows]
            parts.append(f"  sample rows: {rows}")
    return "\n".join(parts)
