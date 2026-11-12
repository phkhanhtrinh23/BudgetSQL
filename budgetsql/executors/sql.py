from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass


@dataclass
class ExecResult:
    ok: bool
    rows: list | None
    error: str = ""
    seconds: float = 0.0

    @property
    def n_rows(self) -> int:
        return len(self.rows or [])

    @property
    def status(self) -> str:
        if not self.ok:
            return "error"
        if self.n_rows == 0:
            return "empty"
        return "valid.one" if self.n_rows == 1 else "valid.many"

    def preview(self, rows: int = 3, width: int = 400) -> str:
        if not self.ok:
            return self.error[:width]
        head = (self.rows or [])[:rows]
        return str(head)[:width]


def execute(db_path: str, sql: str, timeout: float = 10.0,
            max_rows: int = 100_000) -> ExecResult:
    started = time.time()
    if not sql.strip():
        return ExecResult(False, None, "empty query", 0.0)
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
    except Exception as exc:
        return ExecResult(False, None, str(exc)[:300], time.time() - started)
    deadline = started + timeout
    con.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 10_000)
    try:
        cur = con.execute(sql)
        rows = cur.fetchmany(max_rows)
        return ExecResult(True, rows, "", time.time() - started)
    except Exception as exc:
        return ExecResult(False, None, str(exc)[:300], time.time() - started)
    finally:
        con.close()


def result_key(rows: list | None) -> tuple:
    if rows is None:
        return ("error",)
    return tuple(sorted(tuple(map(_norm, r)) for r in rows))


def _norm(v):
    if isinstance(v, float):
        return round(v, 6)
    return v


def matches(candidate: ExecResult, reference: ExecResult) -> bool:
    if not candidate.ok or not reference.ok:
        return False
    return result_key(candidate.rows) == result_key(reference.rows)
