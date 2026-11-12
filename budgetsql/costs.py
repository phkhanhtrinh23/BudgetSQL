from __future__ import annotations

import numpy as np

from .config import PRICES, ExecutorLimits
from .grammar import Stage


def sql_statements(stage: Stage) -> int:
    if stage.kind == "execute":
        return max(1, stage.samples)
    if stage.kind in ("repair", "escalate"):
        return 1
    return 0


def stage_worst_case(stage: Stage, prompt_tokens: float,
                     limits: ExecutorLimits | None = None,
                     cap: int | None = None, timeout_s: float | None = None,
                     sql_timeout_s: float | None = None) -> np.ndarray:
    limits = limits or ExecutorLimits()
    sql_t = limits.sql_timeout_s if sql_timeout_s is None else float(sql_timeout_s)
    n_sql = sql_statements(stage)
    if stage.kind == "context":
        return np.array([0.0, 0.0, float(limits.context_timeout_s)])
    if not stage.llm_calls:
        return np.array([0.0, 0.0, n_sql * sql_t])
    price = PRICES.get(stage.tier)
    n = max(1, stage.samples)
    c = limits.cap_for(stage.tier, stage.cap) if cap is None else int(cap)
    t = limits.timeout_s if timeout_s is None else float(timeout_s)
    usd = 0.0 if price is None else (prompt_tokens * price.inp + c * price.out) * 1e-6 * n
    return np.array([usd, (prompt_tokens + c) * n, t * n + n_sql * sql_t])
