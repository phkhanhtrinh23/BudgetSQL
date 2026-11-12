from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from ..config import PRICES, Price

DEFAULT_BASE_URL = "https://api.openai.com/v1"


@dataclass
class ChatClient:
    """One tier of executor, or the planning agent."""

    model: str
    tier: str = "M"
    base_url: str = field(default_factory=lambda: os.environ.get(
        "BUDGETSQL_BASE_URL", DEFAULT_BASE_URL))
    api_key_env: str = "OPENAI_API_KEY"
    price: Price | None = None
    reasoning_effort: str | None = None
    retries: int = 4
    timeout: float = 120.0

    def __post_init__(self) -> None:
        self.price = self.price or PRICES.get(self.tier) or PRICES["M"]

    def chat(self, messages: list[dict], max_tokens: int = 600,
             temperature: float = 0.0, timeout: float | None = None,
             n: int = 1, seed: int | None = None, retries: int | None = None) -> dict:
        body = {"model": self.model, "messages": messages,
                "max_completion_tokens": int(max_tokens)}
        if n > 1:
            body["n"] = n
        if seed is not None:
            body["seed"] = seed
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        else:
            body["temperature"] = temperature
        data = json.dumps(body).encode()
        key = os.environ.get(self.api_key_env, "")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        url = self.base_url.rstrip("/") + "/chat/completions"
        last = ""
        for attempt in range(self.retries if retries is None else max(1, retries)):
            started = time.time()
            try:
                req = urllib.request.Request(url, data, headers)
                with urllib.request.urlopen(req, timeout=timeout or self.timeout) as fh:
                    payload = json.load(fh)
                return self._reply(payload, time.time() - started)
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}"
                if exc.code in (400, 401, 403, 404):
                    break
                time.sleep(2 * (attempt + 1))
            except Exception as exc:
                last = str(exc)[:200]
                time.sleep(2 * (attempt + 1))
        return {"text": "", "texts": [], "prompt_tokens": 0, "completion_tokens": 0,
                "usd": 0.0, "latency": 0.0, "ok": False, "error": last}

    def _reply(self, payload: dict, latency: float) -> dict:
        usage = payload.get("usage") or {}
        pin = int(usage.get("prompt_tokens", 0))
        pout = int(usage.get("completion_tokens", 0))
        texts = [(c.get("message") or {}).get("content") or ""
                 for c in payload.get("choices", [])]
        return {
            "text": texts[0] if texts else "",
            "texts": texts,
            "prompt_tokens": pin,
            "completion_tokens": pout,
            "usd": self.price.usd(pin, pout),
            "latency": latency,
            "finish": (payload.get("choices") or [{}])[0].get("finish_reason"),
            "ok": True,
        }


def extract_sql(text: str) -> str:
    blocks = re.findall(r"```(?:sql)?\s*(.*?)```", text, flags=re.S | re.I)
    sql = (blocks[-1] if blocks else text).strip()
    return sql.rstrip(";").strip()


def tiers_from_env() -> dict[str, ChatClient]:
    return {
        "S": ChatClient(os.environ.get("BUDGETSQL_MODEL_S", "qwen2.5-7b-instruct"), "S"),
        "M": ChatClient(os.environ.get("BUDGETSQL_MODEL_M", "gpt-4.1-mini"), "M"),
        "L": ChatClient(os.environ.get("BUDGETSQL_MODEL_L", "gpt-4.1"), "L"),
        "agent": ChatClient(os.environ.get("BUDGETSQL_MODEL_AGENT", "gpt-5"),
                            "agent", reasoning_effort=os.environ.get(
                                "BUDGETSQL_AGENT_EFFORT", "minimal")),
    }
