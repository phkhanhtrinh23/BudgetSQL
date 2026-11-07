# Presets

`budgetsql` takes its settings from `budgetsql/config.py`, which is plain Python,
so a deployment overrides them in code rather than through a config file:

```python
from budgetsql.config import BUDGETS, PRICES, PlannerConfig, Price, vec

BUDGETS["house"] = vec(2.0e-3, 4000, 3.0)     # per question: USD, tokens, seconds
PRICES["L"] = Price(1.00, 4.00)               # USD per million tokens
cfg = PlannerConfig(replan_block=50, planning_cap=0.05)
```

Executor endpoints and models come from the environment:

| variable | what it names |
| --- | --- |
| `BUDGETSQL_BASE_URL` | an OpenAI-compatible endpoint (a local vLLM server works) |
| `BUDGETSQL_MODEL_S` / `_M` / `_L` | the three executor tiers, cheapest first |
| `BUDGETSQL_MODEL_AGENT` | the model that plans the batch |
| `BUDGETSQL_AGENT_EFFORT` | its reasoning effort, where the provider takes one |
| `OPENAI_API_KEY` | the key sent to that endpoint |
