# BudgetSQL

Code for *Effective Text-to-SQL under Multi-Budget Constraints with Agentic
Planning and Execution-Aware World Models*.

BudgetSQL plans a Text-to-SQL workflow (generation, execution, repair,
verification, escalation) for every question of a batch, under one budget on
**money, tokens and service time**, and pays for its own planning from the same
budget. A world model predicts, from what executed stages returned and without
an LLM call, which further stages would fix or break an answer and what they
would cost. An LLM agent composes candidates from a fixed-size digest of the
batch, a linear program allocates the budget among them, and a meter reserves
every call's worst case so the budget is never exceeded.

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,plots]"
```

## Data

Download the benchmarks yourself: [BIRD](https://bird-bench.github.io) dev and
[Spider 2.0-Lite](https://github.com/xlang-ai/Spider2). Every method replays a
**trace corpus** (every stage on every question, with its outcome and cost).
Record one with real models, or simulate one:

```bash
budgetsql collect --dev-json dev.json --db-root dev_databases --out data/bird.jsonl
budgetsql simulate --out data/sim.jsonl          # no API needed
```

The simulator draws outcomes from a difficulty model instead of calling a
model. It is there for development and the tests, and for trying the planner
without an API key; it is not the paper's protocol, which replays a corpus
recorded with real executors.

Models are set by environment variable (`BUDGETSQL_MODEL_S/M/L`,
`BUDGETSQL_MODEL_AGENT`, `BUDGETSQL_BASE_URL`); any OpenAI-compatible endpoint
works.

## Use

```bash
budgetsql run --method budgetsql --budget V4
budgetsql exp main --trace data/bird.jsonl       # budgetsql exp --help lists the others
budgetsql library --trace data/bird.jsonl        # accuracy and cost of the reference workflows
budgetsql verify                                 # exact checks of the escalation algebra
budgetsql report runs/main.json
```

Budget vectors (`V1`–`V4`, `SP`) are in `budgetsql/config.py`, planner defaults
in `PlannerConfig`; see [configs/](configs/). Add `--workload spider2_lite` for
Spider 2.0-Lite.

## Layout

```
budgetsql/
  grammar.py       stages, states and the workflows composed from them
  features.py      the state summary the world model reads
  costs.py         worst-case cost of a stage under its limits
  trace/           trace corpora: record, simulate, replay
  executors/       model clients, prompts, schemas, SQL execution
  worldmodel/      stage heads, imagined rollouts, calibration
  planner/         agent, tools, allocation program, certificate, meter
  baselines/       fixed workflows, cascades, routers, planners, agents
  analysis/        escalation algebra
  eval/            runs, statistics, tables
```

## License

MIT. See [LICENSE](LICENSE).
