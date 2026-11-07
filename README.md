# BudgetSQL

Text-to-SQL systems answer a question with a *workflow*, not a single call: schema
context, a generator, a few samples, execution, repair, a vote, maybe an escalation to
a bigger model. BudgetSQL plans that workflow per question for a whole batch, under one
budget on **money, tokens and service time at once** — and pays for its own planning out
of the same budget.

The idea is to spend a cheap stage first and then let a *world model* read what it
returned. From an execution error, an empty result or disagreeing samples, the world
model predicts — with no further model call — which stages would fix an answer, which
would break it, and what each would cost. An LLM agent composes candidates from a
fixed-size digest of the batch; a linear program splits the budget among them and its
dual prices say which dimension binds. A meter reserves every call's worst case before
launching it, so the budget is never exceeded on any run.

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,plots]"
```

## Run it

```bash
PROFILE=smoke scripts/run_all.sh     # a couple of minutes, end to end
scripts/run_all.sh                   # the full set, results under runs/
```

Or one piece at a time:

```bash
budgetsql library                    # accuracy and cost of the reference workflows
budgetsql run --method budgetsql --budget V4
budgetsql exp main --runs 3          # every method under every budget vector
budgetsql exp ablation               # what each design choice contributes
budgetsql verify                     # exact checks of the escalation algebra
budgetsql report runs/<stamp>/main.json
```

Budget vectors live in `budgetsql/config.py`: `V1` prices money, `V2` money and time,
`V3` money and tokens, `V4` all three. Planner defaults — the re-solve block, the drift
that flags a digest cell, the share of money planning may take — are in `PlannerConfig`
and can be overridden per run; see [configs/](configs/).

## Traces

Everything replays a **trace corpus**: every stage of the grammar on every question with
its own prompt, sample, outcome and charged cost, so all methods see the same outcomes
and a cheaper composition is exactly the prefix of a longer one.

```bash
# record one with real models
export OPENAI_API_KEY=...
scripts/collect_trace.sh dev.json dev_databases data/bird.jsonl 25
budgetsql exp main --trace data/bird.jsonl

# or generate one with the built-in simulator (the default)
budgetsql simulate --questions 1530 --dbs 11 --out data/sim.jsonl
```

Executors are named by environment variable (`BUDGETSQL_MODEL_S/M/L`,
`BUDGETSQL_MODEL_AGENT`, `BUDGETSQL_BASE_URL`), so any OpenAI-compatible endpoint works,
including a local vLLM server.

## Layout

```
budgetsql/
  grammar.py       stages, states and the workflows composed from them
  features.py      the state summary the model reads
  trace/           trace corpora: record with real models, simulate, replay
  worldmodel/      stage heads, imagined rollouts, held-out calibration
  planner/         the agent, its tools, the allocation program, the meter
  baselines/       fixed mixtures, cascades, routers, workflow planners, agents
  analysis/        escalation algebra and its exact checks
  eval/            runs, statistics and tables
```

## Tests

```bash
pytest -q
```

The suite checks the parts that have to be exactly right: the meter never overspends, a
stage that only observes cannot change expected correctness, the allocation's dual prices
satisfy complementary slackness, and the escalation identities hold in exact rational
arithmetic.
