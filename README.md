# BudgetSQL

Code for *Effective Text-to-SQL under Multi-Budget Constraints with Agentic
Planning and Execution-Aware World Models*.

A Text-to-SQL system answers a question with a *workflow*, not a single call:
schema context, a generator, a few samples, execution, repair, a vote, maybe an
escalation to a bigger model. BudgetSQL plans that workflow per question for a
whole batch, under one budget on **money, tokens and service time at once** — and
pays for its own planning out of the same budget.

It spends a cheap stage first and lets a *world model* read what came back. From
an execution error, an empty result or disagreeing samples, the world model
predicts — with no further model call — which stages would fix the answer, which
would break it, and what each would cost. An LLM agent composes candidates from
a fixed-size digest of the batch; a linear program splits the budget among them
and its dual prices say which dimension binds. A meter reserves every call's
worst case before launching it, so the budget is never exceeded on any run.

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,plots]"
```

## Data

Nothing here downloads anything. Get the benchmarks yourself and point the
collector at your copy:

- BIRD dev — <https://bird-bench.github.io>
- Spider 2.0-Lite — <https://github.com/xlang-ai/Spider2>

Everything replays a **trace corpus**: every stage of the grammar on every
question with its own prompt, sample, outcome and charged cost, so all methods
see the same outcomes and a cheaper composition is exactly the prefix of a
longer one. Record one with real models, or generate one with the built-in
simulator.

```bash
export OPENAI_API_KEY=...
budgetsql collect --dev-json dev.json --db-root dev_databases --out data/bird.jsonl --cap 25
budgetsql simulate --out data/sim.jsonl          # no API needed
```

Executors are named by environment variable (`BUDGETSQL_MODEL_S/M/L`,
`BUDGETSQL_MODEL_AGENT`, `BUDGETSQL_BASE_URL`), so any OpenAI-compatible
endpoint works, a local vLLM server included.

## Use

```bash
budgetsql library                                # accuracy and cost of the reference workflows
budgetsql run --method budgetsql --budget V4
budgetsql exp main --runs 5 --trace data/bird.jsonl
budgetsql exp ablation                           # what each design choice contributes
budgetsql verify                                 # exact checks of the escalation algebra
budgetsql report runs/main.json
```

`budgetsql exp` also takes `sweep`, `grid`, `pooled`, `sensitivity`, `online`,
`overhead`, `scaling`, `frontier`, `worldmodel`, `calibration`, `unprofiled`,
`plans`, `scenarios`, `orders`, `strata` and `perdb`. Add `--workload
spider2_lite` for Spider 2.0-Lite, which has its own budget vector, folds,
completion caps and large tier.

Budget vectors live in `budgetsql/config.py`: `V1` prices money, `V2` money and
time, `V3` money and tokens, `V4` all three, `SP` Spider 2.0-Lite. Planner
defaults — the re-solve block, the drift that flags a digest cell, the share of
money planning may take — are in `PlannerConfig`; see [configs/](configs/).

## Layout

```
budgetsql/
  grammar.py       stages, states and the workflows composed from them
  features.py      the state summary the model reads
  trace/           trace corpora: record with real models, simulate, replay
  worldmodel/      stage heads, imagined rollouts, held-out calibration
  planner/         the agent, its tools, the allocation program, the meter
  baselines/       fixed workflows and mixtures, cascades, routers, planners, agents
  analysis/        escalation algebra and its exact checks
  eval/            runs, statistics and tables
```

## Tests

```bash
pytest -q
```

The suite checks the parts that have to be exactly right: the meter never
overspends, a stage that only observes cannot change expected correctness, the
allocation's dual prices satisfy complementary slackness, and the escalation
identities hold in exact rational arithmetic.
