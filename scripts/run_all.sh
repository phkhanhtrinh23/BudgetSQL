#!/usr/bin/env bash
# End to end: set up, build a trace corpus, run every comparison, write results.
#
#   scripts/run_all.sh                 # the default workload
#   PROFILE=smoke scripts/run_all.sh   # a quick pass, a couple of minutes
#   TRACE=data/bird.jsonl scripts/run_all.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE="${PROFILE:-full}"
QUESTIONS="${QUESTIONS:-1530}"
DBS="${DBS:-11}"
RUNS="${RUNS:-3}"
MAX_STATES="${MAX_STATES:-24}"
OUT="${OUT:-runs/$(date +%Y%m%d-%H%M%S)}"
VENV="${VENV:-.venv}"
TRACE="${TRACE:-}"

if [[ "$PROFILE" == "smoke" ]]; then
  QUESTIONS="${QUESTIONS_SMOKE:-200}"; DBS=4; RUNS=1; MAX_STATES=16
fi

# ---------------------------------------------------------------- environment
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "== creating $VENV"
  python3 -m venv "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -e ".[dev,plots]"

mkdir -p "$OUT"
echo "== results -> $OUT"

COMMON=(--profile "$PROFILE" --questions "$QUESTIONS" --dbs "$DBS"
        --max-states "$MAX_STATES" --cache runs/cache)
if [[ -n "$TRACE" ]]; then COMMON+=(--trace "$TRACE"); fi

run () {                       # run <name> [extra args...]
  local name="$1"; shift
  echo "== $name"
  "$PY" -m budgetsql.cli exp "$name" "${COMMON[@]}" --out "$OUT/$name.json" "$@"
}

# ------------------------------------------------------------------ pipeline
echo "== self-checks"
"$PY" -m pytest -q
"$PY" -m budgetsql.cli verify

echo "== reference workflows"
"$PY" -m budgetsql.cli library "${COMMON[@]}" --out "$OUT/library.json"

run frontier
run worldmodel
run main        --runs "$RUNS"
run sweep
run plans
if [[ "$PROFILE" == "smoke" ]]; then
  run ablation    --budgets V4 --runs 1
else
  run ablation    --runs "$RUNS"
fi
run sensitivity --budgets V4 --runs "$RUNS"
run overhead
run online      --days 5 --day-questions $(( QUESTIONS / 4 ))
run scenarios   --budgets V4

echo
echo "== done; tables in $OUT/*.txt"
