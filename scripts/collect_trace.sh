#!/usr/bin/env bash
# Record a trace corpus by running the stage grammar against real models.
#
#   export OPENAI_API_KEY=...
#   scripts/collect_trace.sh /path/to/dev.json /path/to/dev_databases data/bird.jsonl 25
#
set -euo pipefail
cd "$(dirname "$0")/.."

DEV_JSON="${1:?usage: collect_trace.sh DEV_JSON DB_ROOT OUT [CAP_USD]}"
DB_ROOT="${2:?}"
OUT="${3:?}"
CAP="${4:-10}"
PY="${VENV:-.venv}/bin/python"

: "${BUDGETSQL_MODEL_S:=qwen2.5-7b-instruct}"
: "${BUDGETSQL_MODEL_M:=gpt-4.1-mini}"
: "${BUDGETSQL_MODEL_L:=gpt-4.1}"
export BUDGETSQL_MODEL_S BUDGETSQL_MODEL_M BUDGETSQL_MODEL_L

mkdir -p "$(dirname "$OUT")"
"$PY" -m budgetsql.cli collect \
  --dev-json "$DEV_JSON" --db-root "$DB_ROOT" --out "$OUT" \
  --cap "$CAP" --max-states "${MAX_STATES:-48}" \
  --sample-rows "${SAMPLE_ROWS:-0}" ${QUESTIONS:+--questions "$QUESTIONS"}

echo "== recorded $OUT; use it with --trace $OUT"
