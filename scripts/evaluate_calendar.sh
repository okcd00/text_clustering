#!/usr/bin/env bash
set -euo pipefail
# Existing external verified archive + vector cache required. Never publish automatically.
: "${CALENDAR_PYTHON:?Set CALENDAR_PYTHON to the pinned environment Python}"
: "${CALENDAR_CACHE:?Set CALENDAR_CACHE to an external cache directory}"
: "${CALENDAR_OUTPUT:?Set CALENDAR_OUTPUT to an external output directory}"
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1
"$CALENDAR_PYTHON" -c 'import sys; sys.path.insert(0,"scripts"); from prepare_news_calendar import external; external(sys.argv[1]).mkdir(parents=True,exist_ok=True)' "$CALENDAR_OUTPUT"
for method in frozen unbounded cap-2048 cap-128 rolling-2048; do
  "$CALENDAR_PYTHON" scripts/benchmark_news_calendar.py --cache "$CALENDAR_CACHE" --method "$method" --output "$CALENDAR_OUTPUT/$method.json"
done
printf '%s\n' 'Completed all five methods. Results remain external; review before publication.'
