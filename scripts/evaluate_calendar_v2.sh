#!/usr/bin/env bash
set -euo pipefail
: "${CALENDAR_PYTHON:?Set pinned environment Python}"
: "${CALENDAR_V2_CACHE:?Set prepared external v2 cache}"
: "${CALENDAR_V1_CACHE:?Set external v1 cache, including verified first-run results}"
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1
for repeat in 1 2; do
  for method in frozen unbounded cap-2048 rolling-2048; do
    "$CALENDAR_PYTHON" scripts/benchmark_calendar_v2.py --cache "$CALENDAR_V2_CACHE" --method "$method" --output "$CALENDAR_V2_CACHE/$method-$repeat.json"
  done
done
"$CALENDAR_PYTHON" scripts/summarize_calendar_v2.py --cache "$CALENDAR_V2_CACHE" --v1-cache "$CALENDAR_V1_CACHE" --output "$CALENDAR_V2_CACHE/aggregate.json"
printf '%s\n' 'Two repeats and v1 prefix checked. Aggregate remains external; no publication.'
