#!/usr/bin/env bash
# Run supported survey scenarios from pinned caches; never publishes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${SUITE_PYTHON:?Set SUITE_PYTHON to the isolated static suite Python}"
: "${ROUTING_PYTHON:?Set ROUTING_PYTHON to the pinned lexical benchmark Python}"
: "${DATA_CACHE:?Set DATA_CACHE to the external data cache}"
: "${RUN_DIR:?Set RUN_DIR to an external directory for this run}"
"$SUITE_PYTHON" -c '
from pathlib import Path
import sys
root=Path(sys.argv[1]).resolve()
for value in sys.argv[2:]:
    p=Path(value).resolve()
    if p==root or root in p.parents:
        raise SystemExit("Data and evaluation outputs must stay outside the repository.")
' "$ROOT" "$RUN_DIR" "$DATA_CACHE"
if (( $# > 1 )) || [[ ${1:-} != "" && ${1:-} != "--prepare-vectors" ]]; then
  printf 'Usage: %s [--prepare-vectors]\n' "$0" >&2
  exit 2
fi
mkdir -p "$RUN_DIR"
RUN_DIR="$RUN_DIR/static" bash "$ROOT/scripts/evaluate_static.sh" "$@"
"$ROUTING_PYTHON" "$ROOT/scripts/benchmark_routing_v2.py" --cache "$DATA_CACHE" --output "$RUN_DIR/routing.json"
"$SUITE_PYTHON" "$ROOT/scripts/export_survey_results.py" --static "$RUN_DIR/static/results.json" --routing "$RUN_DIR/routing.json" --output "$RUN_DIR/unified-results.json"
printf 'Static, routing and combined comparison artifacts: %s\n' "$RUN_DIR"
