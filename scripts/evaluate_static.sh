#!/usr/bin/env bash
# One round after the documented pinned environment/data/model setup.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${SUITE_PYTHON:?Set SUITE_PYTHON to the isolated unified-static-v1 Python}"
: "${DATA_CACHE:?Set DATA_CACHE outside the repository}"
: "${RUN_DIR:?Set RUN_DIR outside the repository}"
mkdir -p "$RUN_DIR"
# Embeddings are generated only on an explicit request. Otherwise the suite
# verifies cached model revision, row-order hash and vector-file hash.
if [[ "${1:-}" == "--prepare-vectors" ]]; then
  : "${SEMANTIC_PYTHON:?Set SEMANTIC_PYTHON to the pinned semantic-pilot-v1 Python}"
  : "${MODEL_DIR:?Set MODEL_DIR to the pinned local E5 artifacts}"
  "$SEMANTIC_PYTHON" "$ROOT/scripts/prepare_semantic_model.py" --model-dir "$MODEL_DIR"
  "$SEMANTIC_PYTHON" "$ROOT/scripts/benchmark_semantic_pilot.py" --cache "$DATA_CACHE" --model-dir "$MODEL_DIR" --output "$RUN_DIR/semantic-pilot.json"
fi
"$SUITE_PYTHON" "$ROOT/scripts/run_comparison_suite.py" --cache "$DATA_CACHE" --work "$RUN_DIR/work" --output "$RUN_DIR/results.json"
echo "Results: $RUN_DIR/results.json (aggregate JSON; no automatic publication)"
