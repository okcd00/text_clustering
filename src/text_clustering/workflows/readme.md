# workflows

Modules in this directory (see their class/function docstrings for parameters):

- `partitioning.py` — Task orchestration independent of the imported clustering algorithm.
- `__init__.py` — Executable workflows; research topics are broader than this implemented set.
- `evolution.py` — Persistent vector-stream orchestration, deliberately without gold labels.
- `navigation.py` — Budgeted multi-path navigation with independent ENTER / EMIT / ADD.
- `prototype_routing.py` — Frozen known-support routing over precomputed vectors, without calibration.
- `routing.py` — Retrieve, gate, optionally rerank with NLI; pure prediction, explicit update.
- `text_evolution.py` — Causal raw-text windows, intentionally separate from the dense workflow.
- `text_snapshot_evolution.py` — Explicit raw-text facade and separately identified snapshot-window protocol.
