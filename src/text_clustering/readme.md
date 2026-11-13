# text_clustering

Modules in this directory (see their class/function docstrings for parameters):

- `__init__.py` — Composable text clustering research primitives.
- `coldstart.py` — Compatibility imports; implementation lives in workflows.partitioning.
- `embeddings.py` — Offline fixture encoder plus explicit opt-in model loading.
- `factory.py` — Explicit allowlist: configurations select independent algorithm modules.
- `hierarchical.py` — Compatibility imports; implementation lives in workflows.navigation.
- `index.py` — Versioned sufficient statistics; safe JSON persistence, no pickle.
- `onepass.py` — Compatibility imports; implementation lives in workflows.routing.
- `prototype_snapshot.py` — Non-executable snapshots of fitted prototype routing state.
- `types.py` — Shared contracts. Scores are not calibrated probabilities unless stated.
- `vectors.py` — Finite, nonzero normalized vectors and bounded-memory exact retrieval.
