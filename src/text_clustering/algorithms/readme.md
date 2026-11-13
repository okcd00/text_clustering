# algorithms

Modules in this directory (see their class/function docstrings for parameters):

- `gpkmeans.py` — GPKmeans assignment core: equations (5)--(12), Algorithm 2 of nie2026gpkmeans.
- `__init__.py` — Shared implementation module.
- `birch_maintenance.py` — Unlabelled, single-writer maintenance wrappers for the audited BIRCH backend.
- `birch_stream.py` — Independent experimental BIRCH CF-leaf stream; not factory/admission wiring.
- `classix_backend.py` — Unknown-K CLASSIX candidate using the released classixclustering==1.5.3.
- `clustream_data_init.py` — Explicit CluStream macro-readout adaptation; native microcluster updates retained.
- `density_backends.py` — Fixed, label-blind sklearn density baselines (not paper reproductions).
- `discovery_backends.py` — Label-blind, fixed-preset discovery pipelines over caller-owned embeddings.
- `fast_density_backend.py` — Fixed, label-blind UMAP + fast-hdbscan engineering adaptation.
- `finch_backend.py` — Optional finch-clust==0.2.3, exact cosine, fixed first-partition preset.
- `geometric_backends.py` — Given-K geometric engineering baselines; neither is a paper reproduction.
- `knn_routing.py` — Exact cosine kNN engineering baseline, not a named-paper reproduction.
- `leiden_backend.py` — Standalone, label-blind kNN + Leiden engineering baseline (not text SOTA).
- `navigation.py` — Embedding and optional directional-NLI navigation policies.
- `optics.py` — Bounded, label-blind sklearn OPTICS Xi flat-partition baseline.
- `parc_backend.py` — Optional PARC 0.40, with fail-closed guards around unsafe default paths.
- `prototype_routing.py` — Train-support cosine routing: flat or parent-first; no query fitting.
- `river_micro_stream.py` — Pinned River microcluster adapters. Native algorithms, not paper reproductions.
- `river_stream_backend.py` — Guarded native River DBSTREAM, not a batch-fit clustering adapter.
- `sklearn_backends.py` — Official sklearn backends. These are baselines, not paper reproductions.
- `textclust.py` — Pinned River 0.26.1 sparse-TF micro adapter (copy-native-v1).
- `textclust_snapshot.py` — Read-only effective-window prediction on pinned River 0.26.1.
