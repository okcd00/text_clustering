# research

Modules in this directory (see their class/function docstrings for parameters):

- `denstream_cached.py` — Research-only derived-state cache; the frozen v1 core remains unchanged.
- `denstream_cached_stream_backend.py` — Research-only cached DenStream adapter with a distinct protocol identity.
- `denstream_faded.py` — Research-only ``denstream-river-faded-cf-v1``; not native River.
- `denstream_stream_backend.py` — Research-only causal window adapter for the audited faded-CF candidate.
- `faded_statistics.py` — Independent, dense faded moments; not a DenStream implementation.
- `minibatch_stream_backend.py` — Research-only, fixed-K sklearn MiniBatchKMeans with explicit chunk arrivals.
