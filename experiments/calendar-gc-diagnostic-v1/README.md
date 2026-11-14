# Rolling BIRCH lifecycle diagnostic (not a new quality benchmark)

Base: published calendar v2 `b2d3db7`. Keep its source, selection, features, dates, row order, threshold, update/rebuild rule, predictions-before-update, mmap pre-touch and fresh-process measurement unchanged. Compare automatic Python GC with one explicit `gc.collect()` immediately after each 2,048-arrival model swap. Run two fresh processes per arm, alternating order. No tuning based on labels; this is a mechanism diagnostic motivated by v2's measured rolling peak, not held-out confirmatory evidence.

Both arms use identical weak-reference and RSS instrumentation. Probe before construction, old/new overlap, after swap and after explicit collection (or no-op). Weak references cover retired linked leaf nodes including dummy head, not every object/array. Record all retired weakref survivors, collected-object count (not bytes, may include unrelated objects), GC duration, anonymous/process RSS, baseline and sampled algorithm peak. Neither weakrefs nor event records retain the retired tree.

Required checks: exact daily prediction digest, CF counts and scores across arms/repeats; same v2 public per-day scores and counts. Reconstructed feature bytes may differ on another CPU/runtime; report any difference and fail parity claims rather than silently reuse old quality. Automatic GC remains enabled in both arms; do not disable GC or call malloc_trim, which would change the question.

Interpretation: disappearance of weakrefs after GC supports delayed collection of cyclic leaf structures. RSS remaining after weakrefs die does **not** alone prove allocator retention: live model/buffer, workspaces, Python/NumPy allocators, mappings and measurement granularity also contribute. Peak includes unavoidable old/new overlap. No claim of hard memory cap, production benefit, leak, or cross-task policy ranking.

Nominal 5 ms sampler is observational; short peaks/GIL delays may be missed. Explicit probes capture specified lifecycle points, not continuous memory ownership. Timing includes probes and GC. Estimator serialization excludes rolling replay/controller state; the numeric replay payload alone remains 3 MiB and is already in RSS.

## Run after preparing the unchanged v2 external cache

Use v1's pinned dependency lock in an isolated environment. Store official source/model/feature caches outside Git in an owner-private persistent workspace folder. Do not commit raw rows, labels, vectors, per-row predictions, or private digests.

```bash
for pair in 'automatic 1' 'explicit-after-swap 1' 'explicit-after-swap 2' 'automatic 2'; do
  set -- $pair
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON" scripts/diagnose_calendar_gc.py \
    --cache "$CACHE" --arm "$1" --output "$CACHE/gc-$1-$2.json"
done
```

The published v2 worker/protocol/results are not modified by this diagnostic.

Also run an unchanged fresh v2 control, then validate/export only aggregates:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON" scripts/benchmark_calendar_v2.py \
  --cache "$CACHE" --method rolling-2048 --output "$CACHE/gc-uninstrumented.json"
"$PYTHON" scripts/summarize_calendar_gc.py --cache "$CACHE" \
  --output results/calendar-gc-diagnostic-v1/results.json
```

The exporter fails if current per-day scores/counts differ from published v2, if any arm/repeat prediction digest differs from the fresh uninstrumented control, or if explicit GC leaves any observed retired leaf node alive. Original private feature/prediction byte digests were not published, so comparison against those unavailable original bytes is not claimed.
