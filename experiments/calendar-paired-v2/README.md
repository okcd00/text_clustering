# Calendar maintenance v2: longer horizon and fresh-worker memory

The [protocol](protocol.json) freezes 28 UTC days, Mar 24–Apr 20, before examining v2 scores. Dates are an unfiltered consecutive continuation, not selected for label mix or performance. Empty calendar days remain visible. Identical v1 sampling namespace preserves the complete old prefix; v1 files are unchanged. Thus the first week is not new validation evidence.

Four methods retain threshold .3 and all other v1 settings. The known-degenerate 128 budget is not re-run. V2 reports full and post-cap horizons, daily differences, direction counts and worst daily loss. The post-cap horizon begins with the first day whose prediction snapshot starts after the 2048-CF guard's first skipped update. No minimum-quality acceptance threshold is invented from the observed scores.

Raw archive parsing and encoding run only in preparation. Each worker starts fresh and maps numeric arrays, touches file-backed pages before its baseline, and preloads numerical/scoring APIs. Baseline and phase peaks distinguish process RSS, anonymous RSS, file RSS, high-water RSS and model serialization. A 5ms sampler plus synchronous boundary/rebuild probes is observational, not guaranteed continuous allocation profiling. Deltas above baseline include allocator/workspace/worker overhead, not just BIRCH object ownership. Rebuild old/new trees coexist and are measured. No memory limits or OOM guarantee are inferred.

## Reproduce separately

Keep v1's environment, hash-verified data/model and first-run method JSONs. Prepare the extended array cache once (raw values remain external):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/calendar-env/bin/python scripts/prepare_calendar_v2.py \
  --cache /tmp/calendar-data --model-dir /tmp/calendar-model --output /tmp/calendar-v2
CALENDAR_PYTHON=/tmp/calendar-env/bin/python \
CALENDAR_V1_CACHE=/tmp/calendar-data CALENDAR_V2_CACHE=/tmp/calendar-v2 \
  bash scripts/evaluate_calendar_v2.sh
```

The first command checks v1 ordered input and vector hashes, reuses those exact 5,632 vectors and encodes only the appended rows. Worker methods consume the same four mapped numeric arrays. They do not load a model or source archive. Label arrays are pre-touched only for resident-memory accounting and used for quality only in the scoring phase; their values never enter fitting, maintenance or hyperparameter selection. All timing is one numerical thread. The 5ms memory sampler and explicit probes add overhead to these timings; they are not unsampled production throughput.

The independent wrapper executes two repetitions, fails if either is incomplete or non-resource/time results differ, checks that all seven v1 days and their predictions are identical, and emits aggregate-only output. It never uploads or changes existing benchmark files. V1's source parser and scorer protocol remain available unchanged. To rebuild lost v1 caches, follow the v1 instructions first; no network/API fallback is used by the worker.

Memory fields are bytes. `imports` is the post-import/post-hash process before maps; `mapped_before_touch` shows mapped-but-not-resident state; `after_touch` visits file-backed pages; the reported baseline additionally primes sklearn and starts the sampler. Peak-over-baseline is an incremental **process** observation. It includes BIRCH, rolling buffer, allocator retention, numerical workspaces and sampler overhead. RSS components can peak at different instants, so separate maxima are not a single simultaneous decomposition. `resource_highwater` is lifetime process high-water, not resettable at baseline. Final storage measurement is taken after sampling ends so its pickle buffer does not inflate the sampled algorithm peaks.

## Execution

[Measured v2 results](../../results/calendar-paired-v2/README.md) contain two completed repetitions. They report both qualities and the counterexample to treating a smaller serialized rolling tree as smaller peak RSS. The environment uses v1's pinned CPU dependency lock. The v1 result filenames accepted by the prefix checker are `{method}-1.json` or the one-command wrapper's `{method}.json` in `CALENDAR_V1_CACHE`.

Estimator pickle size excludes the rolling replay/controller state. Rolling's numeric replay payload alone is 2,048 × 384 float32 values = 3.00 MiB, plus Python/container overhead; observed RSS already includes it. This is not a complete disk restart footprint.
