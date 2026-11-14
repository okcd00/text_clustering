# Calendar maintenance v2: extended horizon and isolated worker memory

Executed 2026-11-14. [Predeclared protocol](../../experiments/calendar-paired-v2/protocol.json) · [reproduction](../../experiments/calendar-paired-v2/README.md) · [aggregate JSON](results.json) · [unchanged v1](../calendar-paired-v1/README.md).

## What changed, and what did not

V2 extends the fixed calendar from Mar 24 to Apr 20 (28 UTC days), with the same warmup, sampling salt, no-labels-in-fitting/maintenance discipline, E5 revision and BIRCH settings. First-week selected IDs/vectors and all method predictions/metrics match v1 exactly. The new horizon was frozen before v2 scoring, but its first week was already examined in v1: this is a prespecified continuation, not an untouched confirmatory test.

There are 2,048 warmup rows plus 11,776 scored rows over 23 nonempty days. Five empty dates (Apr 3–6 and Apr 10) remain in the calendar; they contribute no score or update. Extra 8,192 titles were locally encoded once (120.65 seconds, zero truncation); the old 5,632 vectors were reused byte-for-byte. No model or full-source parser runs inside the algorithm workers.

Two complete fresh-process repetitions have identical scores, predictions and all non-timing/memory fields. V1 prefix checks pass separately. These are deterministic repeats, not independent random samples or confidence intervals.

## Selected-history recurrence audit (not novelty detection)

A read-only audit rechecked all cached-array hashes and matched the v1 prefix against the committed [calendar preflight](../../experiments/calendar-preflight-v1/preflight.json): every window's selected-row, STORY, recurrence and first-title counts agrees. A recurring STORY-day means a STORY observed in **selected warmup or an earlier selected day**, not repeated rows within the same day. The STORY-day denominator sums distinct STORYs per day; it is not the number of globally distinct STORYs.

| Scope | Raw scored rows | STORY-day occurrences | Recurring STORY-day occurrences | Distinct recurrent STORYs | Rows belonging to recurring STORYs |
|---|---:|---:|---:|---:|---:|
| v1: 7 days | 3,584 | 597 | 2 | 2 | 3 |
| v2: 28 days, 23 nonempty | 11,776 | 1,807 | 3 | 3 | 5 |

V2 has 1,804 distinct test STORYs. None recurs from the selected warmup (2,048 rows, 303 STORYs); all three recur from earlier selected test days. Nonzero dates are Mar 28 (1 STORY, 1 raw row), Mar 30 (1 STORY, 2 raw rows), and Apr 13 (1 STORY, 2 raw rows). Other days have zero selected-history recurrence.

The **first-title scoring view retains the same raw learning/history stream**: v1 has 2 recurring rows / 3,553 scored rows and 2 recurring STORY-day occurrences / 595; v2 has 4 / 11,685 rows and 3 / 1,800 STORY-day occurrences. This is title deduplication for scoring, not deduplicated learning or a different historical-label search.

Only 5 / 11,776 raw v2 rows (0.0425%) belong to previously selected STORYs. This offers very little evidence for retention of returning event identities. Absence from the thinned selected history is **not semantic novelty**; recurrence does not prove disappearance and reappearance after a gap. No calibrated novelty alarm, event-return detection, or persistent cluster-identity score is measured. New-label routing, genuine new-topic emergence, and recurrence after absence require separate protocols; do not infer them from daily ARI.

## Primary post-cap comparison

The guard again first skips arrival 2,641 (Mar 29, selected arrival 81). Since prediction precedes the day's updates, the post-cap evaluation begins Mar 30: **17 nonempty scored days, 8,704 rows**. There are 9,136 skipped updates in total. No new parameters were selected using these outcomes.

| Policy | Post-cap STORY ARI | Paired delta vs unbounded | Worse / equal / better days | Worst daily delta | Full-horizon STORY ARI |
|---|---:|---:|---:|---:|---:|
| Frozen after warmup | .269151 | −.062846 | 16 / 0 / 1 | −.149040 | .285768 |
| Unbounded updates | .331996 | .000000 | 0 / 17 / 0 | .000000 | .357993 |
| Freeze at 2,048 CF | .297384 | −.034613 | 12 / 0 / 5 | −.127358 | .332409 |
| Rolling last 2,048 / every 2,048 arrivals | .363977 | +.031980 | 4 / 0 / 13 | −.051816 | .380452 |

Scores are row-weighted within-day ARI, not pooled IDs across dates. The first-title scoring view includes 8,634 post-cap rows and yields deltas −.035023 (cap) and +.032291 (rolling), with the **same raw learning stream**. CATEGORY results are a separate coarse four-class objective and remain in JSON rather than contributing to a combined ranking.

**Interpretation within this pilot:** the v1 short suffix understated the sustained freeze tradeoff. Rolling is a stronger quality candidate here, but still loses on four post-cap days, and its memory result below rules out a simple “better and smaller” conclusion. Nothing here selects a universal production policy, establishes statistical significance, or measures downstream intent/recommendation gain.

## Fresh-worker memory and maintenance cost

Every method/repetition uses a fresh process and the same 13,824 × 384 numeric vector matrix plus small label/mask maps. Hash verification uses bounded 1MiB blocks. All mapped pages are touched, the same dummy clustering/scoring APIs are primed, garbage collection runs and the memory sampler starts **before the baseline**. No archive parsing, text list, tokenization or encoder is present.

| Policy | Baseline RSS, MiB (runs 1 / 2) | Sampled peak RSS above baseline, MiB (1 / 2) | Anonymous-RSS increase, MiB (1 / 2) | Final CF | Final estimator pickle, MiB | Update + maintenance seconds (1 / 2) |
|---|---:|---:|---:|---:|---:|---:|
| Frozen | 162.83 / 163.60 | 9.99 / 10.04 | 9.30 / 9.29 | 891 | 7.05 | .012 / .013 |
| Unbounded | 162.93 / 162.68 | 68.21 / 68.44 | 67.58 / 67.69 | 6,483 | 50.50 | 26.244 / 27.026 |
| 2,048 CF freeze | 163.15 / 162.75 | 20.57 / 20.11 | 19.89 / 19.42 | 2,048 | 15.74 | 3.011 / 2.902 |
| Rolling-2,048 | 162.92 / 162.93 | 69.31 / 68.50 | 68.54 / 67.74 | 1,689 | 13.49 | 13.098 / 13.847 |

Peak delta uses only warmup/predict/update phases; scorer phase and whole-run peaks are also saved. Timing includes sampler/probe overhead and rebuilding; it is not unsampled production throughput. Warmup, prediction and encoding are separate.

**Rolling's small final tree does not give a small observed peak process footprint.** It rebuilt at arrivals 2,048, 4,096, 6,144, 8,192 and 10,240. Synchronous overlap probes observe both old/new trees; first-run RSS at those points was approximately 188.98, 198.15, 208.19, 217.56 and 228.18 MiB. Its observed peak delta is essentially the same scale as unbounded. Tree overlap, allocator retention and other process workspaces are included; this run does not isolate one root cause. Cap reduces the measured delta here but loses quality and is not a hard byte-memory guarantee.

### What the counters mean

- The sampler requests a **nominal 5ms** interval and also probes phase boundaries and rebuilt-old/new-tree overlap. Scheduling/GIL delays can exceed 5ms; a shorter transient can be missed. Reported sampled peaks are observations, not exact allocation maxima or a guaranteed OOM bound. No upper bound on missed excursions is established.
- The post-touch baseline includes approximately 20.38 MiB of resident mapped inputs above the pre-touch state, plus common API/runtime memory. In this container the added mappings are mainly accounted under `RssShmem`, not `RssFile`; retain both counters rather than labeling one as the entire feature cache. The file bytes and four baseline stages are in JSON.
- Incremental RSS/anonymous RSS include model allocations, rolling buffer, allocator retention, numerical workspaces and measurement overhead. They are more isolated than v1, but **not exact BIRCH-owned heap bytes**. Independently maximized components may occur at different instants and need not sum to a simultaneous RSS decomposition.
- Lifetime `ru_maxrss` is separately recorded, before serialization. It cannot be reset to the chosen baseline and is not used as the incremental comparator. Final estimator pickle bytes are measured after sampling stops, avoiding inclusion of the serialization buffer in algorithm peaks. This is only the BIRCH estimator, not a complete saved restart state. Rolling additionally keeps 2,048 × 384 × 4 = 3,145,728 bytes (3.00 MiB) of numeric replay payload, plus Python/container/controller overhead. The 3 MiB is a numeric-payload lower bound, not measured total buffer memory; it is already inside observed process RSS and must not be added to RSS again. No restart checkpoint is saved.
- V1's ~649MiB workers parsed the entire archive and used different memory instrumentation. Do not compare that RSS directly with v2. Do not compare update timings across versions as a speedup claim; v2 adds sampling and a longer workload.

## Evidence boundary and next useful check

This is a thinned news-title stream with silver STORY event groups, not user behavior or a broad text-clustering ranking. No persistent cross-date identity or split/merge correctness is measured. Longer maintenance behavior is now visible, but multiple independent periods, alternate domains and explicit memory-reclamation experiments remain separate future work. Preserve this frozen result before any tuning.

No raw text, source IDs, vector arrays, per-row predictions or private input/prediction digests are committed. Public JSON contains aggregate windows, costs and paired differences only.
