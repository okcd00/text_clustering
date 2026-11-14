# Supported-parent hierarchy E5 pilot: small mixed parent gain, costly rejection

Executed 2026-11-14. [Frozen protocol and reproduction](../../experiments/hierarchy-e5-v1/README.md) · [aggregate results](results.json) · [unchanged split preflight](../../experiments/hierarchy-preflight-v1/README.md).

## Scope and verification

Exact official MASSIVE zh-CN revision, all source hashes and the complete committed preflight were verified. Support: 606 utterances, 31 known leaves, 18 parents. Validation: 223 known / 111 hidden queries. Test: 230 known / 104 hidden queries, with 13 test-hidden leaves disjoint from the 14 validation-hidden leaves. Known topology uses opaque IDs; no original semantic names or hidden-node descriptions enter the methods. No clustering K, absent-parent inference or new-node creation is evaluated.

Pinned multilingual E5-small encodes only utterances. Both methods use identical normalized leaf prototypes and **the same frozen leaf predictions/rejection mask**. Flat maps the maximum-similarity leaf to its parent; the alternative compares equal-leaf-weight parent means. Two fresh workers agree exactly on predictions/masks and every non-resource/time result. The test scorer is loaded after prediction/mask freezing. No test-driven reranking or parameter adjustment occurred.

## Validation-calibrated rejection is not production-ready

The preregistered objective maximizes the average of known acceptance and hidden rejection on validation; ties favor known coverage then the smaller threshold. Its selected threshold is **0.9438210129737854** (reject below). Validation balanced accuracy is .654426: 97 / 223 known accepted, 97 / 111 hidden rejected. This objective has no business-specific false-alarm constraint.

On test, it rejects **89 / 104 hidden queries (85.58%)**, but also **132 / 230 known queries (57.39% false novelty)**. Known acceptance coverage is only **98 / 230 (42.61%)**. Unconditional leaf accuracy is 192 / 230 (83.48%) before applying the rejection rule; among accepted known queries it is 93 / 98 (94.90%), but joint acceptance-and-correct-leaf is only 93 / 230 (40.43%). Do not present the high accepted-subset accuracy without its coverage cost. Here rejection detects a held-out leaf under a fixed supervised split, not semantic novelty in a natural stream.

## Test parent comparison

Parent correctness is evaluated over **all hidden queries**, irrespective of rejection. Joint success requires both leaf rejection and correct parent. Conditional accuracy uses only rejected hidden queries and reports its denominator. Parent-macro averages gold parents with hidden queries equally, not support sizes.

| Supported children of gold parent | Hidden rows / parents | Hidden rejected (shared) | Arm | Correct parent / all hidden | Joint / all hidden | Correct parent / rejected hidden | Parent macro / joint macro |
|---|---:|---:|---|---:|---:|---:|---:|
| All | 104 / 13 | 89 / 104 | Flat max-leaf | 64 / 104 | 49 / 104 | 49 / 89 | .615385 / .471154 |
| All | 104 / 13 | 89 / 104 | Equal-leaf mean | 66 / 104 | 51 / 104 | 51 / 89 | .634615 / .490385 |
| One | 56 / 7 | 47 / 56 | Flat max-leaf | 31 / 56 | 22 / 56 | 22 / 47 | .553571 / .392857 |
| One | 56 / 7 | 47 / 56 | Equal-leaf mean | 28 / 56 | 19 / 56 | 19 / 47 | .500000 / .339286 |
| Multiple | 48 / 6 | 42 / 48 | Flat max-leaf | 33 / 48 | 27 / 48 | 27 / 42 | .687500 / .562500 |
| Multiple | 48 / 6 | 42 / 48 | Equal-leaf mean | 38 / 48 | 32 / 48 | 32 / 42 | .791667 / .666667 |

Each test-hidden parent has eight queries, so row accuracy and parent-macro coincide here; they are not independent evidence. Macro conditional parent accuracy among rejected hidden queries is .572344 (flat) versus .594322 (mean), over 13 parents. All per-role/subgroup macro denominators are in JSON; an empty conditional denominator is null, never perfect accuracy.

**The mean-parent arm gains only 2 / 104 overall**, combining +5 / 48 under multi-child parents with −3 / 56 under single-child parents. Even single-child gold parents can differ between arms because *competing* multi-child parent representations change. This is not evidence for a universal advantage or choosing an arm post hoc by a query's unknown gold-parent subgroup.

Known-query denominators remain visible:

| Supported-child stratum | Known rows / parents | Accepted / known | False novelty / known | Correct leaf / accepted |
|---|---:|---:|---:|---:|
| All | 230 / 18 | 98 / 230 | 132 / 230 | 93 / 98 |
| One | 89 / 12 | 29 / 89 | 60 / 89 | 28 / 29 |
| Multiple | 141 / 6 | 69 / 141 | 72 / 141 | 65 / 69 |

Validation parent results (calibration split, not independent test evidence): flat 70 / 111 parent hits and 57 / 111 joint; mean 75 / 111 and 62 / 111. Parent-macro is .632653 / .677296 and joint macro .516582 / .561224 respectively, over 14 parents. Hidden rejection is shared at 97 / 111. Full one-/multi-child validation breakdowns remain in JSON, not pooled with test.

## Cost and limits

One-time CPU encoding took 6.18 / 3.57 / 3.38 seconds for support / validation / test (13.13 seconds total), with zero truncated utterances. Encoder-process high-water was about 1,044.43 MiB, including source preparation and model/runtime. Two encoder-free workers took 1.43 / 1.71 ms to fit, .28 / .32 ms for test prediction, 4.84 / 5.85 ms for validation calibration, and 2.54 / 2.69 ms for scoring. Tiny operations and two runs are not a throughput benchmark.

Worker input-loaded RSS was 32.72 / 33.77 MiB; end RSS 36.57 / 36.75 MiB; lifetime `ru_maxrss` 36.50 / 36.61 MiB. Kernel counter sources can differ slightly. These counters were not continuously sampled and high-water is not reset at baseline: **no exact phase-peak or isolated algorithm-memory claim**. Do not compare these values directly with the calendar sampler or interpret encoder memory as parent-router memory.

The preflight caps/hidden-leaf construction change natural label frequencies. Results concern supported-parent routing, not discovering parents, discovering taxonomy, maintaining temporal identity, or downstream recommendation value. The high known false-novelty cost should motivate a separately preregistered application-relevant validation constraint, not retuning this test. No broad survey ranking, original-42-run count, or existing benchmark data is altered. Raw utterances, IDs, embeddings, individual predictions and private digests stay outside Git.
