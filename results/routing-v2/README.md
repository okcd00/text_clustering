# Routing v2: supervised known-class routing with abstention

Executed 2026-11-14. This is a small **routing** experiment, not a static clustering score or a reproduction of the historical WOS encoder. [Aggregate results](results.json) retain all 12 dataset × method × SVD-seed rows and every validation trial. No example text, row IDs, embeddings or predictions are published.

## One-command replay

From an environment satisfying `scripts/requirements-lexical-v1.txt`, with pinned public files already prepared by `scripts/prepare_datasets.py`:

```bash
python scripts/benchmark_routing_v2.py --cache /tmp/semantic-data --output /tmp/routing-v2-results.json
```

No encoder weights or inference APIs are used. The script verifies pinned artifact hashes before evaluating. Dependencies are the same isolated lexical stack, not all optional project extras. Results include actual Python/package versions and script hash. Dataset/model caches stay outside the checkout.

## Frozen protocol

- Split training label vocabulary by deterministic SHA-256 order into approximately 60% known, 20% validation-hidden, 20% test-hidden. Hidden labels do not receive support. Label sets are disjoint; this is a deliberately constructed open-set task, not native dataset OOS data or natural temporal arrival.
- MASSIVE uses official train/dev/test. BANKING divides official train into a deterministic 70/30 support/calibration source; official test remains test. Remove normalized cross-role duplicates: support excludes full validation-source and full test texts; validation excludes full test texts. Normalization is NFKC + casefold + whitespace collapse, only for duplicate auditing. Dataset text passed to features remains unchanged. Within-role duplicates remain.
- Up to 20 labeled support examples per known class. Validation and test have independent deterministic, **label-blind row selection** after their role-label filtering, capped at 512 each. BANKING support has 920 rows/46 classes, MASSIVE 698 rows/36 classes. Test denominators are respectively 383 known + 129 hidden and 413 known + 99 hidden. Single fixed split, not multiple split replications.
- Fit character TF-IDF (2–5 grams, `min_df=2`, maximum 10,000 features, sublinear TF), SVD64 and L2 **only on support**. Transform validation and test; never refit their vocabulary/SVD. Three SVD seeds 0/1/2; zero-vector queries are always rejected.
- Prototype uses normalized summed support embeddings per class; class-max-1NN uses maximum support cosine per class, then compares the two best **distinct classes**. This is a controlled nearest-neighbor method, not majority-vote kNN. A regression test checks prototype decisions against the repository's existing `OnePass` adapter on synthetic non-tied vectors.
- Each method/seed independently selects cosine threshold in 0.0–0.9 step 0.1 and class-margin threshold in {0, .05, .1}, only on labeled validation-known/validation-hidden samples. Objective: mean of known correct-acceptance and hidden rejection. Ties choose ascending cosine, then margin. No test tuning. This explicitly uses validation labels; it is not fully unsupervised.
- Report both acceptance coverage and accepted accuracy for known classes, correct accepted / all known, and accepted / all hidden. `null` accepted accuracy means nothing accepted, not zero accuracy. The fixed split's class support and example imbalance are part of the experiment, not corrected by downstream metric selection.

## Observed comparison

Means over three SVD seeds (not confidence intervals or independent dataset replications):

| Dataset | Method | Known coverage | Accepted-known accuracy | Correct / all known | Hidden false accept |
|---|---|---:|---:|---:|---:|
| BANKING77 | Prototype | .447 | .942 | .420 | .114 |
| BANKING77 | Class-max 1NN | .415 | .933 | .387 | .124 |
| MASSIVE zh-CN | Prototype | .401 | .795 | .319 | .192 |
| MASSIVE zh-CN | Class-max 1NN | .277 | .815 | .224 | .199 |

In this fixed lexical cohort, the prototype gate accepts more correct known examples with fewer hidden false accepts. MASSIVE 1NN's higher accepted accuracy comes with much lower coverage, not a blanket quality improvement. Neither method's low known coverage supports deployment or general claims about semantic representations; the test-hidden class mix differs from validation-hidden, and no real event stream, new-class aggregation, NLI verification or downstream recommendation outcome is measured.

Timing is local wall-clock with BLAS thread limit 1: feature fitting/transformation is shared and recorded separately; validation includes all 30 trials; test routing includes prototype/support similarity construction and gating, not encoding or downloads. Memory is not measured. Millisecond timings are noisy and not an infrastructure benchmark. Do not compare them with archived WOS costs or static-batch E5 costs.
