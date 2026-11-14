# Semantic pilot v1: measured paired CPU results

Run date: 2026-11-14. **Local execution, not paper-reported scores.** [Frozen protocol and reproduction](../../experiments/semantic-pilot-v1/README.md) · [Aggregate run 1](run1.json) · [Aggregate run 2](run2.json). No examples, model artifacts or vector arrays are published.

## Scope and main observation

On the same fixed 512-row test subsets, E5 improves oracle-K KMeans ARI over the lexical pipeline on both datasets. This does **not** establish a consistent benefit for unknown-K density clustering: fixed MASSIVE ARI improves but coverage falls; validation-tuned E5 has lower ARI than tuned lexical on both. Sparse classes, density geometry, model, dimension and dtype all remain possible contributors. These are complete-pipeline comparisons, not controlled causal claims about the encoder alone.

BANKING77 test has 77 labels, one singleton; MASSIVE zh-CN test has 57 labels, three singletons and two normalized duplicate excess rows. Each has 512 validation rows, with 76 / 56 labels. Validation texts shared with any official test text were removed before label-blind sampling. No test-based resampling or tuning occurred.

## Same-row results

Means across seeds 0/1/2. E5 encoding and HDBSCAN are deterministic here; their repeated rows are **not independent experiments**. Seed variation in lexical includes SVD; KMeans also varies its initialization. No confidence intervals or significance claims.

| Dataset | Representation | Method / information | All-row ARI | Coverage | Covered-only ARI | Cluster counts by seed |
|---|---|---|---:|---:|---:|---|
| banking77 | lexical | kmeans-oracle-k | 0.2934 | 100.0% | 0.2934 | 77/77/77 |
| banking77 | lexical | hdbscan-fixed | 0.0184 | 43.3% | 0.1748 | 7/7/7 |
| banking77 | lexical | hdbscan-validation-tuned | 0.0335 | 56.2% | 0.2833 | 25/27/27 |
| banking77 | e5 | kmeans-oracle-k | 0.3458 | 100.0% | 0.3458 | 77/77/77 |
| banking77 | e5 | hdbscan-fixed | 0.0086 | 28.7% | 0.1103 | 4/4/4 |
| banking77 | e5 | hdbscan-validation-tuned | 0.0138 | 64.3% | 0.0110 | 7/7/7 |
| massive_zh_cn | lexical | kmeans-oracle-k | 0.1482 | 100.0% | 0.1482 | 57/57/57 |
| massive_zh_cn | lexical | hdbscan-fixed | 0.0146 | 79.3% | 0.0951 | 16/2/2 |
| massive_zh_cn | lexical | hdbscan-validation-tuned | 0.0768 | 75.7% | 0.2008 | 48/20/43 |
| massive_zh_cn | e5 | kmeans-oracle-k | 0.3131 | 100.0% | 0.3131 | 57/57/57 |
| massive_zh_cn | e5 | hdbscan-fixed | 0.0398 | 52.0% | 0.0501 | 4/4/4 |
| massive_zh_cn | e5 | hdbscan-validation-tuned | 0.0393 | 53.1% | 0.0538 | 5/5/5 |

All-row ARI/NMI treat all `-1` noise as one cluster. Covered-only ARI is conditional on accepted rows; the full JSON includes exact `covered_n` denominators, coverage, NMI and every seed. One-cluster controls give 100% coverage and zero ARI/NMI, illustrating why coverage alone is not quality. Oracle K is 77 / 57, privileged test-label information; do not rank that arm against unknown-K as an equal-information contest.

## Validation selection and negative evidence

Each dataset/representation/seed evaluates six validation parameter pairs (min_cluster_size 5/10/20, min_samples 3/5), maximizing all-row validation ARI with lexicographic ties. The selected pairs are frozen before test. This is **labeled-validation tuning**, not fully unsupervised selection. Detailed trials and times are in JSON. E5 selects (5,3) on BANKING77 and (5,5) on MASSIVE, yielding only 7 / 5 test clusters. The protocol is not changed to rescue these test outcomes.

MASSIVE lexical fixed clustering is unstable across SVD seeds (16 / 2 / 2 clusters); a mean alone conceals this. BANKING E5 validation-tuned coverage is higher than lexical (64.3% vs 56.2%) while all-row ARI is lower (0.0138 vs 0.0335). On MASSIVE, fixed E5 all-row ARI rises (0.0398 vs 0.0146) but coverage falls (52.0% vs 79.3%). Different denominators and failure modes must remain visible.

## Measured cost

One CPU thread, float32 E5 / float64 lexical, Intel Xeon Platinum 8370C. Shared cloud timing is descriptive, not a dedicated hardware benchmark. In run 1, encoding 512 test texts took **8.98 s BANKING77 / 5.18 s MASSIVE zh-CN**. Lexical TF-IDF+SVD took about **0.23 / 0.05 s** per test representation (three-seed mean). Model load was **1.26 s**, measured separately. No input exceeded the 512-token truncation limit.

E5 oracle-K clustering took 0.560 s / 0.486 s for BANKING77 / MASSIVE, excluding encoding. Validation computation and model loading are additional costs; per-trial times are preserved. Downloads and dependency installation are excluded. Whole-process high-water RSS was **1,150.7 MiB** in run 1; this includes weights, tokenizer, artifact hashing and both datasets, **not a per-method memory comparison**.

## Interpretation boundaries

- Fixed pretrained E5 384D and lexical SVD 64D differ in information, dimension and dtype. No fine-tuning was done. Pretraining contamination cannot be ruled out by split deduplication.
- Both validation and test lexical vocabularies/SVD are fit separately, transductively, on their own unlabeled batches; no validation-fitted transform is claimed. E5 is frozen across roles.
- One fixed sample per dataset is not a resampling study. BANKING and MASSIVE differ in task and language; cross-dataset score gaps cannot isolate language effects.
- Results do not establish streaming adaptation, unseen-intent rejection, hierarchy maintenance or recommendation business value. Historical full-test lexical-v1 remains a separate result, even where a rounded number coincides.
- Model/data download hashes, installed versions and script hashes are preserved. External cache files contain vectors and private alignment hashes, but are intentionally not committed.

## Verification

Two complete local executions produced identical sample summaries, selected parameters, cluster counts and every non-timing metric across 48 test result rows per run. Run 2 additionally saved external vector caches; script hashes record this non-numerical addition. Timings and whole-process RSS vary. [Machine-readable repeat check](reproducibility.json).
