# Lexical-v1: executable sanity baseline

This is a **new, isolated transductive batch-clustering experiment**, not a reproduction of a paper, a semantic encoder benchmark or a comparison with historical repository numbers. It tests the public official **test corpora in full**, with no subset selection. No hyperparameter search or validation-based selection was performed. All configurations below were fixed before scores were inspected.

## Reproduce

Use CPython **3.12.14** and a dedicated virtual environment. Do not install this alongside the project's core scikit-learn 1.6.1 environment. This milestone records the available scikit-learn **1.8.0** runtime explicitly. The main project's dependencies are unchanged.

```bash
python3.12 -m venv /tmp/lexical-v1-env
/tmp/lexical-v1-env/bin/pip install -r scripts/requirements-lexical-v1.txt
/tmp/lexical-v1-env/bin/python scripts/prepare_datasets.py banking77 --cache /tmp/lexical-data/banking77 --download --report /tmp/lexical-data/banking-audit.json
/tmp/lexical-v1-env/bin/python scripts/prepare_datasets.py massive_zh_cn --cache /tmp/lexical-data/massive_zh_cn --download --report /tmp/lexical-data/massive-audit.json
/tmp/lexical-v1-env/bin/python scripts/benchmark_lexical.py --cache /tmp/lexical-data --output /tmp/lexical-data/results.json
```

Downloads are pinned and SHA-256/size-checked using existing manifests. After caching, execution is offline. No LLM/API inference, model downloads or external text transmission occurs. Raw data and assignments remain outside the repository; the checked-in JSON contains aggregate measurements only. Data attribution: BANKING77, PolyAI (CC BY 4.0); MASSIVE, Amazon Science (CC BY 4.0), exact versions/notices linked in `datasets/*/source.json`.

## Frozen protocol

- Representation fitted **on each evaluation corpus itself**: character TF-IDF 2–5 grams, lowercase, min_df=2, max_features=10,000, sublinear_tf=True; randomized SVD 64 dimensions, n_iter=7; L2 row normalization. No semantic encoder or language-specific tokenization.
- Seeds 0/1/2 govern SVD and KMeans. HDBSCAN is deterministic given a representation, but the representation changes across seeds. One CPU thread through threadpoolctl, HDBSCAN n_jobs=1.
- Oracle-K: KMeans, K = number of gold labels **present in the tested corpus**, n_init=10, max_iter=300. It intentionally receives privileged label information; gold labels are otherwise only for evaluation.
- Unknown-K: sklearn HDBSCAN, min_cluster_size=10, min_samples=5, Euclidean, EOM. These are fixed choices, not claimed universal defaults or tuned optima. Labels/K never passed to the algorithm.
- Sanity control: assign every document to one cluster. It has 100% coverage but ARI/NMI=0 on these nontrivial corpora, illustrating why coverage alone cannot rank systems.
- `official-test` preserves the official split unchanged. Training/dev texts are **not** used to fit features.
- `duplicate-excluded` is a separate sensitivity corpus: normalize with NFKC+casefold+whitespace; exclude every test group with more than one member, and every test text present in train or validation. Remove **all** members, not a label-selected representative. No label-based deletions; lost labels/counts are reported. Refit representation for that corpus; therefore score changes combine corpus and representation effects and do not measure causal leakage inflation.
- ARI/NMI on all rows treat noise -1 as one cluster (explicitly named in JSON), complemented by noise-excluding coverage, cluster count and covered-only ARI. Covered-only ARI is conditional on a method's selected subset and **not a fair standalone leaderboard**. These corpora lack native OOS; density noise is not evidence of OOS recognition.
- Raw per-seed results are authoritative; summary uses arithmetic mean and min–max over 3 seeds, **not confidence intervals**. Wall-clock durations are descriptive on this executor, not portable latency benchmarks or RAM measurements.

## Scope and next gate

A lexical baseline verifies protocol wiring and establishes a transparent low-cost floor. Chinese short strings and English banking language have different lexical redundancy; cross-dataset absolute scores are not a quality ranking. A poor fixed HDBSCAN configuration does not refute density clustering. No conclusions about LLM-assisted approaches, routing, hierarchical navigation, evolving streams or downstream recommendation are justified here.

Next: separately pinned multilingual semantic embeddings with model revision, pooling/prefixes, normalization and license; compare the same corpus/feature matrix, report noise and reassignment policy, then add validation-only parameter selection and cost-aware methods. Do not tune on these reported test results.

Keep environments separated by track: the repository's PARC extra pins NumPy 1.26.4/SciPy 1.13.1, while River 0.26.1's streaming requirements need newer versions. An “install all extras” quickstart is not supported. This lexical environment neither installs nor validates PARC/River.

## Executed results

Two complete executions produced identical score/coverage/count fields in all 36 rows (timings excluded). Original result files were not changed.

| Dataset / track / method | ARI mean [min, max] | NMI mean | Coverage mean | Clusters [min,max] |
|---|---|---|---|---|
| banking77 / official-test / one-cluster | 0.0000 [0.0000, 0.0000] | 0.0000 | 100.0% | [1, 1] |
| banking77 / official-test / kmeans-oracle-k | 0.2934 [0.2907, 0.2977] | 0.6406 | 100.0% | [77, 77] |
| banking77 / official-test / hdbscan-fixed | 0.0155 [0.0153, 0.0156] | 0.4357 | 42.0% | [43, 44] |
| banking77 / duplicate-excluded / one-cluster | 0.0000 [0.0000, 0.0000] | 0.0000 | 100.0% | [1, 1] |
| banking77 / duplicate-excluded / kmeans-oracle-k | 0.2927 [0.2907, 0.2960] | 0.6423 | 100.0% | [77, 77] |
| banking77 / duplicate-excluded / hdbscan-fixed | 0.0154 [0.0152, 0.0156] | 0.4301 | 42.0% | [42, 43] |
| massive_zh_cn / official-test / one-cluster | 0.0000 [0.0000, 0.0000] | 0.0000 | 100.0% | [1, 1] |
| massive_zh_cn / official-test / kmeans-oracle-k | 0.1790 [0.1773, 0.1824] | 0.5105 | 100.0% | [59, 59] |
| massive_zh_cn / official-test / hdbscan-fixed | 0.0387 [0.0371, 0.0410] | 0.4664 | 62.9% | [78, 79] |
| massive_zh_cn / duplicate-excluded / one-cluster | 0.0000 [0.0000, 0.0000] | 0.0000 | 100.0% | [1, 1] |
| massive_zh_cn / duplicate-excluded / kmeans-oracle-k | 0.1777 [0.1752, 0.1796] | 0.5054 | 100.0% | [59, 59] |
| massive_zh_cn / duplicate-excluded / hdbscan-fixed | 0.0393 [0.0368, 0.0406] | 0.4468 | 63.1% | [66, 68] |

[Aggregate JSON with per-seed scores and timings](results.json). Duplicate-excluded counts: BANKING77 3,071 (9 removed); MASSIVE zh-CN 2,696 (278 removed); no original test labels lost.

Under this representation and fixed configuration, HDBSCAN leaves many examples unclustered. Oracle-K KMeans is a useful lexical control, not evidence of superiority under equal information. Dedup changes these particular aggregate ARI means only slightly, but that is not a general claim that duplicates are harmless.
