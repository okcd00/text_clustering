# Unified static v1

The first reproducible rerun of the **five batch families already displayed**: KMeans, MiniBatchKMeans, HDBSCAN, BERTopic clustering core, EVoC. This is a fresh comparison on pinned public data and a pinned encoder, not a numerical reproduction of the old undistributed encoder.

## One round

On Linux, after the one-time setup below:

```bash
SUITE_PYTHON=/tmp/unified-env/bin/python DATA_CACHE=/tmp/semantic-data \
RUN_DIR=/tmp/comparison-round bash scripts/evaluate_static.sh
```

This runs both BANKING77 and MASSIVE zh-CN, the same 512 test rows and same 384-dimensional E5 vectors per dataset, all five methods, seeds 0/1/2, fresh worker process per fit. Output contains 30 runs in four comparison cohorts. Downloading, model inference, aggregation/publication, or unknown-K tuning never happens implicitly. A failed worker is reported without a score and makes the suite `partial`; never rank error rows.

The model cache is not an undocumented precondition. Prepare it explicitly once:

1. Use CPython 3.12 and separate environments. Install the pinned [semantic environment](../semantic-pilot-v1/environment-lock.txt) following its CPU Torch instructions. Install this suite with `python -m venv /tmp/unified-env && /tmp/unified-env/bin/pip install -r experiments/unified-static-v1/environment-lock.txt`. Do not install every optional project extra together.
2. Using the semantic Python, run `scripts/prepare_datasets.py banking77 --cache /tmp/semantic-data/banking77 --report /tmp/banking-audit.json --download`, then the equivalent command for `massive_zh_cn`. Source manifests fix URLs, revisions, SHA-256, sizes and licenses. Data remain outside Git.
3. Using the semantic Python, run `scripts/prepare_semantic_model.py --model-dir /tmp/semantic-e5-model --download`. Only pinned inert safetensors/tokenizer artifacts are accepted.
4. Run the one-round command above with `--prepare-vectors`, and additionally set `SEMANTIC_PYTHON` and `MODEL_DIR`. This calls the semantic-pilot runner locally and writes `.npy` plus provenance sidecars before the suite. Later rounds reuse the vectors and check model revision, ordered-text digest and file hash. No external inference API. No expansion beyond the fixed pilot rows.

The setup requires public dataset/model downloads and official PyPI packages. Scikit-learn is BSD-3-Clause; UMAP/HDBSCAN are BSD-licensed and EVoC is BSD-2-Clause (installed distribution metadata); upstream dependency/data/model terms still apply. No paper PDF or raw dataset is redistributed. Pinned packages are for reproduction, not a promise they remain the best security choice indefinitely.

## Comparability contract

- **Known K**: KMeans and MiniBatchKMeans receive the number of distinct gold labels present in the sampled test set. Both use 10 initializations and seed 0/1/2. This privileged input is not shared with unknown-K methods.
- **Unknown K, fixed**: the existing adapters receive neither labels nor K. Direct sklearn HDBSCAN uses min_cluster_size=15, min_samples=5; BERTopic core uses UMAP 5D then external HDBSCAN min_cluster_size=15, min_samples=4 (external count excludes self). EVoC selects its default upstream layer with approximate/base K unset. Presets are unchanged by test scores. There is no claim of optimal tuning.
- **Pipeline rather than isolated clusterer comparison**: BERTopic core performs its own UMAP reduction; runtime includes it. Full BERTopic topic words, descriptions and reassignment are excluded. EVoC is the upstream library pipeline, not this repository's reimplementation.
- All methods use the exact same input matrix in a dataset cohort. Original rows, model version, vectors, Python/packages and seeds are identified. No historical scores are copied into these cohorts.
- ARI and arithmetic NMI on all rows retain noise as one label; covered-only ARI must be read with coverage. Coverage denominator is all 512 rows. A single returned cluster is not automatically meaningful.
- Timings include the method's reduction and JIT if incurred, and algorithm lazy imports, but exclude worker bootstrap imports, embedding/data loading and scoring. Peak RSS is the full worker high-water mark, not incremental algorithm allocation. Timings are descriptive cold/mixed-cache observations, not steady-state speed rankings. Seed spread is not a confidence interval over unseen datasets.

## Inventory beyond this first tranche

| Historical family | Decision | Exact gate |
|---|---|---|
| KMeans, MiniBatchKMeans, HDBSCAN, BERTopic core, EVoC | Rerun here | Same E5 rows, immutable inputs, explicit K regime |
| Centroid / multi-prototype / NLI routing | Separate routing protocol | Needs train support, hidden-label splits, validation gates and NLI model/cost budget; batch clustering cannot establish routing quality |
| Hierarchical centroid / NLI navigation | Deferred | Needs auditable hierarchy edges and support/query information; intent flat labels alone are insufficient |
| DBSTREAM / MiniBatch / textClust / BIRCH frozen-versus-updating | Separate future streaming round | Must evaluate before each update with identical order/window policy, not fit on all test data |
| Calendar BIRCH and maintenance strategies | Deferred | True timestamps, capacity budget and label meaning must be fixed; BANKING/MASSIVE row positions are not time |

These independent task families remain archived rather than silently rebranded as comparable. Optional paper backends such as TabClustPFN retain their own license and task applicability gates.

### Compatibility discovery

The first setup with scikit-learn 1.8.0 failed inside external HDBSCAN 0.8.40: `check_array(..., force_all_finite=...)` is no longer accepted. The suite therefore pins **scikit-learn 1.6.1 for all five algorithms**, matching the old adapters, rather than monkey-patching upstream code or mixing backend versions. Failed setup attempts are not quality measurements; all reported comparison rows are rerun in the final unified environment. The separate semantic pilot uses 1.8.0 and is not merged into this version's comparison cohorts.
