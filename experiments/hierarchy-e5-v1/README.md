# MASSIVE zh-CN supported-parent E5 pilot v1

This is a separate supervised-topology routing experiment, not static clustering, a calendar replay, or an addition to the original 42 runs. [Protocol](protocol.json) was committed as `c17cd8c` before any model scores. [Measured results](../../results/hierarchy-e5-v1/README.md).

## Frozen inputs and decision rules

The complete committed hierarchy preflight, including source hashes, revision, split fingerprint, caps and script hash, must match before encoding. Support has 606 utterances from 31 known leaves / 18 parents; validation has 223 known and 111 hidden queries; test has 230 known and 104 hidden queries. Hidden leaves are disjoint across validation/test. Deduplication, row order and selection are exactly the preflight's, not new model-dependent choices.

Use the pinned E5 model/artifact hashes and CPU environment lock. Encode **utterances only**, with `query: ` prefix, masked mean pooling, L2 normalization, batch 16, maximum 512 tokens, float32, inference mode and one Torch/BLAS thread. Support prototypes normalize each leaf's mean embedding. No node-name/descriptive embedding, original semantic labels or hidden nodes are supplied to either method.

Leaf prediction maximizes cosine over the sorted opaque leaf prototypes; exact ties choose the lexicographically smallest opaque ID. Validation sets one rejection threshold using binary known/hidden labels only. Reject iff maximum leaf similarity is below the threshold. Candidate thresholds and objective/ties are fully specified in protocol.json; integer arithmetic compares the balanced known-accept/hidden-reject objective. Parent gold is not used in calibration, and no test metric changes this threshold.

Two parent arms share every leaf prediction, score and rejection mask:

- **Flat max-leaf → parent:** map the winning known leaf to its supplied parent.
- **Equal-leaf mean parent:** average the normalized known leaf prototypes equally per parent, normalize again, and maximize cosine over supported parents. This is not weighting by support-row frequency, and not restricting/reranking candidates using a query's winning leaves.

Test gold is loaded into the worker's scoring stage only after all predictions and masks have been frozen into a private digest. Both methods fit only support; their signatures accept no scorer. File/hash validation may read scorer bytes without parsing them, and this Python boundary is not a security sandbox. No original labels, query IDs, individual predictions, embeddings or private digests are published.

## Reproduce separately

Create an isolated environment with `environment-lock.txt` (Torch CPU wheels use the official PyTorch CPU index). Set `$PYTHON`, `$DATA`, `$MODEL` and `$CACHE` to external private workspace locations, not paths inside this repository.

```bash
"$PYTHON" scripts/prepare_datasets.py massive_zh_cn --cache "$DATA" --download --report "$CACHE-source.json"
"$PYTHON" scripts/prepare_semantic_model.py --model-dir "$MODEL" --download
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON" scripts/prepare_hierarchy_e5.py \
  --cache "$DATA" --model-dir "$MODEL" --output "$CACHE"
for repeat in 1 2; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PYTHON" scripts/benchmark_hierarchy_e5.py \
    --cache "$CACHE" --output "$CACHE/run-$repeat.json"
done
"$PYTHON" scripts/summarize_hierarchy_e5.py --cache "$CACHE" --output results/hierarchy-e5-v1/results.json
"$PYTHON" -m pytest -q tests/test_hierarchy_e5.py tests/test_hierarchy_preflight.py
```

Exporter checks exact repeat agreement excluding resource/time fields, prediction/mask digests, source/split/hash provenance, subgroup denominators and shared rejection denominators. It exports aggregates only. Two deterministic repeats are not statistical confidence intervals.

Encoder preparation has its own process high-water counter. Each source-free, encoder-free worker records input-loaded RSS, end RSS and lifetime `ru_maxrss`, plus fit/predict/calibration/scoring times. These are coarse process observations, not sampled phase maxima, reset-baseline peaks, isolated model memory or production throughput. No cross-backend memory ranking is implied.
