# Semantic pilot v1 — prepared, NOT executed

**Status:** data preparation and dependency preflight executed; no encoder, embeddings, clustering scores or semantic/lexical comparison executed. This is a bounded next experiment, not a new leaderboard. Existing lexical-v1 and historical results remain untouched.

## Verified blocker

The current isolated executor lacks **torch, transformers, tokenizers and safetensors**; it also has no sentence-transformers or ONNX runtime. With existing model/runtime packages only, local embedding inference is unavailable. No model runtime was installed, no weights were downloaded, no remote model code was executed, and no dataset text was sent to an inference API. Installing packages is not silently treated as an executed experiment.

`preflight.json` records the actual module presence and aggregate prepared sample counts. Module availability alone would not prove that compatible versions, cached model files, memory or inference work. Before running, use an existing trusted CPU model environment and record exact dependency versions; dependency resolution and model loading remain unverified here.

## Run the preparation already verified

Use the lexical-v1 environment and previously hash-checked external data cache:

```bash
python scripts/prepare_semantic_pilot.py --cache /tmp/lexical-data --report /tmp/lexical-data/semantic-preflight.json
```

The command is **not** a semantic benchmark runner. It downloads nothing and writes no examples, row IDs or per-text hashes to the report. Reuse `scripts/prepare_datasets.py` for initial pinned source preparation. The fixed 512-row cap applies per dataset and role, totaling at most 2,048 texts before caching identical strings. These pilot samples must not be compared directly with lexical-v1's full-test numbers.

## Frozen representation contract

Candidate: [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3), MIT license, revision **614241f622f53c4eeff9890bdc4f31cfecc418b3** (official repository API checked). This is a reproducible multilingual control, not a claim of current state of the art.

Per its [pinned model card](https://huggingface.co/intfloat/multilingual-e5-small/blob/614241f622f53c4eeff9890bdc4f31cfecc418b3/README.md): prepend `query: ` for clustering in both languages; use attention-mask mean pooling and L2 normalization; max length 512 tokens, truncation enabled. Keep the source text unchanged before that prefix: no lowercasing, NFKC or whitespace rewriting in the semantic input. NFKC+casefold is for duplicate auditing only. Output is 384 dimensions; CPU float32, eval mode, inference mode, batch size 16, one thread, no mixed precision. Record truncation counts and validate finite vectors and unit norms.

Load only standard Transformers architecture with `trust_remote_code=False`; require `use_safetensors=True`, pin tokenizer and model to the same revision, and after approved cache preparation load with `local_files_only=True`. Do not fall back to pickled weights, remote code or API inference. Record downloaded artifact SHA-256 values and installed package versions before results can be marked reproducible. **Weights and tokenizer artifacts were not downloaded or validated in this milestone.**

## Sample and validation contract

- Sampling does not read labels. Sort original row positions by SHA-256 of the fixed namespace, dataset, source-role and position; select the first 512, exactly as the committed preparation function specifies.
- Test comes from the official test corpus. MASSIVE validation comes from official validation; BANKING validation is a fixed sample of train. Remove validation-source rows whose normalized text appears anywhere in official test, before selecting validation. No samples are moved between roles.
- Preserve test duplicates in the pilot. Report singleton classes, class counts and duplicate excess. No “every class must appear” resampling; small random subsets cannot support robust conclusions about rare intents.
- Gold test labels are only used for evaluation and the explicitly oracle-K arm. Do not use test scores to choose hyperparameters, representation or model revision. E5 was selected as a low-cost multilingual control, not by observed test scores.

## Planned comparisons (none run)

1. Re-run the lexical representation on **these exact same sampled rows**, rather than import full-test scores. Semantic uses 384 normalized features; lexical uses TF-IDF/SVD64/L2. This compares two pipelines, not a dimension-controlled causal encoder ablation.
2. Oracle-K KMeans: test-present K, seeds 0/1/2, n_init=10, max_iter=300. Report the privileged K budget.
3. Unknown-K fixed HDBSCAN: min_cluster_size=10, min_samples=5, Euclidean/EOM, no label access, no noise reassignment. This is the direct fixed-configuration control.
4. Separate **labeled-validation-tuned, unknown-test-K** arm: grid min_cluster_size={5,10,20}, min_samples={3,5}; select highest validation ARI (noise as one cluster), ties by lexicographic parameter order. Select separately per dataset/representation; freeze before test. Never call this label-free tuning. Compare fixed and tuned arms without hiding the six validation trials.
5. Keep one-cluster control, ARI/NMI on all rows with noise convention explicit, covered-only ARI, coverage, cluster count, elapsed encoding/clustering time and peak memory if measurable. Report per-seed rows rather than only a best score.

This pilot's gate is: successful local encoding, exact sample alignment, sensible finite features, intact metadata and honest information budgets. It cannot establish universal algorithm rankings, statistically reliable rare-intent performance, streaming behavior or downstream recommendation value. Only after this gate should the full official corpora be encoded and results reviewed.
