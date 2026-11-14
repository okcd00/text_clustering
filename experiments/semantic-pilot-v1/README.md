# Semantic pilot v1 — execution protocol

**Status:** now executed after the owner approved installation of an isolated CPU environment. [Measured results](../../results/semantic-pilot-v1/README.md) are separate from historical full-test lexical-v1. The original `preflight.json` is retained as the earlier preparation-only snapshot, not current runtime status.

## Isolated installation and reproduction

Tested CPython 3.12.14 on Linux x86-64. Do not install all project extras: `parc` and `streaming` have incompatible NumPy/SciPy constraints. This standalone pilot intentionally uses the lexical-v1 versions, not the package's scikit-learn 1.6.1 pin. No dependency in the shared interpreter was changed.

```bash
python -m venv /tmp/semantic-pilot-env
/tmp/semantic-pilot-env/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
/tmp/semantic-pilot-env/bin/pip install -r experiments/semantic-pilot-v1/environment-lock.txt
/tmp/semantic-pilot-env/bin/pip check
/tmp/semantic-pilot-env/bin/python scripts/prepare_datasets.py banking77 --cache /tmp/semantic-data/banking77 --report /tmp/semantic-data/banking-audit.json --download
/tmp/semantic-pilot-env/bin/python scripts/prepare_datasets.py massive_zh_cn --cache /tmp/semantic-data/massive_zh_cn --report /tmp/semantic-data/massive-audit.json --download
/tmp/semantic-pilot-env/bin/python scripts/prepare_semantic_model.py --model-dir /tmp/semantic-e5-model --download
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /tmp/semantic-pilot-env/bin/python scripts/benchmark_semantic_pilot.py --cache /tmp/semantic-data --model-dir /tmp/semantic-e5-model --output /tmp/semantic-data/pilot.json
```

Downloads contain only version-pinned public artifacts and do not send dataset texts. The runner sets Hugging Face offline flags, requires local files and safetensors, verifies the committed model-artifact manifest, and never falls back to remote code or an inference API. Raw texts, model weights, vectors, sample IDs and predictions stay outside Git. Only aggregate measurements are published. Recreate the cache if the cloud environment is replaced; `/tmp` is not a durability guarantee. The full dependency freeze records what was installed; it does not imply cross-platform binary equivalence.

## Run the preparation already verified

Use the lexical-v1 environment and previously hash-checked external data cache:

```bash
python scripts/prepare_semantic_pilot.py --cache /tmp/lexical-data --report /tmp/lexical-data/semantic-preflight.json
```

The command is **not** a semantic benchmark runner. It downloads nothing and writes no examples, row IDs or per-text hashes to the report. Reuse `scripts/prepare_datasets.py` for initial pinned source preparation. The fixed 512-row cap applies per dataset and role, totaling at most 2,048 texts before caching identical strings. These pilot samples must not be compared directly with lexical-v1's full-test numbers.

## Frozen representation contract

Candidate: [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3), MIT license, revision **614241f622f53c4eeff9890bdc4f31cfecc418b3** (official repository API checked). This is a reproducible multilingual control, not a claim of current state of the art.

Per its [pinned model card](https://huggingface.co/intfloat/multilingual-e5-small/blob/614241f622f53c4eeff9890bdc4f31cfecc418b3/README.md): prepend `query: ` for clustering in both languages; use attention-mask mean pooling and L2 normalization; max length 512 tokens, truncation enabled. Keep the source text unchanged before that prefix: no lowercasing, NFKC or whitespace rewriting in the semantic input. NFKC+casefold is for duplicate auditing only. Output is 384 dimensions; CPU float32, eval mode, inference mode, batch size 16, one thread, no mixed precision. Record truncation counts and validate finite vectors and unit norms.

Load only standard Transformers architecture with `trust_remote_code=False`; require `use_safetensors=True`, pin tokenizer and model to the same revision, and after approved cache preparation load with `local_files_only=True`. Do not fall back to pickled weights, remote code or API inference. Record downloaded artifact SHA-256 values and installed package versions before results can be marked reproducible. The executed milestone records all six artifact hashes in `model-artifacts.json`.

## Sample and validation contract

- Sampling does not read labels. Sort original row positions by SHA-256 of the fixed namespace, dataset, source-role and position; select the first 512, exactly as the committed preparation function specifies.
- Test comes from the official test corpus. MASSIVE validation comes from official validation; BANKING validation is a fixed sample of train. Remove validation-source rows whose normalized text appears anywhere in official test, before selecting validation. No samples are moved between roles.
- Preserve test duplicates in the pilot. Report singleton classes, class counts and duplicate excess. No “every class must appear” resampling; small random subsets cannot support robust conclusions about rare intents.
- Gold test labels are only used for evaluation and the explicitly oracle-K arm. Do not use test scores to choose hyperparameters, representation or model revision. E5 was selected as a low-cost multilingual control, not by observed test scores.

## Frozen comparisons (executed without test-based changes)

1. Re-run the lexical representation on **these exact same sampled rows**, rather than import full-test scores. Semantic uses 384 normalized features; lexical uses TF-IDF/SVD64/L2. This compares two pipelines, not a dimension-controlled causal encoder ablation.
2. Oracle-K KMeans: test-present K, seeds 0/1/2, n_init=10, max_iter=300. Report the privileged K budget.
3. Unknown-K fixed HDBSCAN: min_cluster_size=10, min_samples=5, Euclidean/EOM, no label access, no noise reassignment. This is the direct fixed-configuration control.
4. Separate **labeled-validation-tuned, unknown-test-K** arm: grid min_cluster_size={5,10,20}, min_samples={3,5}; select highest validation ARI (noise as one cluster), ties by lexicographic parameter order. Select separately per dataset/representation; freeze before test. Never call this label-free tuning. Compare fixed and tuned arms without hiding the six validation trials.
5. Keep one-cluster control, ARI/NMI on all rows with noise convention explicit, covered-only ARI, coverage, cluster count, elapsed encoding/clustering time and peak memory if measurable. Report per-seed rows rather than only a best score.

This pilot's gate is: successful local encoding, exact sample alignment, sensible finite features, intact metadata and honest information budgets. It cannot establish universal algorithm rankings, statistically reliable rare-intent performance, streaming behavior or downstream recommendation value. Only after this gate should the full official corpora be encoded and results reviewed.
