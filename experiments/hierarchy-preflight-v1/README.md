# Supported-parent hierarchy preflight v1

**Data/interface preflight only. No model, clustering, routing or performance evaluation was run.** This is isolated from routing-v2 and calendar results.

## Frozen construction

Use pinned official MASSIVE zh-CN revision and all three verified Parquet hashes in `preflight.json`. Preserve official train/validation/test membership. Remove train texts occurring in either full later split, validation texts occurring in full test, and every within-role normalized duplicate group. A leaf is eligible only with at least two remaining train rows and one row in each later split.

Hash-order parents and leaves using the protocol namespace. For parents with three or more eligible leaves, reserve distinct validation/test hidden leaves and keep the rest known. With two leaves, keep one known and alternate the hidden role by deterministic parent index. Singleton leaves remain known. This is schema-aware benchmark construction, not label-blind sampling. It does not choose using semantic similarity or test performance.

Cap support at 20 per known leaf and queries at 8 per eligible role leaf. Globally hash-order each query role by source ID with a separate namespace, never by gold label. Check disjoint normalized texts and known sibling support **after** deduplication and caps. Caps change the natural class-frequency distribution.

Verified real-data counts: 58/60 eligible leaves, 31 known, 14 validation-hidden and 13 test-hidden, 18 supported parents. Support has 606 rows; validation 334 (111 hidden); test 334 (104 hidden). Zero cross-role text overlap and zero unsupported hidden parents. The aggregate does not contain utterances, IDs, or per-example predictions.

## Method/scorer boundary

Method input contains pure utterances and support annotations using opaque node codes. Every node uses its code for both `id` and nonempty `label`, with empty description. One opaque root R000 parents all P nodes; known L nodes point to their P nodes. No original semantic names, hidden-leaf nodes, or query parent labels may reach the method. Query scorer records stay aligned with globally shuffled texts. The supplied hierarchy is supervised known topology, not inferred hierarchy.

The Python return-value separation is an interface contract, not a security sandbox. A future runner must keep scorer objects private, fit only support, calibrate only validation, freeze configuration and predictions before test scoring, and never allow test scorer feedback to alter the run. An absent-parent diagnostic is not included; if added it must have a separate split/report and must not be pooled with supported-parent results.

## Gates before a model replay

1. Reproduce source hashes, split fingerprint, caps, counts, root/reference integrity and deterministic ordering. Freeze protocol, model revision, pooling, normalization, batch settings, seed and hardware. Do not reuse unrelated pilot row embeddings.
2. Encode only utterances with the same pinned E5 configuration for all arms. Share the exact same support vectors and topology. Compare flat maximum-leaf similarity followed by parent mapping against direct parent centroids: normalize each leaf centroid, take an **equal-weight** mean of child leaf centroids, then normalize the parent centroid. Do not weight parents by support sample frequency.
3. Select leaf-rejection thresholds and any other hyperparameters with validation only. Define tie rules and validation objective before test execution. Report validation-hidden and test-hidden separately; their leaves are disjoint.
4. Report parent accuracy over all hidden rows, correct-parent conditional on leaf rejection with its denominator, and joint leaf-rejection AND correct-parent over **all** hidden rows. Report known-leaf accuracy, coverage and false-novelty rate separately, with counts. A zero rejected denominator is undefined, not perfect accuracy.
5. Publish runtime/memory and provenance alongside metrics. One bounded split is not a universal/business claim, and no numerical reproduction of legacy WOS results is implied.

## Reproduce (external scratch paths)

```sh
python -m venv /tmp/hierarchy-preflight-env
/tmp/hierarchy-preflight-env/bin/pip install pyarrow==19.0.1
/tmp/hierarchy-preflight-env/bin/python scripts/prepare_datasets.py massive_zh_cn --cache /tmp/hierarchy-data/massive_zh_cn --download --report /tmp/hierarchy-source-audit.json
/tmp/hierarchy-preflight-env/bin/python scripts/prepare_hierarchy.py --cache /tmp/hierarchy-data/massive_zh_cn --output /tmp/hierarchy-audit.json
/tmp/hierarchy-preflight-env/bin/python -m unittest discover -s tests -p test_hierarchy_preflight.py
```

Only the aggregate audit is versioned. Raw data/cache and any future embeddings remain external to the checkout.
