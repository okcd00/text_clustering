# MASSIVE sibling breadth preflight v1

Dataset-only, exploratory schema allocation. **No model, rejection, clustering or outcome scoring. Not an untouched confirmation set.** This is a new split, not an extension of the frozen 221 rejected rows.

The protocol was committed before real-data execution. Preserve the pinned MASSIVE zh-CN source artifacts, original full hierarchy preflight, official source roles and deduplication rules. Select every parent with at least four eligible leaves. Within each parent, SHA256-order original leaf metadata under `massive-sibling-breadth-v1|role-allocation|leaf`: one validation-hidden, two test-hidden, all others known. No text similarity or ratings select the roles.

Support is capped at 20 rows per known leaf; queries at 8 per allowed role leaf. Within-role duplicate groups are completely removed, and earlier source roles exclude normalized text occurring in later source roles. Query order is globally hash-shuffled, not grouped by gold label. The method sees opaque known topology and query text only; hidden labels and source IDs stay in separate private files. `post_cap_leaf_counts` publishes only opaque leaf/parent codes and actual counts.

## Executed dataset-only check

All three official artifact hashes and the complete original hierarchy aggregate match. Two independent preparation passes produce identical aggregate reports and private-file manifests. No encoder or model was loaded.

| Role | Actual rows | Known rows | Hidden rows | Leaves |
|---|---:|---:|---:|---:|
| Support | 256 | 256 | 0 | 13 |
| Validation | 143 | 98 | 45 | 19 |
| Test | 191 | 99 | 92 | 25 |

Six supported parents each retain one validation-hidden leaf and two distinct test-hidden siblings. Actual per-leaf support counts are 16 or 20; validation counts 5 or 8; test counts 3, 4 or 8. Do not replace these with nominal caps. The aggregate `preflight.json` records every opaque leaf/parent count. No cross-role normalized-text overlap or selected ID reuse remains. Hidden parents retain known sibling support.

No selected-row manifest or raw text is committed. Historical sample exposure remains unverified; this is an exploratory split, not a new independent confirmation set. The previous 221-reject test population is not reused as this split's rejection population.

## Replay

```sh
python -m venv /tmp/sibling-env
/tmp/sibling-env/bin/pip install pyarrow==19.0.1
/tmp/sibling-env/bin/python scripts/prepare_datasets.py massive_zh_cn --cache /tmp/sibling-source --download --report /tmp/sibling-source-audit.json
/tmp/sibling-env/bin/python scripts/prepare_hierarchy_siblings.py --cache /tmp/sibling-source --private-output /tmp/sibling-private --output /tmp/sibling-report.json
python -m unittest discover -s tests -p test_hierarchy_siblings.py
```

All payloads remain outside Git. The private directory contains method inputs, aligned scorer annotations, selected source IDs and a hash manifest. The aggregate alone is not sufficient to recover selected rows. IDs are preserved for subsequent historical-overlap audit; their existence does not prove untouchedness.

## Gates before future scoring

1. Reproduce official hashes and the entire old aggregate; pass expected six-parent / 13-known / 6-validation-hidden / 12-test-hidden schema, normalized-text and ID disjointness, known sibling support, and actual post-cap counts.
2. Review the new source-selection manifest privately against all prior experiment selections before making any novelty-of-evaluation claim. Without that audit label results exploratory.
3. Freeze a separate rejection/clustering protocol, input representation and scoring denominators before any new outcomes. This preflight does not authorize reusing an old threshold under a changed label split. Pinned E5 may be reused later under that new protocol; no encoder runs here.
4. One hidden validation leaf per parent cannot tune separation of two hidden siblings. Keep clustering fixed or design a separate validation task; do not choose global vs bucket using test outcomes.
5. Report all actual rejects, including known contamination; report missed hidden rows in end-to-end denominators. Sample caps alter frequencies. Per-parent counts are small, and neither this schema nor repeated runs establish generality.

Alternate only: three parents with at least five leaves can reserve two validation-hidden and two test-hidden siblings with remaining known support. This narrower strict design is not run, pooled or selected post hoc against this breadth split.
