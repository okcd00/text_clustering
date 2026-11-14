# Pinned data preparation · 2026-11-14

No model, embedding, API inference or benchmark score is produced. Historical result files are untouched. Payloads and per-row outputs are never stored in this repository. This workflow verifies sources, counts and split risks before experiments.

## Run

From the repository root, choose a cache/report directory **outside** the checkout. The script rejects checkout paths (including symlinks resolving into it). It does not execute downloaded code.

```bash
python scripts/prepare_datasets.py banking77 \
  --cache /tmp/text-clustering-data/banking77 --download \
  --report /tmp/text-clustering-data/banking77-audit.json

# Use an isolated environment for the optional trusted Parquet parser:
python -m venv /tmp/text-clustering-data-venv
/tmp/text-clustering-data-venv/bin/pip install pyarrow==19.0.1
/tmp/text-clustering-data-venv/bin/python scripts/prepare_datasets.py massive_zh_cn \
  --cache /tmp/text-clustering-data/massive_zh_cn --download \
  --report /tmp/text-clustering-data/massive-zh-audit.json
```

Without `--download`, only cached files are checked. Every artifact must match the pinned SHA-256 and byte length. MASSIVE is the versioned Hugging Face conversion under the official AmazonScience namespace, **not a claim of byte-for-byte original tarball validation**. BANKING reuses the existing pinned source rather than replacing it. Internet access is needed only on the first download; no text is transmitted outward.

The checker audits nonempty text, exact/normalized duplicate counts, cross-split text overlap and ID overlap, label coverage, locale/partition consistency (MASSIVE), and one-parent-per-intent consistency. It fails on hash/count, ID or hierarchy violations. Text overlap is reported as a warning condition rather than silently deleting official samples. Normalization is Unicode NFKC + casefold + collapsed whitespace, with punctuation preserved; it is **not semantic deduplication**.

## Observed aggregate evidence

| Data | Official split observations | Cross-split overlap | Meaning |
|---|---|---|---|
| BANKING77 | 10,003 / 3,080 rows; 77 / 77 intents; no native validation, hierarchy or persistent utterance IDs | train/test: 0 exact, **7 normalized** distinct shared strings | Exact-only checks would miss collisions; synthesized row positions are not proof of independent source examples |
| MASSIVE zh-CN | 11,514 / 2,033 / 2,974 rows; **60 / 59 / 59** intents; 18 scenarios each | train/dev **132**, train/test **187**, dev/test **46** distinct shared exact strings; same normalized counts; no shared IDs | Unique IDs do not guarantee unique content; localization can map distinct sources to identical text; no claim of misconduct or score inflation magnitude |

Within-split duplicate excess: BANKING train/test normalized 4/1 (exact 0/0); MASSIVE train/dev/test exact and normalized 492/15/53. Excess counts are rows minus distinct strings, while cross-split counts are intersection sizes, **not numbers of row pairs**.

[MASSIVE aggregate report](massive_zh_cn/audit_2026-11-14.json) includes missing labels (`audio_volume_other` in dev, `cooking_query` in test); [BANKING report](banking77/audit_2026-11-14.json) retains the pinned artifacts and full public-label histograms. These are source audits, not model results. They contain no text, per-row IDs or text hashes.

## Before an actual comparison

Keep two explicitly named tracks: **official-split** for comparability and a separately specified **duplicate-controlled sensitivity analysis**. Do not silently alter official splits or transfer samples between them. Before the second track, pre-register normalized-text grouping, precedence/dedup rules, new split counts/class coverage and loss of rare classes. Matching text can also have different labels, so inspect aggregate conflicts before treating it as redundant. No deduplicated benchmark has been executed here.

Known-K means the documented information budget: 60 training intents, 59 test-present intents and a K selected from validation are different conditions. Computing K from test labels is oracle information and must be labelled as such; unknown-K methods do not receive it. BANKING still needs a training-derived validation split. MASSIVE cross-language evaluations must group aligned original IDs as well as audit text overlap.

Next gate: approve/freeze this data protocol, then establish a new pinned encoder baseline. Do not backfill or compare directly to unreconstructable historical encoder scores.
