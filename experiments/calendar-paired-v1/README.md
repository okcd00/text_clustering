# Calendar paired BIRCH pilot v1

Protocol frozen before examining scored labels: [protocol.json](protocol.json). Uses the exact 2,048-row warmup plus seven 512-row UTC calendar days from [calendar-preflight-v1](../calendar-preflight-v1/README.md), not the historical 26,842-row replay. All five methods share the same vectors, full warmup, and chronological input order. No labels tune threshold, maintenance, sampling or representation.

## Reproduce

Use the isolated semantic-pilot-v1 environment lock (CPU Torch 2.8.0 and scikit-learn 1.8.0). No remote inference, no private data, no credentials, no raw rows in Git.

```bash
python scripts/prepare_news_calendar.py --cache /tmp/calendar-data --output /tmp/calendar-data/preflight.json --download
python scripts/prepare_semantic_model.py --model-dir /tmp/calendar-model --download
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/calendar-env/bin/python scripts/benchmark_news_calendar.py \
  --cache /tmp/calendar-data --model-dir /tmp/calendar-model --encode
CALENDAR_PYTHON=/tmp/calendar-env/bin/python \
CALENDAR_CACHE=/tmp/calendar-data CALENDAR_OUTPUT=/tmp/calendar-results \
  bash scripts/evaluate_calendar.sh
```

Subsequent evaluations skip preparation and encoding. The runner validates the archive, exact sample order and vector-byte hash; metadata must carry the same committed model artifact manifest. Raw IDs/texts, labels, per-row predictions, vectors and their private digests remain external. Cache files must be regenerated after environment loss. The wrapper is fail-fast and does not automatically publish outputs or partial runs.

## Method boundaries

- All use BIRCH threshold 0.3, branching factor 50, n_clusters=None, float32 normalized E5 features, one thread. These are preset pilot settings, not a tuned recommendation.
- Every day's **entire** query set is predicted before any rows from that day update the model. Updating is then chronological, one row at a time.
- Cap policies preserve the identical full warmup. If it already exceeds a budget, no new samples are learned: this is an update-freeze guard, **not** a strict count/byte cap on the initialized model. A new insertion can reach the cap before subsequent insertions stop. Both 2048 and the prespecified 128 sensitivity are reported even if no trigger occurs.
- Rolling-2048 updates normal arrivals, but at each 2048th arrival rebuilds from the latest 2048 rows **instead of** also inserting that row into the old model. Two trees can coexist briefly; rolling maintenance is not a hard memory cap.
- Primary scores include all 3584 rows. First-title scoring includes 3553, while learning still uses the identical raw stream. This is a scoring sensitivity, not deduplicated training.
- STORY is silver news-event grouping. CATEGORY is a distinct four-class coarse metric, not evidence of intent recognition or product recommendation value. No oracle K or future labels are used.
- ARI is reported within each day and as a row-weighted mean, not pooled across changing cluster IDs. No claim is made about persistent identities, merger/split detection or cross-window stability.
- Serialized model bytes are only a storage-size proxy. Fresh worker process RSS includes source parsing and vector loading, excludes E5, and is not isolated algorithm memory. Rebuild and insertion time are included, while encoding/model preparation are separate. No performance claims from one machine/one pilot.

Tests cover warmup alignment, cap behavior, exact rolling rebuild count, frozen state, snapshot predictions and the independent calendar source selection.

## Execution and independent repeat

[Executed results and per-day paired differences](../../results/calendar-paired-v1/README.md). The aggregate exporter `scripts/summarize_calendar_paired.py` requires two complete sets named `{method}-1.json` and `{method}-2.json` in the external cache, validates exact dates/counts and all non-resource/time fields plus private prediction digests, then removes those digests from its export. It fails on missing, different or incomplete runs. Model/source bytes and exact split provenance remain pinned in the linked manifests.
