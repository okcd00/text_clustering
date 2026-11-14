# Calendar replay preflight v1 (not a benchmark)

No encoder, clustering algorithm or evaluation was run. This milestone verifies an official archive and defines a bounded, reproducible stress subset. It does **not** reproduce the historical 26,842-row seven-window experiment.

## Source and audit

[UCI source](https://archive.ics.uci.edu/dataset/359/news+aggregator), attributed to Gasparetti, F. (2017), [DOI](https://doi.org/10.24432/C5F61C). The UCI page specifies CC BY 4.0. [source.json](../../datasets/news_aggregator/source.json) pins the official ZIP and the selected member with byte counts and SHA-256. No archive, title, URL, row ID or model vector is committed.

Actual downloaded ZIP: 29,224,203 bytes. `newsCorpora.csv`: 102,297,000 bytes; 422,937 records, strict UTF-8, eight tab-separated fields without a header. There are 7,231 STORY groups, four coarse categories, two empty publisher fields, one timestamp inversion, 17,862 normalized-title duplicate excess rows and 196 duplicate-URL excess rows.

**Source discrepancy:** the UCI prose describes a March 10–August 10, 2014 collection. The pinned archive's observed UTC timestamps instead span **2014-03-10 16:52:50.698 through 2014-08-28 12:33:11.565**. Do not silently clip the latter records or use source row order as time. The full-source audit is separate from the selected replay in [preflight.json](preflight.json).

## Bounded calendar contract

The [machine-readable protocol](protocol.json) freezes:
- Warmup: `[2014-03-17, 2014-03-24)` UTC, 2,048 selected rows from 22,730 eligible.
- Scored windows: seven consecutive **calendar days**, March 24–30 inclusive, each `[00:00 UTC, next 00:00 UTC)`. Each has 512 selected rows; total 3,584. Every selected date is nonempty in the archive.
- Selection uses the smallest SHA-256 hashes of protocol/role/numeric ID, **before** sorting selected items by `(timestamp_ms, numeric ID)`. It does not inspect story/category labels. This is a thinned replay stress subset, not an authentic event-by-event user stream.
- The primary view scores all selected rows. The secondary view scores only the first normalized title encountered since the warmup boundary (NFKC, casefold, whitespace collapse), totaling 3,553 scored rows. Both views learn from the **same raw selected stream**. This is scoring sensitivity, not duplicate-free training or a leakage correction.
- STORY is a silver news-event grouping for scoring only. CATEGORY is a separate, four-class coarse task; neither is an input feature or hyperparameter-selection signal. Publication timestamps and page pairs are not logged behavior/intent events.

## Proposed next comparison, not executed

Use one frozen public encoder and identical warmup vectors for:
1. A frozen-after-warmup BIRCH control.
2. Unbounded BIRCH updates.
3. Capacity freeze after 2,048 leaf micro-clusters.
4. Rolling rebuild from the last 2,048 learned rows every 2,048 arrivals.

These are proposed engineering constants, not test-label-tuned parameters. The bounded replay may never hit the capacity guard, depending on the selected representation and BIRCH threshold. Record actual triggers rather than infer a capacity benefit; choose a separate calibration prefix or prespecified smaller-budget sensitivity before examining scored labels if a triggering comparison is needed. Micro-cluster count is not a byte-memory bound. Rebuild can temporarily retain two trees and does not impose a hard micro-cluster cap.

At each UTC boundary, snapshot the current state; predict **all selected texts in that day before learning any of them**, then update chronologically. Existing equal-row-count window runners must not stand in for these calendar boundaries. Report per-day quality, counts, state size, observed maintenance events, update/predict time, and whole-process versus algorithm memory. Define the cold/steady compilation policy before timing.

Within-window ARI is label-permutation invariant and **does not** establish persistent cross-window identities. A separate cluster matching/identity protocol is required to measure stable IDs, mergers or splits. Define score weighting and singleton/empty-window behavior before fitting; preserve empty calendar windows if the protocol is extended.

## Reproduce the preflight only

Using Python's standard library:

```bash
python scripts/prepare_news_calendar.py --cache /tmp/uci-news-preflight --output /tmp/uci-news-preflight/report.json --download
```

The flag downloads the official inert archive only when absent and verifies bytes before use. Later omit `--download`. No model/package installation, remote API inference, filesystem extraction or public publication occurs. Copy only a reviewed aggregate report into the repo. Run `pytest tests/test_news_calendar.py`.

Next gates: encoder version/hash and normalization; explicit calendar snapshot implementation; identical maintenance policies and warmup; budget-trigger observations; future-isolation tests; identity metrics. The archived scenario-18 lacks an exact version/split manifest, so its historical numbers remain archived and are not the target values of this new protocol.
