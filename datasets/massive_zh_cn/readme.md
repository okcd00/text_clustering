# MASSIVE zh-CN · pinned preparation

Source: [AmazonScience/massive](https://huggingface.co/datasets/AmazonScience/massive), original project [alexa/massive](https://github.com/alexa/massive), [paper](https://arxiv.org/abs/2204.08582).

This preparation pins **Hugging Face's converted Parquet revision**, not the original Amazon tarball bytes. `source.json` records both the converted revision and separately checked upstream metadata/code revisions. Only three Chinese split files are fetched (~1.08 MB total); the remote dataset loader is not executed. The downloaded metadata names the intent/scenario labels; `utt` is used, not `annot_utt`. Worker IDs/judgments are ignored and never published.

Data license: CC BY 4.0; original SLURP attribution must be retained. See pinned upstream NOTICE. Repository code is Apache 2.0, a separate license. This repository redistributes neither original text nor Parquet.

Observed counts: train 11,514 / validation 2,033 / test 2,974. Train has 60 intents, validation/test each have **59**, all have 18 scenarios. Validation lacks `audio_volume_other`; test lacks `cooking_query`. All 60 intents consistently map to one parent in the combined released splits. Do not use this full-data audit to infer permission to expose hidden labels to an algorithm.

See [aggregate audit](audit_2026-11-14.json) and [reproducible preparation](../preparation.md). Cross-split IDs are disjoint, but identical text overlaps exist; **not** a leakage-free benchmark claim. Chinese utterances are localized assistant interactions, not real e-commerce behavior streams.

The ordinary three splits total 16,521. The 19,521-per-locale figure additionally includes a distinct 3,000-utterance competition holdout, described in [MMNLU-22 §4.1/4.2.1](https://arxiv.org/html/2212.06346). That holdout is not part of the three pinned Parquet files or this audit.
