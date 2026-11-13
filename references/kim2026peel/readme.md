# Peeling the Matryoshka: Extracting Ontological Structure from Retrieval Embeddings

Citation key: `kim2026peel`

- Venue: EMNLP main conference, October 2026.
- Source: https://aclanthology.org/2026.emnlp-main.1305/
- PDF (upstream terms apply): https://aclanthology.org/2026.emnlp-main.1305.pdf
- Official code entry: https://github.com/Eunchan24/PEEL
- Reviewed: 2026-11-13; formal proceedings and available upstream source.

## Evidence, limitations and fit

Frozen retrieval embeddings plus a 98K-parameter order-embedding probe recover hierarchy from concept names; tree extraction can be expanded to a DAG. This is a low-parameter parent-relation candidate, not raw-document clustering or concept discovery.

Paper reports Graph F1 0.885/0.931 on two OLLM benchmarks, given gold concept vocabularies. The probe uses source-domain gold edges (80/10/10 train/validation/holdout); target-domain transfer is not evidence of zero-training or fully unsupervised learning. Comparisons with document-fed end-to-end ontology systems are cross-task references.

Two-domain evidence does not establish arbitrary depth/domain transfer or downstream utility. Official repository and Apache-2.0 license were visible; no checkpoints were loaded. Model/data licenses remain separate.

**Next comparison:** only after a justified source-edge training split is available, evaluate held-out parent/ancestor correctness and leakage-safe domain transfer. Keep separate from the unknown-K batch leaderboard.

This reference records analysis and links only. No PDF, upstream code or weights are redistributed; no reproduction or production integration is claimed.
