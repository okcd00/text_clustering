# TopicMerge: A Divide and Conquer Approach to Topic Modeling

Citation key: `raffel2026topicmerge`

- Venue: EMNLP Findings, October 2026.
- Source: https://aclanthology.org/2026.findings-emnlp.2248/
- PDF (upstream terms apply): https://aclanthology.org/2026.findings-emnlp.2248.pdf
- Official code entry: https://github.com/pnnl/TopicMerge
- Reviewed: 2026-11-13; formal proceedings and available upstream source.

## Evidence, limitations and fit

Over-segment documents into micro-clusters, then use an LLM to arbitrate local-neighborhood merges and relabel clusters. Candidate for offline unknown-K clustering and rejected-buffer consolidation, not an online router.

Paper Tables 1–2: Gemini 2.5 Flash-Lite yields Bills ARI 0.42/NMI 0.53 and Wiki ARI 0.71/NMI 0.73. It uses 645/562 prompts versus algorithm-derived lower bounds of 8,024+/15,242+ for document-level baselines. The reported >92% call reduction is not an equivalent token, monetary-cost or latency reduction.

Only Bills/Wiki are tested, with Gemini embeddings, PaCMAP to 64 dimensions and 80 initial clusters. Dependence on initial purity, short-text embedding quality and closed models limits transfer. The paper-listed code URL returned HTTP 404 during this review; implementation availability is unverified.

**Next comparison:** same embedding matrix and initial micro-clusters, geometric versus LLM-assisted merges; measure ARI/NMI, false merges, tokens and latency on a small public short-text subset. No automatic taxonomy updates.

This reference records analysis and links only. No PDF, upstream code or weights are redistributed; no reproduction or production integration is claimed.
