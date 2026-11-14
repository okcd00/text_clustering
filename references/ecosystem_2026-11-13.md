# Text clustering and topic-model ecosystem: 2026-11-13

## Scope and evidence

This is a curated practical shortlist, **not an exhaustive GitHub ranking**. Stars are a 2026-11-13 GitHub repository API snapshot and measure adoption/visibility, not clustering quality. Recent accepted papers and established popular software are separate selection axes. No dependencies, upstream code or weights were installed or executed. Existing references were deduplicated; no experiment scoreboard is changed.

| Repository | Stars | Repository license | Role / reference |
| --- | ---: | --- | --- |
| [BERTopic](https://github.com/MaartenGr/BERTopic) | 7,957 | MIT | Existing [reference](grootendorst2022bertopic/readme.md), modular topic-model baseline; not added again |
| [Top2Vec](https://github.com/ddangelov/Top2Vec) | 3,115 | BSD-3-Clause | [Classic](angelov2020top2vec/readme.md) and [Contextual-Top2Vec](angelov2024contextualtop2vec/readme.md); one shared repo/star count |
| [Contextualized Topic Models](https://github.com/MilaNLProc/contextualized-topic-models) | 1,276 | MIT | [CTM](bianchi2021ctm/readme.md), trained soft-topic baseline |
| [TopMost](https://github.com/bobxwu/TopMost) | 294 | Apache-2.0 | [ACL 2024 demo toolkit](wu2024topmost/readme.md), not a standalone clustering algorithm |
| [FASTopic](https://github.com/bobxwu/FASTopic) | 170 | Apache-2.0 | [NeurIPS 2024 main paper](wu2024fastopic/readme.md), fixed-K learned topic model |

License labels concern repositories, not blanket clearance for all models, data or hosted inference services. Contextual-Top2Vec is a beta path in the Top2Vec library with restricted encoder support; do not assume every classic Top2Vec feature is supported by the contextual mode.

## Recent conference additions and task separation

- **Offline text/intent clusters:** [TopicMerge](raffel2026topicmerge/readme.md), EMNLP Findings 2026; [Dial-In LLM](hong2025dialin/readme.md), EMNLP main 2025. TopicMerge's listed code returned 404; Dial-In releases samples and does not establish availability of the full training data or fine-tuned checkpoints.
- **Concept hierarchy, not raw-document clustering:** [SPARROW](zhang2026sparrow/readme.md) and [PEEL](kim2026peel/readme.md), EMNLP main 2026. SPARROW assumes a complete concept inventory and one parent; PEEL needs source-domain gold parent edges and a supplied concept vocabulary.
- **Soft topic interpretation:** [LLM-ITL](yang2025llmitl/readme.md), ACL main 2025. Compare within topic models rather than claiming its coherence scores beat flat-cluster ARI/NMI.
- **Established practical additions:** Contextual-Top2Vec is EMNLP Findings 2024, CTM is ACL 2021 short, TopMost is ACL 2024 system demonstrations, and FASTopic is NeurIPS 2024 main. They are included for practical baseline value, not advertised as 2025/26 discoveries.

## Suggested experiment order (prospective, not executed)

1. **Low-cost public-data baseline:** use one pinned semantic encoder and preserved public split; compare existing batch backends with classic Top2Vec/BERTopic. Include the same-matrix geometry comparison separately from complete native pipelines whose preprocessing differs. Record noise handling, discovered K, ARI/NMI, coverage, runtime and memory. Neither unknown-K nor density noise labels establish calibrated open-set rejection.
2. **Dedicated soft-topic track:** FASTopic and CTM, optionally organized with TopMost. Explicitly set K, align vocabulary/BOW preprocessing, report coherence/diversity and repeated-seed stability. If argmax topics become hard labels, declare that transformation. Compare topic-model training costs separately from frozen clustering.
3. **Selective semantic refinement:** test a small, held-out rejected-text buffer or cluster-review set. TopicMerge-inspired merge proposals and Dial-In-style coherence checks need verified implementations/data or an explicitly labeled independent reimplementation. Count calls **and** tokens, false merges/false acceptance, latency and variance. Keep review decisions separate from automatic taxonomy insertion.
4. **Hierarchy-only study:** SPARROW is a near-term parent-placement reference; PEEL is conditional on a defensible source-edge supervision set. Measure direct-edge and ancestor quality as well as false attachments. Preserve NodeProposal review/version controls; do not silently replace them with generated edges.

Batch, routing, hierarchy navigation and evolution protocols remain distinct. A method's incremental block fusion or document-add API is not proof of eventwise prequential learning, stable cluster identity or concept-drift handling. Offline evidence and small paired experiments should precede any wider integration. No downstream value is inferred from paper benchmark rankings alone.

## Continual-protocol follow-through

[COSID / CIDA](chen2026cosid/readme.md) (EMNLP main 2026) adds a staged mixture of recurring and new intents. Prioritize its low-compute **split-only** stress test using existing backends before any LLM-agent reproduction. The official CIDA configuration defaults to ground-truth K; keep oracle and estimated-K results separate. This remains synthetic staged evaluation, not actual event-time drift.
