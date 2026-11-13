# Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence

Citation key: `bianchi2021ctm`

- Authors: Bianchi, Federico and Terragni, Silvia and Hovy, Dirk.
- Venue/status: ACL-IJCNLP 2021, Volume 2: Short Papers; 2021.
- Original source: https://aclanthology.org/2021.acl-short.96/
- Official implementation: https://github.com/MilaNLProc/contextualized-topic-models
- Reviewed: 2026-11-13.

## Fit and limits

Combines contextualized document representations with neural topic models. Established soft document-topic baseline, not a recent 2025/26 result. Requires training, chosen topic count and consistent vocabulary/BOW preprocessing. Evaluate topic coherence/diversity and document-topic quality; converting distributions with argmax is a separate declared protocol.

See the [ecosystem and evaluation guide](../ecosystem_2026-11-13.md) for the dated popularity snapshot and proposed comparisons. This is a reference, not a reproduction, installed dependency or implemented backend. No upstream files/weights are redistributed.
