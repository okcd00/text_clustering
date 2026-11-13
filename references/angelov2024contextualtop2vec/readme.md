# Topic Modeling: Contextual Token Embeddings Are All You Need

Citation key: `angelov2024contextualtop2vec`

- Authors: Angelov, Dimo and Inkpen, Diana.
- Venue/status: Findings of the Association for Computational Linguistics: EMNLP 2024; 2024.
- Original source: https://aclanthology.org/2024.findings-emnlp.790/
- Official implementation: https://github.com/ddangelov/Top2Vec
- Reviewed: 2026-11-13.

## Fit and limits

Contextual-Top2Vec uses contextual token embeddings for multiple topics and topic spans within documents, hierarchical topics and phrase labels. Useful when a document contains several intents; token-level representations add resource cost. This is a distinct paper in the same Top2Vec software repository, not another independent star count. Evaluate span/topic quality separately from one-label document clustering.

See the [ecosystem and evaluation guide](../ecosystem_2026-11-13.md) for the dated popularity snapshot and proposed comparisons. This is a reference, not a reproduction, installed dependency or implemented backend. No upstream files/weights are redistributed.
