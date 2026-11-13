# Neural Topic Modeling with Large Language Models in the Loop

Citation key: `yang2025llmitl`

- Venue: ACL main conference long papers, July 2025.
- Source: https://aclanthology.org/2025.acl-long.70/
- PDF (upstream terms apply): https://aclanthology.org/2025.acl-long.70.pdf
- Official code entry: https://github.com/Xiaohao-Yang/LLM-ITL
- Reviewed: 2026-11-13; formal proceedings and available upstream source.

## Evidence, limitations and fit

A neural topic model learns global document/topic representations, while LLM-suggested topic words provide confidence-weighted optimal-transport refinement. This is an interpretable soft-topic modeling branch, not a drop-in hard-cluster backend.

Official implementation supports multiple neural topic models and public text datasets. Additional topic-model training, LLM inference and external evaluation resources are needed. The paper explicitly warns that LLM refinements can introduce out-of-corpus information and pretraining bias.

**Next comparison:** topic coherence/diversity and document representation quality plus repeated-seed stability; report training and generation budgets. Do not substitute topic coherence for ARI, rejection quality or downstream value.

This reference records analysis and links only. No PDF, upstream code or weights are redistributed; no reproduction or production integration is claimed.
