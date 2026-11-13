# Dial-In LLM: Human-Aligned LLM-in-the-loop Intent Clustering for Customer Service Dialogues

Citation key: `hong2025dialin`

- Venue: EMNLP main conference, November 2025.
- Source: https://aclanthology.org/2025.emnlp-main.300/
- PDF (upstream terms apply): https://aclanthology.org/2025.emnlp-main.300.pdf
- Official code entry: https://github.com/mengze-hong/Dial-in-LLM
- Reviewed: 2026-11-13; formal proceedings and available upstream source.

## Evidence, limitations and fit

LLM-assisted cluster coherence assessment, naming and iterative clustering, with dialogue-role and contextual merging. Useful as an offline intent-cluster reviewer.

Table 4: Chinese customer-service NMI 0.7899±0.0135 for KMeans, 0.8208±0.0090 for context-free LLM-ITL, and 0.8826±0.0060 with role+merge. The full gain includes domain/context processing, not just a clustering backend. The abstract's >95% assessment alignment is not overall clustering accuracy; naming accuracy in Table 3 is approximately 92.8–94.4%.

Official README exposes sample code/data (110 annotated cluster examples, 1,507 intent labels) and says fine-tuned models will be uploaded later. Full 100K-dialogue data and released fine-tuned weights were not established. Chinese dialogue roles do not automatically transfer to other text streams.

**Next comparison:** small held-out cluster-coherence review set, measuring false acceptance, calibration and annotation agreement independently of NMI. Keep expensive model calls offline.

This reference records analysis and links only. No PDF, upstream code or weights are redistributed; no reproduction or production integration is claimed.
