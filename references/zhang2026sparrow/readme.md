# SPARROW: Scalable Taxonomy Induction via Structure-Preserving Partitioning and Constraint-Guided Merging

Citation key: `zhang2026sparrow`

- Venue: EMNLP main conference, October 2026.
- Source: https://aclanthology.org/2026.emnlp-main.1502/
- PDF (upstream terms apply): https://aclanthology.org/2026.emnlp-main.1502.pdf
- Official code entry: https://github.com/rebeccazyr/SPARROW
- Reviewed: 2026-11-13; formal proceedings and available upstream source.

## Evidence, limitations and fit

Concept-vocabulary taxonomy induction, not document clustering: spectral partitioning, local LLM trees and constraint-guided fusion. Relevant to hierarchy proposal review and parent-placement consistency.

Table 1, 1K nodes with GPT-5: Ancestor F1 is 0.623 on CCS and 0.745 on Google Product Taxonomy (LLMscorer: 0.476/0.453), but Edge F1 0.412/0.579 is below LLMscorer 0.482/0.588. Do not claim universal metric superiority.

Assumes a complete concept inventory and single-parent rooted trees; missing intermediate concepts and multiple-parent DAGs are not solved. Incremental fusion of blocks is not streaming event learning. Multi-round LLM merging adds token overhead. Official source was accessible; no implementation was executed or license clearance inferred.

**Next comparison:** parent accuracy, Edge/Ancestor F1, false attachment and rejection, while retaining reviewable NodeProposal and single-parent checks. Separate from flat clustering metrics.

This reference records analysis and links only. No PDF, upstream code or weights are redistributed; no reproduction or production integration is claimed.
