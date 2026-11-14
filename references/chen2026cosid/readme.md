# Lifelong Intent Discovery: A Continual Learning Protocol and Method

Citation key: `chen2026cosid`

- Authors: Peng Chen, Yu Xia, Bang Wang.
- Venue: EMNLP 2026 main conference, October; pages 16286–16314.
- Formal source: https://aclanthology.org/2026.emnlp-main.705/
- PDF (upstream terms apply): https://aclanthology.org/2026.emnlp-main.705.pdf
- Official repository: https://github.com/hustchenpeng/COSID
- Reviewed: 2026-11-14, formal paper and public README/configuration; no upstream code, dataset or model was executed.

## Contribution and relevance

COSID is the continual open-stream intent-discovery **task/protocol**; CIDA is the proposed **agent**. Static CLINC150/BANKING77 data become staged streams mixing historical and newly introduced intents. This supplies a more targeted staged stress test for routing, rejection and discovery than treating a batch clustering score as evidence of continual adaptation. It is synthetic staging, not authentic event timestamps or proof of real-world drift handling.

CIDA uses retrieval and LLM screening per sample, K-means on the discovery buffer, LLM intent profiles and memory expansion. Its compute profile is not a lightweight drop-in replacement. Earlier mistakes can accumulate in memory.

## Verified paper evidence

Table 1, ClincCL, old-intent recurrence ratio alpha=0.5; percentages, averaged over five incremental stages (experiments report three runs):

| Method | D-score | OSR ↓ | USR ↓ | ARI | ACC_all | ACC_new | ACC_old | H-score | AIA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VBCGCD | 85.83 | 0.47 | 21.82 | 76.00 | 82.71 | 65.67 | 85.97 | 74.14 | 90.41 |
| CIDA | 94.72 | 0.73 | 6.22 | 87.41 | 90.94 | 86.08 | 92.13 | 88.77 | 93.90 |

OSR means old samples falsely screened as new; USR means new samples falsely accepted as old. CIDA's lower USR accompanies **worse OSR** in this setting. AIA is the running mean of old-intent accuracy, not all-intent accuracy. These are paper results, not this repository's measurements. See Appendix B for alignment and metric definitions.

## Reproduction and permission caveats

The official [configuration](https://github.com/hustchenpeng/COSID/blob/main/cida/config.py) sets `ESTIMATE_K=False`: default execution uses the protocol's ground-truth count of new intents per stage. [The runner](https://github.com/hustchenpeng/COSID/blob/main/cida/run.py) requires `--estimate-k` for estimation, whereas paper Section 4.3 describes estimating K. Keep oracle-K and estimated-K arms explicitly separate; do not label the default run fully unknown-K or silently assume it reproduces the paper setting.

The released README describes random/cohesion/expansion intent scheduling, old-intent recurrence and Zipf support imbalance. Held-out test data must be filtered to labels introduced by the current stage, not evaluated wholesale before introduction. Code is described as for research reproducibility; a standard permissive license was not established. Source datasets retain their own licenses. Do not infer commercial reuse clearance.

## Small split-only experiment proposal (not executed)

1. Borrow the **protocol idea first**, without CIDA, an LLM service, new dependencies or copied upstream code. Reuse an approved public intent dataset and a pinned cached encoder under the existing benchmark harness. Keep labels private to split construction and scoring.
2. Start with one fixed random stage schedule and seed: labeled base intents, then five unlabeled mixed stages. Compare recurrence alpha=0 against alpha=0.5 with identical intent-introduction order and new-intent samples. Audit sample IDs for train/test separation and reuse without replacement.
3. Hold the existing router, threshold and backend fixed; evaluate before any stage update, then update only under the existing permitted protocol. Report per-stage OSR/USR, discovery ARI, old/new accuracy, buffer volume and runtime; separate oracle-K from existing unknown-K behavior. Do not imply eventwise prequential evaluation when using stage-level windows.
4. Only if this inexpensive split change exposes a meaningful weakness, add further seeds and skew or semantic schedules. Reserve CIDA/LLM screening for a later explicitly budgeted offline comparison. Preserve reviewable taxonomy proposals and do not claim stable cross-stage IDs without testing them.

This note archives links and analysis, not a redistributed PDF, implementation, benchmark run or deployment recommendation.
