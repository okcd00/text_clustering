# UCI calendar paired streaming pilot: measured results

**Executed 2026-11-14**, five methods, two complete deterministic repeats. All non-timing/resource fields and per-row prediction digests match exactly between repetitions. This is repeatability, not ten independent random trials or a confidence interval. [Aggregate JSON](results.json) · [protocol and reproduction](../../experiments/calendar-paired-v1/README.md).

## Same-input comparison

One frozen, pinned multilingual E5-small representation: 5,632 title vectors, including 2,048 warmup and seven UTC days × 512 scored rows. Model weights are hash-verified safetensors; no remote inference. BIRCH threshold 0.3, no final global clustering or oracle K. Predict the entire day before learning any title from it. Ground truth is used only to score.

| Policy | STORY ARI, row-weighted within-day mean | First-title view | Final leaf CF | Final serialized model, MB | Test update/maintenance seconds, runs 1 / 2 |
|---|---:|---:|---:|---:|---:|
| Frozen warmup | .315965 | .316013 | 891 | 7.40 | .0005 / .0005 |
| Unbounded | .415860 | .415508 | 2,473 | 20.37 | 2.767 / 3.356 |
| Freeze at 2,048 CF | .413349 | .413082 | 2,048 | 16.51 | 1.749 / 1.996 |
| Rebuild last 2,048 rows / 2,048 arrivals | .415855 | .415545 | 1,560 | 12.50 | 2.845 / 3.298 |
| 128-CF sensitivity, already exceeded at warmup | .315965 | .316013 | 891 | 7.40 | .0016 / .0015 |

MB are decimal bytes / 1e6. Serialized bytes measure storage, **not resident algorithm memory**. The 128 sensitivity is a degenerate freeze control, not an achieved 128-CF budget: all methods begin from the identical 891-CF warmup state.

## Timing of the effects matters

| Date (UTC) | Unbounded STORY ARI | Frozen delta | 2,048-CF freeze delta | Rolling delta |
|---|---:|---:|---:|---:|
| Mar 24 | .429713 | .000000 | .000000 | .000000 |
| Mar 25 | .395618 | -.054170 | .000000 | .000000 |
| Mar 26 | .443108 | -.155933 | .000000 | .000000 |
| Mar 27 | .433855 | -.153780 | .000000 | .000000 |
| Mar 28 | .420932 | -.074389 | .000000 | -.012583 |
| Mar 29 | .466669 | -.154510 | .000000 | -.014517 |
| Mar 30 | .321123 | -.106484 | -.017579 | +.027067 |

- Capacity freeze first skips **test arrival 2,641**, the **81st selected arrival on Mar 29**. There are 432 skipped updates on Mar 29 and 512 on Mar 30, total 944. Because the day's predictions precede all its updates, **only Mar 30 scores see the freeze**. Its -.017579 ARI change is diluted to -.002511 in the seven-day mean. This short suffix cannot establish acceptable long-term degradation.
- Rolling rebuild occurs after arrival 2,048, at the end of Mar 27, so the first four scored days match. Its three subsequent changes are negative, negative, positive. The almost equal seven-day mean is **cancellation, not unchanged behavior**, and does not prove statistical equivalence. A longer post-rebuild horizon is needed.
- Updating beats the permanently frozen control on every day after the first in this bounded pilot. This is evidence about these news-event labels and preset settings, not all text streams.

## Costs and limits

Local CPU float32 encoding of 5,632 short titles took 83.54 seconds including vector-cache write, no truncation; encoder-process peak RSS 1,062.75 MiB. This is offline batch encoding, not per-request latency. Model load and source download are outside that encoding timer.

Fresh algorithm workers each peak around 648–650 MiB, dominated by source loading/auditing and imports. Thus this run **does not isolate or demonstrate a RAM reduction** for the policies. Rebuild temporarily retains old and new trees. Prediction and full warmup timing are retained separately in JSON; update time includes guard checks and rebuilding. No universal speed advantage is claimed from this one machine; repeats show noticeable timing variability.

STORY is a silver event grouping. CATEGORY is a separate four-category task, with ARI roughly .042–.044; it is not the same semantic objective. First-title scoring (3,553 rows) retains the identical raw learning stream (3,584 rows). This does not remove duplicate exposure from training. Counts are CF microclusters, not selected final K. ARI is within-day and label-permutation invariant; it does not establish stable cross-day IDs or detection of birth/merge/split events.

This is a thinned replay, not the archived scenario-18 row set, not real user behavior, and not an online recommendation experiment. No historical scores are spliced into the comparison. No raw titles, URLs, source row IDs, vectors or individual predictions are included in this commit.
