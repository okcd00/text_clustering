# Minimal comparison plan: TabClustPFN

Status: proposed, not executed. References: [paper and code](readme.md), [batch protocol](../../docs/protocols.md), [scenario protocol](../../docs/scenario-protocols.md).

## 1. Stop/go before compute

- Confirm code and checkpoint permissions for the intended setting; record upstream code SHA, checkpoint checksum and dependency versions. Stop if unresolved.
- Use public, authorized text data only. Historical WOS embeddings have an unspecified checkpoint revision, so this must be a **new paired benchmark**, not a reproduction or direct numerical comparison with historical dashboard rows.
- Pin a downloadable semantic encoder revision, dataset/split, document IDs, seed and matrix digest. Never substitute character hashing for semantic evidence.

## 2. Freeze the problem

Start with public WOS: freeze an existing router and its validation-selected rejection threshold; hold out selected classes, then construct identical rejected-text buffers containing both hidden-class texts and wrongly rejected known-class texts. Use labels only to construct documented benchmark strata and evaluate, not to tune on the test buffers. Do not assume every rejected text belongs to a new class.

Use 500 and 1000 texts, seeds 0–4, with controlled 2–10-class subsets. Add a small diagnostic set for K=1, >10 classes, overlap, imbalance and noise. K=1 is essential: the current CIN candidate range starts at 2, so report unsupported/forced splitting rather than quietly filtering such buffers out. These constrained subsets do not represent full WOS taxonomy discovery.

## 3. Fair backend comparison

Fit PCA on a disjoint calibration split, then freeze it. Start with PCA32; expand to PCA64 only if the first pass is promising. Give every paired backend exactly the same ordered matrix and dtype.

- **Oracle-K diagnostic:** KMeans / MiniBatchKMeans and TabClustPFN with the same true K. Keep separate from deployable unknown-K results.
- **Unknown-K:** TabClustPFN CIN against the repository's HDBSCAN and one preselected unknown-K baseline. Fix settings on calibration data, not the test labels. Report noise/coverage explicitly.
- Retain raw-embedding baseline results to separate the effect of dimensionality reduction from the backend.

The repository's cold-start path defaults to L2 normalization: explicitly use `preprocessing='none'` to preserve the shared PCA matrix. For TabClustPFN call the direct `cluster(model, X, ...)` path rather than its CSV loading wrapper. The wrapper fits feature scaling on the current batch and removes constant columns, so passing a nominal normalization flag is not sufficient to guarantee identical input. Confirm the direct API at the pinned revision and assert matrix equality before dispatch.

A separate, labeled preprocessing arm may test upstream-style feature standardization. Apply the identical transformation to all compared backends, declare whether fitting is calibration-only or transductive per-buffer, and do not mix it with the frozen-PCA primary arm. Paper C.3 uses per-feature standardization; bypassing it is a controlled transfer experiment, not a claim of exact paper reproduction.

## 4. Evidence and promotion gate

Report paired per-seed ARI/NMI, pairwise precision/recall, K error, cluster purity/fragmentation and coverage on the same denominator. Report all-sample and assigned-only results separately for rejecting methods; never make all noise points one meaningful cluster. Include known-class contamination of proposed new groups and cluster stability under resampling/order changes.

Measure encoding, PCA, model load and clustering separately, plus total wall time, peak RSS/VRAM and hardware. Compare warm and cold starts; no published speedup becomes a local SLA.

Before running, predeclare a minimum useful quality delta and resource budget on the calibration split. **Go** only if paired gains persist across seeds/buffer conditions, are not explained solely by preprocessing, and meet that budget without unacceptable contamination or fragmentation. **Stop** if gains are negligible, fragile, costly or license-blocked; archive the negative result. Only after passing this gate consider generating reviewable new-node proposals. Never automatically alter a taxonomy based on this clustering output.
