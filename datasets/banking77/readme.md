# banking77

Upstream: https://github.com/PolyAI-LDN/task-specific-datasets

Download from the upstream repository/archive listed in `source.json`, follow its license, and verify the recorded SHA-256 where supplied. Data is not included in this release. Keep downloads outside this checkout.

Intent classification: 77 labels; published benchmark uses the official test split (3,080 texts).

2026-11-14: the existing pinned three artifacts were downloaded outside the checkout and their hashes verified. Actual train/test counts are 10,003/3,080, both 77 classes. Train/test share 0 exact but 7 normalized distinct text strings. See [aggregate audit](audit_2026-11-14.json) and [preparation workflow](../preparation.md); this is not a model benchmark or a claim of leakage-free data.
