# First public snapshot verification

- Fresh Python 3.10 virtual environment: installed using `python -m pip install -r requirements.txt`; NumPy 2.2.6, scikit-learn 1.6.1, SciPy 1.15.3. `pip check` passed.
- Three offline test cases passed: complete example (partitioning, routing, navigation), cached-vector identity / two backend selections, and published result counts / aggregate consistency.
- All 17 summaries and 85 per-seed projected records were compared with their source evidence without changing numeric values. Parameters retain safe scalar settings only.
- Public-content audit passed: no private-service markers, machine paths, detected credential assignments, dataset/PDF payloads or LFS pointers in the publication tree. This is an allowlisted export with manual scope review, not a claim that regex proves confidentiality.
- Local HTTP dashboard returned 200 for the page and result data. JavaScript syntax check passed.
- Browser automation timed out: visual layout, narrow-screen interaction and native dialog rendering have **not** passed a browser inspection in this release. CSS includes responsive layout and horizontal table scrolling; this is implementation, not visual verification.
- Optional dependency combinations are not part of the clean core installation acceptance.

A CI template is supplied in `docs/ci-example.yml`. The current publishing credential does not grant workflow modification scope, so no GitHub Actions workflow is installed. Linux CI verification remains pending.
