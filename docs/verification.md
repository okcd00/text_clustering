# First public snapshot verification

- Fresh Python 3.10 virtual environment: installed using `python -m pip install -r requirements.txt`; NumPy 2.2.6, scikit-learn 1.6.1, SciPy 1.15.3. `pip check` passed.
- Three offline test cases passed: complete example (partitioning, routing, navigation), cached-vector identity / two backend selections, and published result counts / aggregate consistency.
- All 17 summaries and 85 per-seed projected records were compared with their source evidence without changing numeric values. Parameters retain safe scalar settings only.
- Public-content audit passed: no private-service markers, machine paths, detected credential assignments, dataset/PDF payloads or LFS pointers in the publication tree. This is an allowlisted export with manual scope review, not a claim that regex proves confidentiality.
- Local HTTP dashboard returned 200 for the page and result data. JavaScript syntax check passed.
- Browser automation timed out: visual layout, narrow-screen interaction and native dialog rendering have **not** passed a browser inspection in this release. CSS includes responsive layout and horizontal table scrolling; this is implementation, not visual verification.
- Optional dependency combinations are not part of the clean core installation acceptance.

A CI template is supplied in `docs/ci-example.yml`. The current publishing credential does not grant workflow modification scope, so no GitHub Actions workflow is installed. Linux CI verification remains pending.

## Hosting check

GitHub Pages built the first publication successfully from `main:/docs`. This account already uses a custom Pages domain, so the canonical URL is https://dian.host/text_clustering/ and HTTPS enforcement is enabled. The generic github.io address inherits the account's domain redirect.

The current verification network blocks that custom domain by policy. The deployment status is confirmed through GitHub, but a live-site HTTP/content check and browser inspection are still pending from an allowed network. No account-wide domain or DNS configuration was changed.

## Scenario evidence expansion (2026-11-14)

Added 18 independently grouped historical comparisons, 51 configurations and 185 seed records, plus 10 single-replay view/configuration rows within those 51. These are not 10 independent chronological experiments. Every exported numeric summary and seed projection was checked against its source snapshot during export; no experiment or paid call was rerun. Retained source digests, fixed settings, denominators and applicable caveats. Internal run names, service identities, file paths and text-level outputs were excluded.

The expanded five-test offline suite passed in the clean core environment; JavaScript syntax and public-content audit passed. Browser visual acceptance remains pending; this update does not change that status. The new scenario page separates comparison groups and does not grant cross-method comparability merely because dataset names match.

## Homepage selection summary

The homepage now renders five scenario recommendation cards directly from `scenarios.json`, with dataset/protocol scope, quality and cost alongside each choice. Each detail link selects its exact evidence group. The Node DOM-stub test (`node tests/test_home.cjs`) checks five rendered cards, expected evidence values and deep links; this is functional rendering verification, not browser visual acceptance. No experimental results were changed or rerun.
