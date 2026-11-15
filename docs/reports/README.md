# Experiment report pages

Generated HTML, not a second result source. Every `results/*/README.md` has a report page; the sibling preflight has a separately labelled data-only page. The source file remains authoritative.

- `survey/reports.json`: reviewed conclusion, limitation, status, source/reproduction pointers and return anchor. This is editorial metadata, not newly calculated metrics.
- `scripts/build_reports.py`: deterministic, dependency-free Markdown renderer and directory generator.
- `python scripts/build_survey.py`: regenerates Survey plus all report pages.
- `python scripts/build_survey.py --check`: verifies both sources against checked-in pages.
- `python -m unittest discover -s tests -p test_report_pages.py`: coverage, parity, provenance, local links/anchors and Markdown tables/code.

Keep source result Markdown and numerical artifacts unchanged for presentation-only work. When a source report changes intentionally, regenerate its page and review the curated conclusion for staleness. Reports retain a source hash and source/reproduction links. Compare cards remain available within Survey; `#compare` deep links open them, while the default overview stays compact. Existing roadmap anchors remain summaries linking to the dedicated reports.
