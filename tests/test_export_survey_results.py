import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from export_survey_results import export


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.static = json.loads((ROOT / 'results/unified-static-v1/results.json').read_text())
        self.routing = json.loads((ROOT / 'results/routing-v2/results.json').read_text())

    def test_complete_payload_matches_published_results(self):
        result = export(self.static, self.routing)
        self.assertEqual(result, json.loads((ROOT / 'docs/unified-results.json').read_text()))
        self.assertEqual(len(result['runs']), 42)

    def test_explicit_failure_propagates_partial_and_never_publishes_local_trace(self):
        self.static['status'] = 'partial'
        self.static['runs'][0].update(status='error', error='confidential worker details')
        self.routing['runs'][0].update(known_accepted_accuracy=None, known_coverage=0.0,
                                       per_sample_data='must not publish')
        result = export(self.static, self.routing)
        self.assertEqual(result['status'], 'partial')
        self.assertEqual(result['runs'][0]['status'], 'error')
        self.assertNotIn('metrics', result['runs'][0])
        self.assertNotIn('confidential worker details', str(result))
        self.assertNotIn('per_sample_data', str(result))
        self.assertIsNone(result['runs'][30]['metrics']['known_accepted_accuracy'])
        self.assertEqual(result['runs'][30]['metrics']['known_coverage'], 0.0)
        # Fail closed even if an input mislabels an explicit failure as complete.
        self.static['status'] = 'executed'
        self.assertEqual(export(self.static, self.routing)['status'], 'partial')

    def test_reports_missing_duplicate_and_unexpected_keys(self):
        for source in ('static', 'routing'):
            for shape, error in (('missing', '1 missing'),
                                 ('duplicate', 'duplicate run identities'),
                                 ('unexpected', '1 unexpected')):
                with self.subTest(source=source, shape=shape):
                    static, routing = copy.deepcopy(self.static), copy.deepcopy(self.routing)
                    runs = (static if source == 'static' else routing)['runs']
                    if shape == 'missing':
                        runs.pop()
                    elif shape == 'duplicate':
                        runs.append(dict(runs[0]))
                    else:
                        runs[0]['seed'] = 789
                    with self.assertRaisesRegex(ValueError, error):
                        export(static, routing)

    def test_truncation_cannot_shrink_expected_grid_through_metadata(self):
        self.static['methods'] = [m for m in self.static['methods'] if m['id'] != 'evoc']
        self.static['runs'] = [r for r in self.static['runs'] if r['method'] != 'evoc']
        with self.assertRaisesRegex(ValueError, 'method metadata'):
            export(self.static, self.routing)

    def test_rejects_in_progress_or_explicit_routing_failure(self):
        self.static['status'] = 'running'
        with self.assertRaisesRegex(ValueError, 'Only attempted final runs'):
            export(self.static, self.routing)
        self.static['status'] = 'executed'
        self.routing['runs'][0]['status'] = 'error'
        with self.assertRaisesRegex(ValueError, 'incomplete run'):
            export(self.static, self.routing)


if __name__ == '__main__':
    unittest.main()
