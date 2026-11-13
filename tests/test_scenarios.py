import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class ScenarioPublicationTest(unittest.TestCase):
    def test_evidence_inventory_and_missing_values(self):
        data = json.loads((ROOT / 'docs/scenarios.json').read_text())
        groups = data['groups']
        self.assertEqual(len(groups), 18)
        self.assertEqual(len({g['id'] for g in groups}), 18)
        self.assertEqual(sum(len(g['records']) for g in groups), 51)
        self.assertEqual(sum(len(r['seeds']) for g in groups for r in g['records']), 185)
        for g in groups:
            self.assertTrue(g['caveats'])
            self.assertEqual(len(g['source_digest']), 64)
        nli = groups[2]['records']
        self.assertIsNone(nli[0]['summary']['hidden_false_accept_rate']['sd'])
        self.assertGreater(nli[1]['summary']['hidden_false_accept_rate']['mean'],
                           nli[0]['summary']['hidden_false_accept_rate']['mean'])
        self.assertTrue(all(not r['seeds'] for g in groups[-2:] for r in g['records']))
        self.assertTrue(all(r.get('resources') for g in groups[-2:] for r in g['records']))

    def test_navigation_and_no_internal_payloads(self):
        html = (ROOT / 'docs/index.html').read_text()
        for task in ['onepass', 'hierarchical', 'evolution']:
            self.assertIn('scenarios.html?task=' + task, html)
        text = (ROOT / 'docs/scenarios.json').read_text()
        for marker in ['/' + 'Users/', 'alibaba' + '-inc.com', 'source_embedding_run', 'api_' + 'key']:
            self.assertNotIn(marker, text)
