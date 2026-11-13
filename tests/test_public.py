import json
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np
from text_clustering import Document
from text_clustering.embeddings import HashingEmbedder
from text_clustering.factory import build_tool

ROOT=Path(__file__).resolve().parents[1]
class PublicSmoke(unittest.TestCase):
    def test_examples(self):
        p=subprocess.run([sys.executable,str(ROOT/'examples/offline_demo.py')],capture_output=True,text=True,check=True)
        self.assertIsInstance(json.loads(p.stdout),dict)
    def test_precomputed(self):
        e=HashingEmbedder()
        docs=[Document(str(i),t) for i,t in enumerate(['refund payment','payment refund','reset password','password reset'])]
        for name in ('kmeans','minibatch_kmeans'):
            tool=build_tool('partitioning',e,algorithm={'name':name,'n_clusters':2,'seed':0})
            result=tool.fit(docs,vectors=e.encode([d.text for d in docs]),encoder_identity=e.identity)
            self.assertEqual(len(result.partition.labels),4)
            with self.assertRaises(ValueError):tool.fit(docs,vectors=np.ones((4,8)),encoder_identity='different')
    def test_results(self):
        data=json.loads((ROOT/'docs/results.json').read_text())
        self.assertEqual(len(data['records']),17)
        self.assertEqual(sum(len(r['seeds']) for r in data['records']),85)
        for r in data['records']:
            self.assertEqual([s['seed'] for s in r['seeds']],list(range(5)))
            for key,aggregate in r['summary'].items():
                if not isinstance(aggregate,dict) or 'mean' not in aggregate:continue
                values=[s.get(key,s.get('metrics',{}).get(key)) for s in r['seeds']]
                if all(isinstance(v,(int,float)) for v in values):self.assertAlmostEqual(aggregate['mean'],sum(values)/5)
if __name__=='__main__':unittest.main()
