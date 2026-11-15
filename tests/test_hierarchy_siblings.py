import sys, unittest, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_hierarchy_siblings import construct

def data():
    return {role:[{'text':f'{role} {p} {l} {i}', 'label':f'{p}-{l}', 'parent':str(p),'id':f'{role}-{p}-{l}-{i}'}
                  for p,n in enumerate([9,5,5,4,4,4,3]) for l in range(n) for i in range(10)]
            for role in ['train','validation','test']}
class SiblingPreflight(unittest.TestCase):
    def test_schema_caps_and_opaque(self):
        m,s,r,private=construct(data())
        self.assertEqual((r['supported_parent_count'],r['known_leaf_count'],r['validation_hidden_leaf_count'],r['test_hidden_leaf_count']),(6,13,6,12))
        self.assertTrue(all(v['validation_hidden_leaves']==1 and v['test_hidden_leaves']==2 for v in r['supported_parent_hidden_counts'].values()))
        self.assertEqual(r['roles']['test']['hidden_rows'],96)
        known={x['leaf_id'] for x in m['support']};vh={x['leaf_id'] for x in s['validation'] if not x['is_known']};th={x['leaf_id'] for x in s['test'] if not x['is_known']}
        self.assertFalse(known&vh or known&th or vh&th)
        self.assertTrue(all(n['id']==n['label'] and not n['description'] for n in m['nodes']))
        for role in r['roles']:
            self.assertEqual(sum(x['rows'] for x in r['post_cap_leaf_counts'][role]),r['roles'][role]['rows'])
        self.assertNotIn('source_id',str(r));self.assertEqual((m,s,r,private),construct(data()))
    def test_input_order_independent(self):
        d=data();a=construct(d)
        for rows in d.values():rows.reverse()
        self.assertEqual(a,construct(d))
    def test_dedup_and_role_ids(self):
        d=data();d['train'].append(dict(d['train'][0]));d['validation'][1]['text']=d['test'][1]['text']
        m,_,r,_=construct(d)
        self.assertNotIn(d['train'][0]['text'],[x['text'] for x in m['support']]);self.assertEqual(r['cross_role_normalized_overlap'],0)
        self.assertFalse(set(m['validation_texts'])&set(m['test_texts']))
    def test_committed_real_aggregate(self):
        root=Path(__file__).resolve().parents[1]
        r=json.loads((root/'experiments/hierarchy-sibling-preflight-v1/preflight.json').read_text())
        self.assertEqual([r['roles'][x]['rows'] for x in ['support','validation','test']],[256,143,191])
        self.assertEqual(r['roles']['test']['hidden_rows'],92)
        self.assertTrue(r['original_preflight_verified'])
        self.assertEqual(r['cross_role_normalized_overlap'],0)
        for role,rows in r['post_cap_leaf_counts'].items():
            self.assertEqual(sum(x['rows'] for x in rows),r['roles'][role]['rows'])
            self.assertTrue(all(set(x)=={'leaf_id','parent_id','is_known','rows'} for x in rows))
    def test_bad_parent(self):
        d=data();d['test'][0]['parent']='bad'
        with self.assertRaises(ValueError):construct(d)
