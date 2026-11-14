import sys
import importlib.util
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_hierarchy import construct

def data():
    result={}
    for split in ('train','validation','test'):
        result[split]=[{'text':f'{split} unique {p} {leaf} {i}', 'label':f'{p}_{leaf}',
                       'parent':p, 'id':f'{split}-{p}-{leaf}-{i}'}
                      for p in ('first','second') for leaf in ('a','b','c','d') for i in range(5)]
    return result

class HierarchyPreflight(unittest.TestCase):
    def test_actual_taxonomy_contract(self):
        path=Path(__file__).resolve().parents[1]/'src/text_clustering/workflows/navigation.py'
        spec=importlib.util.spec_from_file_location('_hierarchy_navigation_contract',path)
        nav=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=nav
        spec.loader.exec_module(nav)
        method,_,_=construct(data())
        taxonomy=nav.Taxonomy(nav.Node(**n) for n in method['nodes'])
        self.assertEqual(taxonomy.root_id,'R000')
        self.assertTrue(all(len(taxonomy.path(s['leaf_id']))==3 for s in method['support']))

    def test_supported_disjoint_and_opaque(self):
        method,scorer,r=construct(data())
        known={s['leaf_id'] for s in method['support']}
        vh={s['leaf_id'] for s in scorer['validation'] if not s['is_known']}
        th={s['leaf_id'] for s in scorer['test'] if not s['is_known']}
        self.assertFalse(known & vh or known & th or vh & th)
        self.assertEqual(r['unsupported_hidden_parents_after_caps'],0)
        self.assertTrue(all(n['id']==n['label'] and n['description']=='' for n in method['nodes']))
        self.assertTrue(all(n['id'][0] in 'LPR' for n in method['nodes']))
        self.assertEqual(sum(n['parent_id'] is None for n in method['nodes']),1)
        ids={n['id'] for n in method['nodes']}
        self.assertTrue(all(n['parent_id'] is None or n['parent_id'] in ids for n in method['nodes']))
        self.assertEqual((method,scorer,r),construct(data()))
        query=[s['leaf_id'] for s in scorer['test']]
        self.assertGreater(sum(a!=b for a,b in zip(query,query[1:])),len(set(query))-1)
    def test_dedup_does_not_choose_representative(self):
        rows=data()
        rows['train'].append(dict(rows['train'][0]))
        method,_,_=construct(rows)
        self.assertNotIn(rows['train'][0]['text'],[s['text'] for s in method['support']])
    def test_cross_role_overlap_removed(self):
        rows=data()
        rows['test'][0]['text']=rows['train'][0]['text']
        m,_,r=construct(rows)
        self.assertFalse(set(s['text'] for s in m['support']) & set(m['test_texts']))
        self.assertEqual(r['cross_role_normalized_overlap'],0)
    def test_multi_parent_rejected(self):
        rows=data();rows['test'][0]['parent']='wrong'
        with self.assertRaises(ValueError):construct(rows)

if __name__=='__main__':unittest.main()
