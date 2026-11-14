import importlib.util, unittest, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('prep',ROOT/'scripts/prepare_datasets.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def row(text,ident=None,label='a',parent=None):return dict(text=text,id=ident,label=label,parent=parent)
class DatasetPreparation(unittest.TestCase):
    def test_no_repo_payload(self):
        with self.assertRaises(ValueError):m.external(ROOT/'datasets'/'cache')
    def test_exact_and_normalized_leakage(self):
        r=m.audit({'train':[row('Hello'),row('hello ')],'test':[row('Hello'),row('HELLO')]})
        self.assertEqual(r['cross_split']['train__test']['exact_shared_texts'],1)
        self.assertEqual(r['cross_split']['train__test']['normalized_shared_texts'],1)
        self.assertEqual(r['splits']['train']['normalized_duplicate_excess'],1)
    def test_id_and_parent_conflict_reported(self):
        r=m.audit({'train':[row('x','1',parent='p')],'test':[row('y','1',parent='q')]})
        self.assertEqual(r['cross_split']['train__test']['shared_ids'],1)
        self.assertEqual(r['hierarchy']['multi_parent_children'],1)
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):m.audit({'train':[row(' ')]})

    def test_modified_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'train.csv').write_text('text,category\nwrong,a\n')
            with self.assertRaisesRegex(ValueError,'hash/size mismatch'):
                m.prepare('banking77',temp)
