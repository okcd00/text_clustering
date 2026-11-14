import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_lexical import select_rows, scores

def r(text,label='a'): return {'text':text,'label':label}
def test_dedup_excludes_conflicting_groups_without_label_selection():
    rows={'train':[r('Prior')], 'validation':[r('DEV')], 'test':[r(' prior '),r('dev'),r('Dup','a'),r('DUP','b'),r('unique')]}
    assert select_rows(rows,True)==[r('unique')]
    assert len(select_rows(rows,False))==5

def test_noise_and_degenerate_control():
    s=scores(['a','a','b','b'],[-1,-1,-1,-1])
    assert s['coverage']==0 and s['ari_covered'] is None and s['clusters_excluding_noise']==0
    assert scores(['a','a','b','b'],[0,0,0,0])['ari_all_noise_as_one_cluster']==0
