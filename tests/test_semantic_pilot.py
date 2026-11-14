import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_semantic_pilot import sample, protocol_rows

def test_selection_is_deterministic_and_label_blind():
    a=[{'text':str(i),'label':'a'} for i in range(600)]
    b=[dict(r,label='b') for r in a]
    assert [r['text'] for r in sample(a,'d','test')]==[r['text'] for r in sample(b,'d','test')]
    assert len(sample(a,'d','test'))==512

def test_validation_excludes_normalized_test_overlap():
    r=protocol_rows({'train':[{'text':' TEST ','label':'a'},{'text':'dev','label':'b'}], 'test':[{'text':'test','label':'a'}]},'d')
    assert r['validation']==[{'text':'dev','label':'b'}]
