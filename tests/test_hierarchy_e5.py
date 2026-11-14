import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_hierarchy_e5 import fit,predict,calibrate,evaluate

def fixture():
    # Three support rows for L1, one for L2: parent must not become row-weighted.
    support=[{'leaf_id':'L1','parent_id':'P1'}]*3+[{'leaf_id':'L2','parent_id':'P1'},{'leaf_id':'L3','parent_id':'P2'}]
    nodes=[{'id':n,'label':n,'description':'','parent_id':p} for n,p in [('R000',None),('P1','R000'),('P2','R000'),('L1','P1'),('L2','P1'),('L3','P2')]]
    z=np.array([[1.,0],[1,0],[1,0],[0,1],[-1,0]],dtype=np.float32)
    return fit(z,{'support':support,'nodes':nodes})

def test_parent_equal_leaf_not_row_weighted_and_tie():
    m=fixture();np.testing.assert_allclose(m['parent'][0],[2**-.5,2**-.5],rtol=1e-6)
    p=predict(np.array([[1.,1.]],dtype=np.float32),m)
    assert p['leaf'][0]==0  # sorted opaque ID tie
    assert p['equal_leaf_mean'][0]==0

def test_calibration_rules_and_undefined_conditionals():
    # Constant scores tie on balanced accuracy; maximize known acceptance.
    t,c=calibrate([.5,.5],[True,False]);assert t==-1 and c['balanced_accuracy']==.5
    t,c=calibrate([.9,.8,.2,.1],[True,True,False,False]);assert t==.8 and c['balanced_accuracy']==1
    with pytest.raises(ValueError):calibrate([.4],[True])
    m=fixture();p=predict(np.array([[1.,0],[0,1]],dtype=np.float32),m)
    gold=[{'leaf_id':'H1','parent_id':'P1','is_known':False},{'leaf_id':'L2','parent_id':'P1','is_known':True}]
    r=evaluate(p,np.array([False,False]),gold,m)
    assert r['all']['parent_arms']['flat']['parent_accuracy_among_rejected_hidden']['value'] is None
    assert r['one_supported_child']['hidden_rejection_detection']['denominator']==0
    assert r['multiple_supported_children']['hidden_rows']==1
    assert r['all']['known']['acceptance_coverage']['numerator']==1

def test_method_does_not_accept_scorer_inputs():
    m={'support':[],'nodes':[],'test_gold':[]}
    with pytest.raises(AssertionError):fit(np.zeros((0,2)),m)

def test_parent_macro_and_joint_keep_all_hidden_denominator():
    m=fixture()
    p={'leaf':np.array([0,0,0,0]),'flat':np.array([0,0,0,0]),'equal_leaf_mean':np.array([0,0,0,0])}
    gold=[{'leaf_id':'H','parent_id':'P1','is_known':False}]*3+[{'leaf_id':'H2','parent_id':'P2','is_known':False}]
    r=evaluate(p,np.array([True,False,True,True]),gold,m)['all']
    assert r['hidden_rejection_detection']['numerator']==3
    for arm in r['parent_arms'].values():
        assert arm['all_hidden_parent_accuracy']['value']==.75
        assert arm['joint_reject_and_correct_parent']['value']==.5
        assert arm['parent_accuracy_among_rejected_hidden']['value']==2/3
        assert arm['parent_macro_all_hidden']['value']==.5
        assert arm['parent_macro_joint_all_hidden']['value']==1/3
        assert arm['parent_macro_joint_all_hidden']['parent_count']==2

def test_export_repeat_mismatch_fails_and_private_digest_removed(tmp_path):
    import json,copy
    from benchmark_hierarchy_e5 import ROOT
    from summarize_hierarchy_e5 import main
    existing=json.loads((ROOT/'results/hierarchy-e5-v1/results.json').read_text())
    prep=existing['preparation'];one=copy.deepcopy(existing['runs'][0]);one['private_prediction_digest']='synthetic-unit-test'
    for i in (1,2):(tmp_path/f'run-{i}.json').write_text(json.dumps(one))
    manifest={k:prep[k] for k in ['encode_seconds','truncated_rows','encoder_process_highwater_bytes','model_artifacts']}
    manifest.update(files={},preflight_verified=True,preflight=json.loads((ROOT/'experiments/hierarchy-preflight-v1/preflight.json').read_text()),protocol_sha256=one['protocol_sha256'])
    (tmp_path/'manifest.json').write_text(json.dumps(manifest));out=tmp_path/'out.json';main(tmp_path,out)
    assert 'private_prediction_digest' not in out.read_text()
    one['private_prediction_digest']='tampered';(tmp_path/'run-2.json').write_text(json.dumps(one))
    with pytest.raises(AssertionError):main(tmp_path,out)
