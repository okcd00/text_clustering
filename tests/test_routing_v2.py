import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from benchmark_routing_v2 import metrics,predict_scores,calibrate,split_rows
from text_clustering.workflows.routing import OnePass,RoutingConfig
from text_clustering.index import ClusterIndex
from text_clustering.types import Document

def test_denominators():
    m=metrics(['a','a','hidden'],np.array(['a','b','a']),np.array([True,False,True]),{'a'})
    assert m['known_coverage']==.5 and m['known_accepted_accuracy']==1
    assert m['hidden_false_acceptance']==1 and m['known_correct_acceptance']==.5

def test_prototype_matches_existing_onepass():
    x=np.array([[1.,0.],[.8,.6],[0.,1.]])
    y=['a','a','b'];q=np.array([[.9,.43588989],[0.,1.]])
    q=q/np.linalg.norm(q,axis=1)[:,None]
    pred,best,margin,nz=predict_scores(x,y,q,'prototype')
    idx=ClusterIndex.from_assignments(['1','2','3'],x,y,'test')
    class Embedder:
        identity='test'
    router=OnePass(Embedder(),RoutingConfig(min_cosine=.2,min_margin=.05))
    for i,z in enumerate(q):
        decision=router._predict(Document(str(i),'synthetic'),idx,z)
        expected=best[i]>=.2 and margin[i]>=.05 and nz[i]
        assert (decision.status=='assigned')==expected
        if expected:assert decision.cluster_id==pred[i]

def test_zero_vector_abstains_and_no_accepted_accuracy():
    vals=predict_scores(np.eye(2),['a','b'],np.zeros((2,2)),'prototype')
    selected,_=calibrate(['a','hidden'],{'a'},vals)
    assert selected['known_accepted_accuracy'] is None
    assert selected['known_coverage']==0 and selected['hidden_false_acceptance']==0

def test_class_partition_and_cross_role_exclusion():
    train=[{'label':f'c{i}','text':f'example class {i} row {j}'} for i in range(10) for j in range(50)]
    test=[{'label':f'c{i}','text':f'test class {i} row {j}'} for i in range(10) for j in range(10)]
    support,val,t,known,vh,th=split_rows({'train':train,'test':test},'synthetic')
    assert not known&vh and not known&th and not vh&th
    assert all(r['label'] in known for r in support)
    assert not {r['text'] for r in support}&{r['text'] for r in val}
    assert all(r['label'] not in th for r in val)
