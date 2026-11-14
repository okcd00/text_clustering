import importlib.util
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_comparison_suite import METHODS, sample_digest, fit
from benchmark_lexical import scores

def test_method_information_budget():
 assert {m['id'] for m in METHODS if m['track']=='known-k'}=={'kmeans','minibatch_kmeans'}
 assert {m['id'] for m in METHODS if m['track']=='unknown-k-fixed'}=={'hdbscan','bertopic_pipeline','evoc'}

def test_order_fingerprint():
 a=[{'text':'甲'},{'text':'乙'}]
 assert sample_digest(a)!=sample_digest(a[::-1])
 assert sample_digest(a)==sample_digest([{'text':'甲','label':'changed'},{'text':'乙','label':'changed'}])

def test_noise_denominator():
 s=scores([0,0,1,1],[-1,-1,0,0])
 assert s['coverage']==.5 and s['clusters_excluding_noise']==1
 assert scores([0,1],[-1,-1])['ari_covered'] is None

def test_small_known_k():
 x=np.random.default_rng(0).normal(size=(40,4))
 for name in ['kmeans','minibatch_kmeans']:
  p=fit(name,x,2,0)
  assert p.labels.shape==(40,) and len(set(p.labels))==2

def test_unknown_k_ignores_privileged_k():
 x=np.random.default_rng(0).normal(size=(40,4))
 assert np.array_equal(fit('hdbscan',x,None,0).labels,fit('hdbscan',x,999,0).labels)
