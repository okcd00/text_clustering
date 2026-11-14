import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import benchmark_semantic_pilot as b

def test_validation_grid_tie_is_lexicographic(monkeypatch):
    monkeypatch.setattr(b,'density',lambda z,p:np.zeros(len(z),dtype=int))
    params,trials=b.choose(np.ones((12,2)),['a']*6+['b']*6)
    assert params==[5,3] and len(trials)==6

def test_lexical_dimensions_finite_normalized():
    z=b.lexical([f'example sample number {i} topic {i%4}' for i in range(80)],0)
    assert z.shape==(80,64) and np.isfinite(z).all()
    assert np.allclose(np.linalg.norm(z,axis=1),1)

def test_network_disabled_on_import():
    import os
    assert os.environ['HF_HUB_OFFLINE']=='1'
    assert os.environ['TRANSFORMERS_OFFLINE']=='1'
