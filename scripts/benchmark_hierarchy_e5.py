"""Two frozen parent-only arms with one validation-calibrated leaf rejection mask."""
import argparse, hashlib, importlib.metadata, json, platform, resource, time
from collections import Counter
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def normalize(x):
    d=np.linalg.norm(x,axis=-1,keepdims=True)
    if np.any(d==0): raise ValueError('Zero prototype')
    return x/d

def fit(z, method):
    # Only support annotations and known topology are accepted here.
    assert set(method)=={'support','nodes'}
    support=method['support'];nodes={n['id']:n for n in method['nodes']}
    assert all(n['id']==n['label'] and n['description']=='' for n in nodes.values())
    leaves=sorted({s['leaf_id'] for s in support});parent_of={s['leaf_id']:s['parent_id'] for s in support}
    assert all(nodes[l]['parent_id']==parent_of[l] for l in leaves)
    parents=sorted(set(parent_of.values()));leaf_parent=np.array([parents.index(parent_of[l]) for l in leaves])
    leaf=normalize(np.stack([z[[i for i,s in enumerate(support) if s['leaf_id']==l]].mean(0) for l in leaves]))
    parent=normalize(np.stack([leaf[leaf_parent==i].mean(0) for i in range(len(parents))]))
    return {'leaves':leaves,'parents':parents,'leaf':leaf,'parent':parent,'leaf_parent':leaf_parent,'children':Counter(parent_of.values())}

def predict(z, model):
    similarities=z@model['leaf'].T;winner=similarities.argmax(1)
    return {'leaf':winner,'score':similarities[np.arange(len(z)),winner],
            'flat':model['leaf_parent'][winner], 'equal_leaf_mean':(z@model['parent'].T).argmax(1)}

def calibrate(scores, known):
    scores=np.asarray(scores,dtype=np.float64);known=np.asarray(known,dtype=bool)
    nk=int(known.sum());nh=len(known)-nk
    if not nk or not nh: raise ValueError('Calibration requires known and hidden')
    candidates=np.unique(np.r_[-1.,scores,np.nextafter(scores.max(),np.inf)])
    def key(t):
        reject=scores<t;ka=int((~reject&known).sum());hr=int((reject&~known).sum())
        return (ka*nh+hr*nk,ka,-float(t))
    t=float(max(candidates,key=key));rej=scores<t
    return t,{'known_n':nk,'hidden_n':nh,'known_accepted_n':int((~rej&known).sum()),'hidden_rejected_n':int((rej&~known).sum()),
              'balanced_accuracy':key(t)[0]/(2*nk*nh),'candidate_count':len(candidates)}

def fraction(num, den): return {'numerator':int(num),'denominator':int(den),'value':float(num/den) if den else None}
def macro(values, mask, goldparents):
    groups=sorted(set(goldparents[mask].tolist()))
    return {'parent_count':len(groups),'value':float(np.mean([values[mask&(goldparents==p)].mean() for p in groups])) if groups else None}

def evaluate(pred, rejected, gold, model):
    known=np.array([g['is_known'] for g in gold]);gp=np.array([model['parents'].index(g['parent_id']) for g in gold]);gl=np.array([g['leaf_id'] for g in gold])
    leaflabels=np.array(model['leaves'])[pred['leaf']];correctleaf=leaflabels==gl
    childn=np.array([model['children'][g['parent_id']] for g in gold]);out={}
    for group,mask in [('all',np.ones(len(gold),dtype=bool)),('one_supported_child',childn==1),('multiple_supported_children',childn>1)]:
        kh=mask&known;hh=mask&~known;kr=kh&rejected;hr=hh&rejected;accepted=kh&~rejected
        data={'rows':int(mask.sum()),'known_rows':int(kh.sum()),'hidden_rows':int(hh.sum()),
              'known':{'leaf_accuracy':fraction((correctleaf&kh).sum(),kh.sum()),'acceptance_coverage':fraction(accepted.sum(),kh.sum()),
                       'false_novelty':fraction(kr.sum(),kh.sum()),'accepted_leaf_accuracy':fraction((correctleaf&accepted).sum(),accepted.sum()),
                       'accepted_and_correct_all_known':fraction((correctleaf&accepted).sum(),kh.sum())},
              'hidden_rejection_detection':fraction(hr.sum(),hh.sum()),'parent_arms':{}}
        for arm in ['flat','equal_leaf_mean']:
            hit=pred[arm]==gp;joint=hit&rejected
            data['parent_arms'][arm]={'all_hidden_parent_accuracy':fraction((hit&hh).sum(),hh.sum()),
                'joint_reject_and_correct_parent':fraction((joint&hh).sum(),hh.sum()),
                'parent_accuracy_among_rejected_hidden':fraction((hit&hr).sum(),hr.sum()),
                'parent_macro_all_hidden':macro(hit,hh,gp),'parent_macro_joint_all_hidden':macro(joint,hh,gp),
                'parent_macro_among_rejected_hidden':macro(hit,hr,gp),
                'gold_hidden_parent_count':len(set(gp[hh].tolist())), 'gold_known_parent_count':len(set(gp[kh].tolist()))}
        out[group]=data
    return out

def memory():
    return {s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/self/status').read_text().splitlines() if s.startswith(('VmRSS:','RssAnon:','RssFile:','RssShmem:'))}

def main(cache):
    meta=json.loads((cache/'manifest.json').read_text());assert meta['protocol_sha256']==sha(ROOT/'experiments/hierarchy-e5-v1/protocol.json')
    for n,h in meta['files'].items(): assert sha(cache/n)==h
    z={r:np.load(cache/f'{r}.npy',allow_pickle=False) for r in ['support','validation','test']}
    method=json.loads((cache/'method.json').read_text())
    baseline=memory();timing={};t=time.perf_counter();model=fit(z['support'],method);timing['fit_seconds']=time.perf_counter()-t
    t=time.perf_counter();val=predict(z['validation'],model);timing['validation_predict_seconds']=time.perf_counter()-t
    # Calibration sees binary known/hidden only, never parents or test outcomes.
    validation_gold=json.loads((cache/'validation-scorer.json').read_text());t=time.perf_counter()
    threshold,calibration=calibrate(val['score'],[r['is_known'] for r in validation_gold]);timing['calibration_seconds']=time.perf_counter()-t
    t=time.perf_counter();test=predict(z['test'],model);timing['test_predict_seconds']=time.perf_counter()-t
    # Freeze every prediction/mask into a digest BEFORE any test gold evaluation.
    predictions={'validation':val,'test':test};rejected={r:predictions[r]['score'].astype(np.float64)<threshold for r in predictions}
    h=hashlib.sha256()
    for r in ['validation','test']:
        for k in ['leaf','score','flat','equal_leaf_mean']:h.update(predictions[r][k].tobytes())
        h.update(rejected[r].tobytes())
    frozen_digest=h.hexdigest()
    scorer={'validation':validation_gold,'test':json.loads((cache/'test-scorer.json').read_text())};t=time.perf_counter()
    result={r:evaluate(predictions[r],rejected[r],scorer[r],model) for r in ['validation','test']};timing['scoring_seconds']=time.perf_counter()-t
    return {'status':'executed','protocol_sha256':meta['protocol_sha256'],'threshold':threshold,'calibration':calibration,'roles':result,
            'known_leaf_count':len(model['leaves']),'supported_parent_count':len(model['parents']),'private_prediction_digest':frozen_digest,
            'timing':timing,'memory':{'post_input_load_baseline':baseline,'end':memory(),'lifetime_process_highwater_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                                    'scope':'source-free, encoder-free worker; highwater not reset at baseline, no continuous sampling or owned-heap claim'},
            'python':platform.python_version(),'versions':{n:importlib.metadata.version(n) for n in ['numpy','threadpoolctl']}}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    for path in [a.cache.resolve(),a.output.resolve()]:
        if path==ROOT or ROOT in path.parents: raise ValueError('Worker data/output must be external')
    with threadpool_limits(limits=1): result=main(a.cache)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
