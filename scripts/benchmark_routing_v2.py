"""Reproducible supervised routing/abstention, not batch clustering.
Train-fitted lexical features; no model or API; aggregate public output only.
"""
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as unit
from threadpoolctl import threadpool_limits
from prepare_datasets import prepare, read_rows, normalize, external

PROTOCOL='routing-v2-lexical-heldout-labels'
SEEDS=[0,1,2]
GRID=[(c,m) for c in [0.,.1,.2,.3,.4,.5,.6,.7,.8,.9] for m in [0.,.05,.1]]
def ranked(items, namespace):
    return [r for _,r in sorted(enumerate(items),key=lambda p:hashlib.sha256(f'{PROTOCOL}|{namespace}|{p[0]}'.encode()).digest())]
def split_rows(rows,dataset):
    labels=sorted({r['label'] for r in rows['train']},key=lambda l:hashlib.sha256(f'{PROTOCOL}|{dataset}|{l}'.encode()).digest())
    n=len(labels);a=int(n*.6);b=int(n*.8)
    known=set(labels[:a]);vh=set(labels[a:b]);th=set(labels[b:])
    if 'validation' in rows: support_source=rows['train'];val_source=rows['validation']
    else:
        ordered=ranked(rows['train'],dataset+'-train');cut=int(.7*len(ordered))
        support_source=ordered[:cut];val_source=ordered[cut:]
    test=ranked([r for r in rows['test'] if r['label'] in known|th],dataset+'-test')[:512]
    # Exclude against FULL test corpus, not just sampled rows.
    test_text={normalize(r['text']) for r in rows['test']}
    val=ranked([r for r in val_source if r['label'] in known|vh and normalize(r['text']) not in test_text],dataset+'-validation')[:512]
    val_text={normalize(r['text']) for r in val_source}
    candidates=[r for r in support_source if r['label'] in known and normalize(r['text']) not in test_text|val_text]
    # On BANKING support_source and val_source are disjoint portions.
    support=[]
    for label in sorted(known): support+=ranked([r for r in candidates if r['label']==label],dataset+'-support-'+label)[:20]
    missing=known-{r['label'] for r in support}
    if missing:raise ValueError(f'Known labels without support: {sorted(missing)}')
    for left,right in [(support,val),(support,test),(val,test)]:
        assert not {normalize(r['text']) for r in left}&{normalize(r['text']) for r in right}
    return support,val,test,known,vh,th

def metrics(y,pred,accept,known):
    y=np.array(y);known_mask=np.array([v in known for v in y]);hidden=~known_mask
    ka=known_mask&accept;correct=(y==pred)&ka
    return {'n':len(y),'known_n':int(known_mask.sum()),'hidden_n':int(hidden.sum()),'known_accepted_n':int(ka.sum()),'known_correct_accepted_n':int(correct.sum()),'hidden_accepted_n':int((hidden&accept).sum()),'known_coverage':float(ka.sum()/known_mask.sum()),'known_accepted_accuracy':float(correct.sum()/ka.sum()) if ka.any() else None,'known_correct_acceptance':float(correct.sum()/known_mask.sum()),'hidden_false_acceptance':float((hidden&accept).sum()/hidden.sum())}
def predict_scores(support,y,z,method):
    ids=sorted(set(y));y=np.array(y)
    if method=='prototype':
        centers=unit(np.array([support[y==label].sum(axis=0) for label in ids]));scores=z@centers.T
    elif method=='class-max-1nn':
        sim=z@support.T;scores=np.column_stack([sim[:,y==label].max(axis=1) for label in ids])
    else:raise ValueError(method)
    # Stable class-name tie breaking. Zero representation always rejected.
    order=np.argsort(-scores,axis=1,kind='stable');best=scores[np.arange(len(z)),order[:,0]];margin=best-scores[np.arange(len(z)),order[:,1]]
    return np.array(ids)[order[:,0]],best,margin,np.linalg.norm(z,axis=1)>1e-12

def calibrate(y,known,values):
    pred,best,margin,nonzero=values;trials=[]
    for c,m in GRID:
        stats=metrics(y,pred,(best>=c)&(margin>=m)&nonzero,known)
        objective=(stats['known_correct_acceptance']+1-stats['hidden_false_acceptance'])/2
        trials.append({'min_cosine':c,'min_margin':m,'objective':objective,**stats})
    return min(trials,key=lambda r:(-r['objective'],r['min_cosine'],r['min_margin'])),trials

def run(cache):
    import sklearn,scipy
    result={'schema_version':1,'protocol':PROTOCOL,'scene':'routing','status':'executed','representation':'train-only char TF-IDF(2,5)/SVD64/L2','information_budget':'labeled known-class support plus labeled validation-known/validation-hidden; test-hidden labels never used in fitting or calibration','selection_objective':'mean of known correct-acceptance and hidden rejection on validation; ties ascending cosine then margin','threads':1,'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__},'inputs':{},'runs':[],'validation_trials':[]}
    for dataset in ['banking77','massive_zh_cn']:
        audit=prepare(dataset,cache/dataset);support,val,test,known,vh,th=split_rows(read_rows(dataset,cache/dataset),dataset)
        result['inputs'][dataset]={'source_revision':audit['source_revision'],'artifacts':audit['artifacts'],'support_n':len(support),'validation_n':len(val),'test_n':len(test),'known_classes':len(known),'validation_hidden_classes':len(vh),'test_hidden_classes':len(th),'cross_role_normalized_overlap':0,'sample_sha256':hashlib.sha256(json.dumps([[r['text'] for r in part] for part in [support,val,test]],ensure_ascii=False).encode()).hexdigest(),'support_per_class_cap':20,'evaluation_role_cap':512,'class_split_sha256':hashlib.sha256(json.dumps([sorted(known),sorted(vh),sorted(th)]).encode()).hexdigest()}
        for seed in SEEDS:
            start=time.perf_counter();tf=TfidfVectorizer(analyzer='char',ngram_range=(2,5),max_features=10000,min_df=2,sublinear_tf=True,dtype=np.float64)
            xs=tf.fit_transform([r['text'] for r in support]);svd=TruncatedSVD(n_components=64,n_iter=7,random_state=seed)
            zs=unit(svd.fit_transform(xs));zv=unit(svd.transform(tf.transform([r['text'] for r in val])));zt=unit(svd.transform(tf.transform([r['text'] for r in test])))
            feature_time=time.perf_counter()-start;ys=[r['label'] for r in support];yv=[r['label'] for r in val];yt=[r['label'] for r in test]
            for method in ['prototype','class-max-1nn']:
                start=time.perf_counter();values=predict_scores(zs,ys,zv,method);selected,trials=calibrate(yv,known,values);cal_time=time.perf_counter()-start
                result['validation_trials'].append({'dataset':dataset,'seed':seed,'method':method,'trials':trials})
                start=time.perf_counter();pred,best,margin,nz=predict_scores(zs,ys,zt,method);accept=(best>=selected['min_cosine'])&(margin>=selected['min_margin'])&nz
                elapsed=time.perf_counter()-start
                result['runs'].append({'dataset':dataset,'seed':seed,'method':method,'min_cosine':selected['min_cosine'],'min_margin':selected['min_margin'],'feature_seconds_shared':feature_time,'validation_seconds':cal_time,'test_routing_seconds':elapsed,'zero_test_vectors':int((~nz).sum()),**metrics(yt,pred,accept,known)})
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    with threadpool_limits(limits=1):result=run(external(a.cache))
    out=external(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('12 aggregate routing runs saved')
