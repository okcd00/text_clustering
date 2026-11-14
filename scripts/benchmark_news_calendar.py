"""Deterministic calendar-snapshot BIRCH pilot; raw data/vectors stay outside Git."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import argparse, hashlib, json, time, zipfile, subprocess, sys, resource, platform, pickle
from pathlib import Path
import numpy as np
from prepare_news_calendar import prepare, parse, split, norm, external, SOURCE, PROTOCOL
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'experiments/calendar-paired-v1/protocol.json'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(cache):
    prepare(cache)
    s=json.loads(SOURCE.read_text());p=json.loads(PROTOCOL.read_text())
    with zipfile.ZipFile(cache/s['archive']['name']) as z: raw=parse(z.read(s['member']['name']))
    return split(raw,p)
def ordered_digest(windows):
    return hashlib.sha256(json.dumps([[r['id'],r['text'],r['time']] for _,_,_,rs in windows for r in rs],ensure_ascii=False).encode()).hexdigest()
def encode(cache,modeldir):
    from prepare_semantic_model import MANIFEST,verify
    for name,spec in json.loads(MANIFEST.read_text()).items(): verify(modeldir/name,spec)
    import torch
    from transformers import AutoTokenizer,AutoModel
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    windows=rows(cache);texts=[r['text'] for _,_,_,rs in windows for r in rs]
    tok=AutoTokenizer.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False)
    model=AutoModel.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False,use_safetensors=True).eval().float()
    start=time.perf_counter();vectors=[];trunc=0
    with torch.inference_mode():
        for i in range(0,len(texts),16):
            b=['query: '+x for x in texts[i:i+16]]
            trunc+=sum(len(x)>512 for x in tok(b,truncation=False)['input_ids'])
            x=tok(b,padding=True,truncation=True,max_length=512,return_tensors='pt')
            h=model(**x).last_hidden_state;m=x['attention_mask'].unsqueeze(-1)
            vectors.append(torch.nn.functional.normalize((h*m).sum(1)/m.sum(1),dim=1).numpy())
    z=np.concatenate(vectors);assert z.shape==(len(texts),384) and np.isfinite(z).all() and np.allclose(np.linalg.norm(z,axis=1),1,atol=1e-5)
    np.save(cache/'vectors.npy',z,allow_pickle=False)
    meta={'input_digest':ordered_digest(windows),'sha256':sha(cache/'vectors.npy'),'shape':list(z.shape),'dtype':str(z.dtype),'encode_seconds':time.perf_counter()-start,'truncated_rows':trunc,'model_artifacts':json.loads(MANIFEST.read_text()),'process_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    (cache/'vectors.meta.json').write_text(json.dumps(meta,indent=2)+'\n')
def ari(y,p):
    from sklearn.metrics import adjusted_rand_score
    return float(adjusted_rand_score(y,p)) if len(y)>1 else None
class Stream:
    def __init__(self,warmup,method,p):
        from sklearn.cluster import Birch
        self.new=lambda:Birch(threshold=p['threshold'],branching_factor=p['branching_factor'],n_clusters=None)
        self.model=self.new();self.method=method;self.arrivals=0;self.rebuilds=0;self.skipped=0;self.first_freeze=None
        self.buffer=[x.copy() for x in warmup]
        for x in warmup:self.model.partial_fit(x[None,:])
    def count(self):return len(self.model.subcluster_centers_)
    def update(self,x):
        self.arrivals+=1
        if self.method=='frozen':self.skipped+=1;return
        if self.method.startswith('cap-') and self.count()>=int(self.method.split('-')[1]):
            self.skipped+=1
            if self.first_freeze is None:self.first_freeze=self.arrivals
            return
        if self.method=='rolling-2048':
            self.buffer.append(x.copy());self.buffer=self.buffer[-2048:]
            if self.arrivals%2048==0:
                other=self.new()
                for v in self.buffer:other.partial_fit(v[None,:])
                self.model=other;self.rebuilds+=1;return
        self.model.partial_fit(x[None,:])
def snapshot_predict(state,x):
    # No learning is permitted inside this function, even within the same calendar day.
    return state.model.predict(x) if len(x) else np.array([],dtype=int)
def run(cache,method):
    import sklearn, importlib.metadata
    from threadpoolctl import threadpool_limits
    p=json.loads(P.read_text());windows=rows(cache);meta=json.loads((cache/'vectors.meta.json').read_text())
    assert meta['input_digest']==ordered_digest(windows) and meta['sha256']==sha(cache/'vectors.npy')
    z=np.load(cache/'vectors.npy',allow_pickle=False);assert z.shape==tuple(meta['shape'])
    from prepare_semantic_model import MANIFEST
    assert meta['model_artifacts']==json.loads(MANIFEST.read_text())
    assert z.dtype==np.float32 and z.shape[1]==384 and np.isfinite(z).all() and np.allclose(np.linalg.norm(z,axis=1),1,atol=1e-5)
    n=len(windows[0][3]);start=time.perf_counter()
    with threadpool_limits(limits=1):
        state=Stream(z[:n],method,p);warmtime=time.perf_counter()-start;warmcount=state.count()
        seen={norm(r['text']) for r in windows[0][3]};out=[];digest=hashlib.sha256()
        for date,left,right,rs in windows[1:]:
            x=z[n:n+len(rs)];n+=len(rs);before=state.count();t=time.perf_counter()
            pred=snapshot_predict(state,x)
            predict_seconds=time.perf_counter()-t;digest.update(pred.astype('<i8').tobytes())
            first=[]
            for j,r in enumerate(rs):
                k=norm(r['text'])
                if k not in seen:first.append(j)
                seen.add(k)
            scores={}
            for view,ix in [('all',list(range(len(rs)))),('first_title',first)]:
                scores[view]={'n':len(ix),**{key+'_ari':ari([rs[i][key] for i in ix],pred[ix]) for key in ['story','category']}}
            t=time.perf_counter();prevskip=state.skipped;prevbuild=state.rebuilds
            for v in x:state.update(v)
            update_seconds=time.perf_counter()-t
            out.append({'date':date,'n':len(rs),'leaf_cf_before':before,'leaf_cf_after':state.count(),'skipped_updates':state.skipped-prevskip,'rebuilds':state.rebuilds-prevbuild,'predict_seconds':predict_seconds,'update_seconds':update_seconds,'serialized_model_bytes':len(pickle.dumps(state.model,protocol=5)),'scores':scores})
    weighted={view:{key:sum(d['scores'][view][key]*d['scores'][view]['n'] for d in out if d['scores'][view][key] is not None)/sum(d['scores'][view]['n'] for d in out if d['scores'][view][key] is not None) for key in ['story_ari','category_ari']} for view in ['all','first_title']}
    return {'method':method,'status':'executed','warmup_leaf_cf':warmcount,'warmup_seconds':warmtime,'first_freeze_arrival':state.first_freeze,'total_rebuilds':state.rebuilds,'total_skipped_updates':state.skipped,'windows':out,'row_weighted_daily_scores':weighted,'private_prediction_digest':digest.hexdigest(),'process_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'versions':{x:importlib.metadata.version(x) for x in ['numpy','scipy','scikit-learn','threadpoolctl']},'python':platform.python_version()}
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--cache',required=True);a.add_argument('--model-dir');a.add_argument('--encode',action='store_true');a.add_argument('--method');a.add_argument('--output');v=a.parse_args();cache=external(v.cache)
    if v.encode:encode(cache,external(v.model_dir))
    elif v.method:external(v.output).write_text(json.dumps(run(cache,v.method),indent=2)+'\n')
    else: a.error('Choose --encode or --method')
