"""Fresh, source-free worker for calendar maintenance with sampled process memory."""
import argparse, gc, hashlib, importlib.metadata, json, os, pickle, platform, resource, threading, time
from collections import deque
from pathlib import Path
import numpy as np
from sklearn.cluster import Birch
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'experiments/calendar-paired-v2/protocol.json'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def external(p):
    p=Path(p).resolve()
    if p==ROOT or ROOT in p.parents:raise ValueError('Cache and worker outputs must stay outside repo')
    return p
def memory():
    fields={}
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.split(':',1)[0] in {'VmRSS','RssAnon','RssFile','RssShmem'}:
            name,v=line.split(':',1);fields[name]=int(v.split()[0])*1024
    if set(fields)!={'VmRSS','RssAnon','RssFile','RssShmem'}:raise RuntimeError('Required Linux memory counters unavailable')
    return fields
class Sampler:
    def __init__(self):
        self.lock=threading.Lock();self.phase=None;self.peaks={};self.stop=threading.Event();self.samples=0
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def loop(self):
        while not self.stop.wait(.005):self.probe()
    def probe(self):
        v=memory()
        with self.lock:
            if self.phase:
                p=self.peaks.setdefault(self.phase,{k:0 for k in v})
                for k in v:p[k]=max(p[k],v[k])
                self.samples+=1
        return v
    def set_phase(self,name):
        self.probe()
        with self.lock:self.phase=name
        self.probe()
    def finish(self):
        self.probe();self.stop.set();self.thread.join()
class Stream:
    def __init__(self,warmup,method,p,probe):
        self.new=lambda:Birch(threshold=p['threshold'],branching_factor=p['branching_factor'],n_clusters=None)
        self.model=self.new();self.method=method;self.arrivals=0;self.skipped=0;self.first_freeze=None;self.events=[];self.probe=probe
        self.buffer=deque((x.copy() for x in warmup),maxlen=2048) if method=='rolling-2048' else None
        for x in warmup:self.model.partial_fit(x[None,:])
    def count(self):return len(self.model.subcluster_centers_)
    def update(self,x):
        self.arrivals+=1
        if self.method=='frozen':self.skipped+=1;return
        if self.method=='cap-2048' and self.count()>=2048:
            self.skipped+=1
            if self.first_freeze is None:self.first_freeze=self.arrivals
            return
        if self.buffer is not None:
            self.buffer.append(x.copy())
            if self.arrivals%2048==0:
                before=self.count();other=self.new()
                for v in self.buffer:other.partial_fit(v[None,:])
                self.events.append({'arrival':self.arrivals,'old_leaf_cf':before,'new_leaf_cf':len(other.subcluster_centers_),'overlap_memory':self.probe()})
                self.model=other;self.probe();return
        self.model.partial_fit(x[None,:])
def score(y,p):return float(adjusted_rand_score(y,p)) if len(y)>1 else None
def main(cache,method):
    protocol=json.loads(P.read_text());assert method in protocol['methods']
    meta=json.loads((cache/'manifest.json').read_text());assert meta['protocol_sha256']==sha(P)
    # Byte hashing uses bounded blocks before measured baseline, not full-size byte buffers.
    for n,spec in meta['files'].items():assert (cache/n).stat().st_size==spec['bytes'] and sha(cache/n)==spec['sha256']
    imports_memory=memory();arrays={n:np.load(cache/n,mmap_mode='r',allow_pickle=False) for n in meta['files']}
    z=arrays['vectors.npy'];ys=arrays['story.npy'];yc=arrays['category.npy'];first=arrays['first_title.npy']
    assert list(z.shape)==meta['shape'] and z.dtype==np.float32 and z.shape[1]==384
    assert all(len(a)==len(z) for a in arrays.values())
    mapped_before_touch=memory()
    # Bounded touches visit all file-backed pages without materializing full arrays.
    checksums={}
    for n,a in arrays.items():
        flat=a.reshape(-1);checksums[n]=sum(float(flat[i:i+262144].sum()) for i in range(0,len(flat),262144))
    touched_memory=memory()
    # Resolve lazy clustering/metric/BLAS state on unrelated numeric dummy data before baseline.
    dummy=Birch(n_clusters=None).fit(np.array([[0.,0.],[1.,1.]],dtype=np.float32));dummy.predict(np.array([[0.,0.]],dtype=np.float32));score([0,1],[0,1]);del dummy;gc.collect()
    sampler=Sampler();sampler.probe();baseline=memory();sampler.set_phase('warmup')
    t=time.perf_counter();n0=meta['windows'][0]['stop'];state=Stream(z[:n0],method,protocol,sampler.probe);warm_seconds=time.perf_counter()-t
    warm_memory=sampler.probe();warm_count=state.count();out=[];digest=hashlib.sha256();prefix_digest=None
    for win in meta['windows'][1:]:
        a,b=win['start'],win['stop'];x=z[a:b];date=win['date'];before=state.count()
        sampler.set_phase('predict');t=time.perf_counter();pred=state.model.predict(x) if len(x) else np.array([],dtype=np.int64);prediction_seconds=time.perf_counter()-t
        digest.update(pred.astype('<i8').tobytes())
        if date=='2014-03-30':prefix_digest=digest.hexdigest()
        sampler.set_phase('score')
        scored={}
        for view,mask in [('all',np.ones(len(x),dtype=bool)),('first_title',first[a:b])]:
            scored[view]={'n':int(mask.sum()),'story_ari':score(ys[a:b][mask],pred[mask]),'category_ari':score(yc[a:b][mask],pred[mask])}
        sampler.set_phase('update');t=time.perf_counter();skipped=state.skipped;rebuilds=len(state.events)
        for v in x:state.update(v)
        update_seconds=time.perf_counter()-t;after_memory=sampler.probe()
        out.append({'date':date,'n':len(x),'leaf_cf_before':before,'leaf_cf_after':state.count(),'prediction_seconds':prediction_seconds,'update_seconds':update_seconds,'scores':scored,'skipped_updates':state.skipped-skipped,'rebuilds':len(state.events)-rebuilds,'after_update_memory':after_memory})
    sampler.set_phase('final_idle');gc.collect();final_memory=sampler.probe();sampler.finish()
    phase_peaks=sampler.peaks
    algorithm_peak={k:max(phase_peaks[phase][k] for phase in ['warmup','predict','update']) for k in baseline}
    peak_all={k:max(d[k] for d in phase_peaks.values()) for k in baseline}
    highwater=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    # Storage accounting comes after all memory observations to avoid counting pickle buffers.
    serialized=len(pickle.dumps(state.model,protocol=5))
    return {'status':'executed','method':method,'protocol_sha256':sha(P),'warmup_leaf_cf':warm_count,'warmup_seconds':warm_seconds,'first_freeze_arrival':state.first_freeze,'rebuild_events':state.events,'total_skipped_updates':state.skipped,'windows':out,'private_prediction_digest':digest.hexdigest(),'private_prediction_digest_v1_prefix':prefix_digest,'serialized_final_model_bytes':serialized,'memory':{'imports':imports_memory,'mapped_before_touch':mapped_before_touch,'after_touch':touched_memory,'baseline_after_touch_api_warmup_sampler_start':baseline,'after_warmup':warm_memory,'phase_peaks':phase_peaks,'algorithm_phase_peak':algorithm_peak,'algorithm_peak_delta_from_baseline':{k:algorithm_peak[k]-baseline[k] for k in baseline},'all_phase_peak':peak_all,'final_after_gc':final_memory,'resource_highwater_before_serialization_bytes':highwater,'sample_interval_ms':5,'samples':sampler.samples,'mapped_file_bytes':sum(x['bytes'] for x in meta['files'].values())},'python':platform.python_version(),'versions':{x:importlib.metadata.version(x) for x in ['numpy','scipy','scikit-learn','threadpoolctl']},'hardware':{'machine':platform.machine(),'logical_cpus':os.cpu_count(),'processor':next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'unknown')}}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--method',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    with threadpool_limits(limits=1):result=main(external(a.cache),a.method)
    external(a.output).write_text(json.dumps(result,indent=2)+'\n')
