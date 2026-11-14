"""Bounded local-only semantic/lexical paired pilot. See experiment protocol."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse, hashlib, importlib.metadata, json, platform, resource, time
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2
from threadpoolctl import threadpool_limits
from prepare_datasets import prepare, read_rows, external
from prepare_semantic_pilot import protocol_rows, describe, MODEL, REVISION
from benchmark_lexical import scores
from prepare_semantic_model import verify, MANIFEST

SEEDS=[0,1,2]
GRID=[(m,s) for m in [5,10,20] for s in [3,5]]
def density(z, params):
    return HDBSCAN(min_cluster_size=params[0], min_samples=params[1],metric='euclidean',cluster_selection_method='eom',n_jobs=1,copy=True).fit_predict(z)
def lexical(texts,seed):
    x=TfidfVectorizer(analyzer='char',ngram_range=(2,5),min_df=2,max_features=10000,sublinear_tf=True,lowercase=True,dtype=np.float64).fit_transform(texts)
    return l2(TruncatedSVD(n_components=64,n_iter=7,random_state=seed).fit_transform(x))
def choose(z,labels):
    trials=[]
    for params in GRID:
        start=time.perf_counter(); pred=density(z,params)
        trials.append({'params':list(params),**scores(labels,pred),'seconds':time.perf_counter()-start})
    winner=min(trials,key=lambda r:(-r['ari_all_noise_as_one_cluster'],r['params']))
    return winner['params'],trials

def run(cache,model_dir):
    for name,spec in json.loads(MANIFEST.read_text()).items():verify(model_dir/name,spec)
    import torch
    from transformers import AutoModel,AutoTokenizer
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    start=time.perf_counter()
    tok=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False)
    model=AutoModel.from_pretrained(str(model_dir),local_files_only=True,trust_remote_code=False,use_safetensors=True).eval().float()
    result={'protocol':'semantic-pilot-v1','status':'executed','model':MODEL,'revision':REVISION,'threads':1,'model_load_seconds':time.perf_counter()-start,'versions':{'python':platform.python_version(),**{n:importlib.metadata.version(n) for n in ['torch','transformers','tokenizers','safetensors','numpy','scipy','scikit-learn','pyarrow','threadpoolctl']}},'model_artifacts':{p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(model_dir.iterdir()) if p.is_file()},'scripts':{n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in ['benchmark_semantic_pilot.py','prepare_semantic_pilot.py','prepare_datasets.py','benchmark_lexical.py']},'inputs':{},'representations':[],'validation_trials':[],'runs':[],'hardware':{'machine':platform.machine(),'logical_cpus':os.cpu_count(),'processor':next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown')}}
    def encode(texts):
        start=time.perf_counter();vectors=[];truncated=0
        with torch.inference_mode():
            for i in range(0,len(texts),16):
                batch=['query: '+t for t in texts[i:i+16]]
                truncated+=sum(len(ids)>512 for ids in tok(batch,truncation=False)['input_ids'])
                inp=tok(batch,max_length=512,padding=True,truncation=True,return_tensors='pt')
                hidden=model(**inp).last_hidden_state
                mask=inp['attention_mask'].unsqueeze(-1)
                pooled=(hidden*mask).sum(1)/mask.sum(1)
                vectors.append(torch.nn.functional.normalize(pooled,p=2,dim=1).numpy())
        z=np.concatenate(vectors)
        assert z.shape==(len(texts),384) and np.isfinite(z).all() and np.allclose(np.linalg.norm(z,axis=1),1,atol=1e-5)
        return z,{'seconds':time.perf_counter()-start,'truncated':truncated,'dtype':str(z.dtype),'dimensions':384}
    for dataset in ['banking77','massive_zh_cn']:
        audit=prepare(dataset,cache/dataset)
        parts=protocol_rows(read_rows(dataset,cache/dataset),dataset)
        result['inputs'][dataset]={'source_revision':audit['source_revision'],'artifacts':audit['artifacts'],'splits':{k:describe(v) for k,v in parts.items()}}
        texts={k:[r['text'] for r in v] for k,v in parts.items()};labels={k:[r['label'] for r in v] for k,v in parts.items()}
        sem={}
        for role in parts:
            sem[role],meta=encode(texts[role])
            vector_path=cache/f'{dataset}-{role}-e5.npy'
            np.save(vector_path,sem[role],allow_pickle=False)
            private_meta={'model':MODEL,'revision':REVISION,'rows':len(texts[role]),'ordered_texts_sha256':hashlib.sha256(json.dumps(texts[role],ensure_ascii=False).encode()).hexdigest(),'vector_file_sha256':hashlib.sha256(vector_path.read_bytes()).hexdigest()}
            vector_path.with_suffix('.meta.json').write_text(json.dumps(private_meta,indent=2)+'\n')
            result['representations'].append({'dataset':dataset,'representation':'e5','role':role,'seed':None,**meta})
        for representation in ['lexical','e5']:
            for seed in SEEDS:
                zs={}
                for role in parts:
                    if representation=='e5':zs[role]=sem[role]
                    else:
                        t=time.perf_counter();zs[role]=lexical(texts[role],seed)
                        result['representations'].append({'dataset':dataset,'representation':representation,'role':role,'seed':seed,'seconds':time.perf_counter()-t,'dimensions':64,'dtype':str(zs[role].dtype)})
                params,trials=choose(zs['validation'],labels['validation'])
                result['validation_trials'].append({'dataset':dataset,'representation':representation,'seed':seed,'selected':params,'trials':trials})
                # Parameters are frozen above, before any test scoring.
                y=labels['test'];z=zs['test']
                for method in ['one-cluster','kmeans-oracle-k','hdbscan-fixed','hdbscan-validation-tuned']:
                    t=time.perf_counter()
                    if method=='one-cluster':pred=np.zeros(len(y),dtype=int)
                    elif method=='kmeans-oracle-k':pred=KMeans(n_clusters=len(set(y)),random_state=seed,n_init=10,max_iter=300).fit_predict(z)
                    else:pred=density(z,(10,5) if method=='hdbscan-fixed' else params)
                    elapsed=time.perf_counter()-t
                    result['runs'].append({'dataset':dataset,'representation':representation,'seed':seed,'method':method,'n':len(y),'covered_n':int((pred!=-1).sum()),'oracle_k':len(set(y)) if method=='kmeans-oracle-k' else None,'selected_params':params if method=='hdbscan-validation-tuned' else None,**scores(y,pred),'cluster_seconds':elapsed})
        print(dataset+' complete',flush=True)
    result['process_peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    result['memory_scope']='whole process high-water mark, includes model load, both datasets and artifact hashing; not per-method peak'
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--model-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    with threadpool_limits(limits=1):r=run(external(a.cache),external(a.model_dir))
    out=external(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
