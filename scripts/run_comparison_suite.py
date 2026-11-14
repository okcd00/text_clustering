"""Unified static comparison: one immutable matrix per cohort, no implicit downloads.
Every fit runs in a fresh child; metadata contains aggregates, never examples.
"""
import argparse, hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

METHODS=[
 {'id':'kmeans','title':'KMeans','track':'known-k','highlight':'Lloyd full-batch Euclidean baseline','caveat':'Oracle test-present K; not spherical KMeans'},
 {'id':'minibatch_kmeans','title':'MiniBatchKMeans','track':'known-k','highlight':'Mini-batch centroid updates','caveat':'Batch fit here, not a streaming evaluation'},
 {'id':'hdbscan','title':'HDBSCAN','track':'unknown-k-fixed','highlight':'Density hierarchy with explicit noise','caveat':'Fixed preset; retained noise counts against coverage'},
 {'id':'bertopic_pipeline','title':'BERTopic clustering core','track':'unknown-k-fixed','highlight':'Cosine UMAP then density clustering','caveat':'Not full BERTopic: no topic words/naming/reassignment'},
 {'id':'evoc','title':'EVoC','track':'unknown-k-fixed','highlight':'Upstream label-blind selected cluster layer','caveat':'No gold-scored layer selection; K not supplied'},
]
METRICS={
 'ari_all_noise_as_one_cluster':'Adjusted Rand index on all N rows; all -1 noise forms one cluster; higher is better, may be negative.',
 'nmi_all_noise_as_one_cluster':'Arithmetic NMI on all N rows, with -1 as one cluster; higher is better.',
 'ari_covered':'ARI only on rows with prediction != -1; null if fewer than 2; never compare without coverage.',
 'coverage':'Assigned rows / all N rows; no noise reassignment; higher coverage alone does not imply quality.',
 'clusters_excluding_noise':'Number of nonnegative cluster IDs, not a quality score.',
 'fit_predict_seconds':'Wall seconds for adapter construction+fit_predict, including UMAP/JIT where incurred; includes algorithm lazy imports; excludes worker bootstrap imports, encoder, file loading, scoring. Fresh process per run, cache state recorded.',
 'cpu_seconds':'Process CPU seconds over the same fit_predict region.',
 'process_peak_rss_bytes':'Fresh worker process maximum RSS through scoring; includes imports and loaded arrays, not incremental method memory; bytes.',
}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def sample_digest(items):
 # Aggregate order fingerprint, not individual text hashes or strings.
 return hashlib.sha256(json.dumps([r['text'] for r in items],ensure_ascii=False,).encode()).hexdigest()
def fit(method,x,k,seed):
 from text_clustering.algorithms.sklearn_backends import KMeansBackend,HDBSCANBackend
 from text_clustering.algorithms.discovery_backends import BERTopicPipelineBackend,EVoCBackend
 if method in ['kmeans','minibatch_kmeans']:model=KMeansBackend(k,seed=seed,minibatch=method=='minibatch_kmeans')
 elif method=='hdbscan':model=HDBSCANBackend(min_cluster_size=15,min_samples=5)
 elif method=='bertopic_pipeline':model=BERTopicPipelineBackend(seed=seed)
 elif method=='evoc':model=EVoCBackend(seed=seed)
 else:raise ValueError('Unknown method')
 return model.fit(x)
def worker(a):
 import resource,numpy as np
 from threadpoolctl import threadpool_limits
 from benchmark_lexical import scores
 spec=json.loads(Path(a.worker).read_text());x=np.load(spec['matrix'],allow_pickle=False);y=np.load(spec['labels'],allow_pickle=False)
 if sha(spec['matrix'])!=spec['feature_sha256']:raise ValueError('Feature hash changed')
 with threadpool_limits(limits=1):
  t=time.perf_counter();c=time.process_time();p=fit(spec['method'],x,spec['k'],spec['seed']);wall=time.perf_counter()-t;cpu=time.process_time()-c
  m=scores(y,p.labels)
 result={'status':'ok','metrics':m,'fit_predict_seconds':wall,'cpu_seconds':cpu,'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'params':p.metadata}
 Path(a.result).write_text(json.dumps(result,default=lambda z:z.item() if hasattr(z,'item') else str(z),allow_nan=False))
def run(a):
 import numpy as np,importlib.metadata as md
 from prepare_datasets import prepare,read_rows,external
 from prepare_semantic_pilot import protocol_rows,MODEL,REVISION
 cache=external(a.cache);out=external(a.output);work=external(a.work);work.mkdir(parents=True,exist_ok=True)
 result={'schema_version':1,'protocol':'unified-static-v1','status':'running','generated_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'metric_definitions':METRICS,'methods':METHODS,'cohorts':[],'runs':[],'seeds':[0,1,2],'environment':{'python':platform.python_version(),'platform':platform.platform(),'threads':1,'packages':{n:md.version(n) for n in ['numpy','scipy','scikit-learn','pyarrow','umap-learn','hdbscan','evoc','numba','pynndescent','threadpoolctl','joblib']},'script_sha256':sha(__file__),'adapter_sha256':{p:sha(ROOT/'src/text_clustering/algorithms'/p) for p in ['sklearn_backends.py','discovery_backends.py']}},'representation':{'id':'e5-small-pinned','model':MODEL,'revision':REVISION,'source':'semantic-pilot-v1 external cache; no new encoding'},'deferred':[{'scenario':'routing / hierarchy','status':'deferred','reason':'Needs a shared support/query and hidden-class split; static test rows cannot reproduce supervised router evidence.'},{'scenario':'streaming / temporal / maintenance','status':'deferred','reason':'Needs prequential windows, order/time and frozen-vs-update controls; batch fit is not a streaming test.'}],'notes':['No global ranking across cohorts or information budgets.','Fixed presets are not tuned and do not establish each algorithm best achievable quality.','Same rows and matrices; KMeans arms receive oracle number of labels present in test.','Three seeds quantify initialization variability only, not sampling confidence intervals.','HDBSCAN is deterministic on these fixed inputs; its repeated seeds are timing repetitions.','Includes cold startup/JIT effects in fit region; prior process cache state may affect timings. Do not interpret as steady-state throughput.']}
 for dataset in ['banking77','massive_zh_cn']:
  audit=prepare(dataset,cache/dataset);items=protocol_rows(read_rows(dataset,cache/dataset),dataset)['test'];matrix=cache/f'{dataset}-test-e5.npy';x=np.load(matrix,allow_pickle=False)
  if x.shape!=(len(items),384) or not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-4):raise ValueError('Invalid E5 matrix')
  # Match external manifest to fixed row order, model, and vector bytes before runs.
  manifest=json.loads((cache/f'{dataset}-test-e5.meta.json').read_text())
  if manifest['ordered_texts_sha256']!=sample_digest(items) or manifest['vector_file_sha256']!=sha(matrix) or manifest['revision']!=REVISION or manifest['model']!=MODEL or manifest['rows']!=len(items):raise ValueError('Embedding provenance mismatch')
  y=np.array([r['label'] for r in items]);labels=work/f'{dataset}-labels.npy';np.save(labels,y)
  for track in ['known-k','unknown-k-fixed']:
   cid=f'{dataset}__e5-small-pinned__{track}'; result['cohorts'].append({'id':cid,'dataset':dataset,'representation':'e5-small-pinned','n':len(items),'dimensions':384,'source_revision':audit['source_revision'],'sample_sha256':sample_digest(items),'feature_sha256':sha(matrix),'scenario':track,'information_budget':'oracle test-present K' if track=='known-k' else 'no labels, no K; fixed presets','oracle_k':len(set(y)) if track=='known-k' else None})
   for method in [m for m in METHODS if m['track']==track]:
    for seed in result['seeds']:
     key=f'{cid}__{method["id"]}__{seed}';spec={'matrix':str(matrix),'labels':str(labels),'feature_sha256':sha(matrix),'method':method['id'],'k':len(set(y)) if track=='known-k' else None,'seed':seed};sp=work/f'{key}.spec.json';rp=work/f'{key}.result.json';sp.write_text(json.dumps(spec));env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',PYTHONPATH=str(ROOT/'src'))
     print(key,flush=True)
     start=time.perf_counter()
     try:
      proc=subprocess.run([sys.executable,__file__,'--worker',str(sp),'--result',str(rp)],env=env,capture_output=True,text=True,timeout=a.timeout)
      if proc.returncode:
       (work/f'{key}.stderr.log').write_text(proc.stderr)
       rec={'status':'error','error':'Worker failed; inspect local stderr log. No score published.','exit_code':proc.returncode}
      else:rec=json.loads(rp.read_text())
     except subprocess.TimeoutExpired:rec={'status':'error','error':f'Per-run timeout {a.timeout}s; no score'}
     result['runs'].append({'cohort_id':cid,'method':method['id'],'seed':seed,'worker_elapsed_seconds':time.perf_counter()-start,**rec})
     out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 result['status']='executed' if all(r['status']=='ok' for r in result['runs']) else 'partial'
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache');p.add_argument('--output');p.add_argument('--work');p.add_argument('--timeout',type=int,default=240);p.add_argument('--worker');p.add_argument('--result');a=p.parse_args()
 if a.worker:worker(a)
 elif not all([a.cache,a.output,a.work]):p.error('--cache, --output and --work required')
 else:run(a)
