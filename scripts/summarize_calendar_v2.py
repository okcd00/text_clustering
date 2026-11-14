"""Validate two full extensions and v1 prefix, export aggregate-only comparison."""
import argparse,json,hashlib
from pathlib import Path
from benchmark_calendar_v2 import ROOT,P,sha,external
METHODS=['frozen','unbounded','cap-2048','rolling-2048']
def invariant(x):
    if isinstance(x,dict):return {k:invariant(v) for k,v in x.items() if not k.endswith('_seconds') and k not in ['memory','after_update_memory','overlap_memory']}
    if isinstance(x,list):return [invariant(v) for v in x]
    return x
def summarize(cache,old):
    meta=json.loads((cache/'manifest.json').read_text());runs=[]
    for method in METHODS:
        a,b=[json.loads((cache/f'{method}-{n}.json').read_text()) for n in [1,2]]
        for r in [a,b]:
            assert r['method']==method and r['status']=='executed' and r['protocol_sha256']==sha(P)
            assert [(w['date'],w['n']) for w in r['windows']]==[(w['date'],w['n']) for w in meta['windows'][1:]]
        assert invariant(a)==invariant(b),method
        v1_path=old/f'{method}-1.json'
        if not v1_path.exists():v1_path=old/f'{method}.json'
        v1=json.loads(v1_path.read_text())
        assert a['private_prediction_digest_v1_prefix']==v1['private_prediction_digest']
        for x,y in zip(a['windows'][:7],v1['windows']):
            for k in ['date','n','leaf_cf_before','leaf_cf_after','scores','skipped_updates','rebuilds']:assert x[k]==y[k],(method,k)
        for r in [a,b]:
            r.pop('private_prediction_digest');r.pop('private_prediction_digest_v1_prefix')
        a['serialized_size_scope']='BIRCH estimator pickle only; excludes rolling replay buffer and controller state; no restart checkpoint saved'
        a['rolling_buffer_numeric_payload_bytes']=2048*384*4 if method=='rolling-2048' else 0
        a['repeat_verified']='all non-timing/memory fields and prediction digests exact; seven-day v1 prefix exact'
        a['second_run']={'memory':b['memory'],'warmup_seconds':b['warmup_seconds'],'windows':[{k:w[k] for k in ['date','prediction_seconds','update_seconds','after_update_memory']} for w in b['windows']],'rebuild_events':b['rebuild_events']}
        runs.append(a)
    cap=next(r for r in runs if r['method']=='cap-2048');unbounded=next(r for r in runs if r['method']=='unbounded')
    arrival=cap['first_freeze_arrival'];cumulative=0;trigger_day=None;post_start=None
    for i,w in enumerate(cap['windows']):
        cumulative+=w['n']
        if arrival is not None and cumulative>=arrival:
            trigger_day=w['date'];post_start=i+1;break
    comparisons=[]
    for r in runs:
        for scope,ws in [('full',r['windows']),('post_cap',r['windows'][post_start:] if post_start is not None else [])]:
            for view in ['all','first_title']:
                for metric in ['story_ari','category_ari']:
                    pairs=[]
                    for w in ws:
                        ref=next(x for x in unbounded['windows'] if x['date']==w['date']);s=w['scores'][view];rs=ref['scores'][view]
                        assert s['n']==rs['n']
                        if s[metric] is not None:pairs.append({'date':w['date'],'n':s['n'],'value':s[metric],'reference':rs[metric],'delta':s[metric]-rs[metric]})
                    n=sum(x['n'] for x in pairs)
                    comparisons.append({'method':r['method'],'scope':scope,'view':view,'metric':metric,'scored_n':n,'scored_days':len(pairs),'mean':sum(x['value']*x['n'] for x in pairs)/n if n else None,'reference_mean':sum(x['reference']*x['n'] for x in pairs)/n if n else None,'weighted_delta':sum(x['delta']*x['n'] for x in pairs)/n if n else None,'worst_day_delta':min((x['delta'] for x in pairs),default=None),'direction_counts':{name:sum(test(x['delta']) for x in pairs) for name,test in [('negative',lambda x:x< -1e-12),('zero',lambda x:abs(x)<=1e-12),('positive',lambda x:x>1e-12)]},'daily':pairs})
    meta.pop('private_input_digest');meta.pop('files')
    return {'status':'executed_and_repeated','protocol':json.loads(P.read_text()),'preparation':meta,'trigger_day':trigger_day,'post_cap_start_date':cap['windows'][post_start]['date'] if post_start is not None and post_start<len(cap['windows']) else None,'runs':runs,'comparisons':comparisons,'scripts':{n:sha(ROOT/'scripts'/n) for n in ['prepare_calendar_v2.py','benchmark_calendar_v2.py','summarize_calendar_v2.py']}}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--v1-cache',required=True);p.add_argument('--output',required=True);a=p.parse_args();external(a.output).write_text(json.dumps(summarize(external(a.cache),external(a.v1_cache)),indent=2)+'\n')
