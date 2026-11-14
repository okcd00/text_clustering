"""Validate two complete paired runs, then export only reviewed aggregate fields."""
import argparse,json,hashlib,platform
from pathlib import Path
from benchmark_news_calendar import ROOT,P,sha
from prepare_semantic_pilot import MODEL,REVISION
from prepare_news_calendar import external
METHODS=['frozen','unbounded','cap-2048','cap-128','rolling-2048']
def invariant(x):
    if isinstance(x,dict):return {k:invariant(v) for k,v in x.items() if not k.endswith('_seconds') and k!='process_peak_rss_mib'}
    if isinstance(x,list):return [invariant(v) for v in x]
    return x
def summarize(cache):
    output=[]
    for method in METHODS:
        a,b=[json.loads((cache/f'{method}-{rep}.json').read_text()) for rep in [1,2]]
        for r in (a,b):
            assert r['status']=='executed' and r['method']==method
            assert [w['date'] for w in r['windows']]==[f'2014-03-{day}' for day in range(24,31)]
            assert all(w['n']==512 for w in r['windows'])
            assert sum(w['scores']['first_title']['n'] for w in r['windows'])==3553
        assert invariant(a)==invariant(b),method
        a.pop('private_prediction_digest')
        a['repeat_check']='all non-timing/resource fields and private prediction digest exact'
        a['second_run_timing']={'warmup_seconds':b['warmup_seconds'],'predict_seconds':sum(w['predict_seconds'] for w in b['windows']),'update_seconds':sum(w['update_seconds'] for w in b['windows']),'process_peak_rss_mib':b['process_peak_rss_mib']}
        output.append(a)
    meta=json.loads((cache/'vectors.meta.json').read_text());meta.pop('input_digest');meta.pop('sha256');meta.update(model=MODEL,revision=REVISION)
    report={'status':'executed_and_repeated','protocol':json.loads(P.read_text()),'source':json.loads((ROOT/'datasets/news_aggregator/source.json').read_text()),'representation':meta,'scripts':{p.name:sha(p) for p in [Path(__file__),ROOT/'scripts/benchmark_news_calendar.py',ROOT/'scripts/prepare_news_calendar.py',P]},'runs':output,'limitations':['thinned news-title replay, not behavior stream','single frozen setting; no universal ranking or statistical confidence','STORY silver labels and CATEGORY distinct target','within-day scores do not track identity across days','serialized bytes are storage proxy; RSS includes source parsing','cap128 already exceeded at common warmup, not a 128-budget method']}
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=external(a.output);out.write_text(json.dumps(summarize(external(a.cache)),indent=2)+'\n')
