"""Fail-closed aggregate export, excluding input/prediction payloads and digests."""
import argparse
import json
from pathlib import Path
from benchmark_calendar_v2 import ROOT, sha


def signature(run):
    fields=('date','n','leaf_cf_before','leaf_cf_after','scores','skipped_updates','rebuilds')
    return [{k:w[k] for k in fields} for w in run['windows']]


def main(cache, output):
    order=[('automatic',1),('explicit-after-swap',1),('explicit-after-swap',2),('automatic',2)]
    runs=[json.loads((cache/f'gc-{arm}-{rep}.json').read_text()) for arm,rep in order]
    control=json.loads((cache/'gc-uninstrumented.json').read_text())
    original=json.loads((ROOT/'results/calendar-paired-v2/results.json').read_text())
    reference=next(r for r in original['runs'] if r['method']=='rolling-2048')
    assert signature(control)==signature(reference), 'Reconstructed v2 differs from published daily quality/counts'
    for (arm,rep),r in zip(order,runs):
        assert r['diagnostic_arm']==arm and r['status']=='executed'
        assert r['protocol_sha256']==reference['protocol_sha256']
        assert r['private_prediction_digest']==control['private_prediction_digest']
        assert signature(r)==signature(control)
        assert [e['arrival'] for e in r['rebuild_events']]==[2048,4096,6144,8192,10240]
        assert r['serialized_final_model_bytes']==control['serialized_final_model_bytes']
        if arm=='explicit-after-swap':
            assert all(e['all_retired_alive_after_gc_or_noop']==0 for e in r['rebuild_events'])
    keep=('diagnostic_arm','protocol_sha256','warmup_seconds','rebuild_events','serialized_final_model_bytes','memory','python','versions','hardware','gc_thresholds','diagnostic_script_sha256')
    public=[]
    for (_,rep),r in zip(order,runs):
        v={k:r[k] for k in keep};v['repeat']=rep
        v['update_seconds']=sum(w['update_seconds'] for w in r['windows'])
        v['explicit_gc_seconds']=sum(e['explicit_gc_seconds'] or 0 for e in r['rebuild_events'])
        public.append(v)
    result={'status':'executed_two_repeats_per_arm','base_commit':'b2d3db7a9b1050fb033049bf725002838b61c164',
            'checks':{'matched_current_predictions_all_arms_and_uninstrumented':True,'matched_published_v2_daily_scores_and_cf_counts':True,'original_private_vector_or_prediction_bytes_available':False},
            'limitations':['Reconstructed cache, not a comparison of unavailable original private vector bytes.',
                            'Weakrefs observe linked leaf nodes, not complete ownership or native allocations.',
                            '5 ms nominal sampling can miss transients. RSS is not an ownership profiler.',
                            'No allocator retention, memory cap or production performance causal claim.'],
            'serialized_size_scope':'Estimator only; excludes 3 MiB numeric replay payload plus Python/controller state; RSS already includes replay.',
            'runs':public,'scripts':{n:sha(ROOT/'scripts'/n) for n in ['diagnose_calendar_gc.py','summarize_calendar_gc.py','benchmark_calendar_v2.py']}}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.cache,a.output)
