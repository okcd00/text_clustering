"""Validate two frozen runs, export aggregate hierarchy outcomes only."""
import argparse,json
from pathlib import Path
from benchmark_hierarchy_e5 import ROOT,sha

def main(cache,out):
    runs=[json.loads((cache/f'run-{i}.json').read_text()) for i in (1,2)]
    meta=json.loads((cache/'manifest.json').read_text());protocol=json.loads((ROOT/'experiments/hierarchy-e5-v1/protocol.json').read_text())
    assert meta['preflight_verified'] and meta['preflight']==json.loads((ROOT/'experiments/hierarchy-preflight-v1/preflight.json').read_text())
    for n,h in meta['files'].items():assert sha(cache/n)==h
    stable=lambda r:{k:v for k,v in r.items() if k not in ['timing','memory']}
    assert stable(runs[0])==stable(runs[1]), 'Repeat mismatch'
    for r in runs:
        assert r['status']=='executed' and r['protocol_sha256']==meta['protocol_sha256']==sha(ROOT/'experiments/hierarchy-e5-v1/protocol.json')
        assert r['known_leaf_count']==31 and r['supported_parent_count']==18
        for role,nk,nh in [('validation',223,111),('test',230,104)]:
            g=r['roles'][role]['all'];assert g['known_rows']==nk and g['hidden_rows']==nh
            assert g['rows']==nk+nh
            for field in ['known_rows','hidden_rows','rows']:
                assert sum(r['roles'][role][s][field] for s in ['one_supported_child','multiple_supported_children'])==g[field]
            for arm in ['flat','equal_leaf_mean']:
                p=g['parent_arms'][arm]
                assert p['joint_reject_and_correct_parent']['numerator']==p['parent_accuracy_among_rejected_hidden']['numerator']
                assert p['parent_accuracy_among_rejected_hidden']['denominator']==g['hidden_rejection_detection']['numerator']
    prep={k:meta[k] for k in ['encode_seconds','truncated_rows','encoder_process_highwater_bytes','model_artifacts']}
    prep['source_revision']=meta['preflight']['source_revision'];prep['source_artifacts']=meta['preflight']['artifacts']
    prep['split_fingerprint']=meta['preflight']['split_fingerprint'];prep['preflight_verified']=True
    public=[{k:v for k,v in r.items() if not k.startswith('private_')} for r in runs]
    result={'status':'executed_and_repeated','protocol':protocol,'preparation':prep,'runs':public,
            'checks':{'identical_prediction_and_mask_digest':True,'identical_non_resource_time_fields':True,'shared_leaf_predictions_and_rejection_mask_across_parent_arms':True},
            'scripts':{n:sha(ROOT/'scripts'/n) for n in ['prepare_hierarchy.py','prepare_hierarchy_e5.py','benchmark_hierarchy_e5.py','summarize_hierarchy_e5.py']}}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();main(a.cache,a.output)
