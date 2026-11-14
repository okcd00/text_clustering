"""Prepare bounded, label-blind pilot samples; inspect runtime, never run a model.
No sample texts/IDs/assignments or text hashes are emitted.
"""
import argparse, hashlib, importlib.util, json
from collections import Counter
from pathlib import Path
from prepare_datasets import prepare, read_rows, normalize, external

MODEL = 'intfloat/multilingual-e5-small'
REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'

def sample(rows, dataset, split, limit=512):
    # Pin ordering by original source row position, independent of text/labels.
    ranked=sorted(enumerate(rows), key=lambda p:hashlib.sha256(f'semantic-pilot-v1|{dataset}|{split}|{p[0]}'.encode()).digest())
    return [r for _,r in ranked[:limit]]

def protocol_rows(rows, dataset):
    test=sample(rows['test'],dataset,'test')
    # Test TEXTS only used for exclusion, never labels for validation construction.
    all_test_text={normalize(r['text']) for r in rows['test']}
    source='validation' if 'validation' in rows else 'train'
    eligible=[r for r in rows[source] if normalize(r['text']) not in all_test_text]
    validation=sample(eligible,dataset,source+'-overlap-excluded')
    return {'validation':validation,'test':test}

def describe(items):
    c=Counter(r['label'] for r in items)
    return {'rows':len(items),'classes':len(c),'class_counts':dict(sorted(c.items())),'singleton_classes':sum(n==1 for n in c.values()),'normalized_duplicate_excess':len(items)-len({normalize(r['text']) for r in items})}

def check(cache):
    modules={m:importlib.util.find_spec(m) is not None for m in ['torch','transformers','tokenizers','safetensors']}
    result={'status':'preparation_only_no_embeddings_or_scores','model':MODEL,'revision':REVISION,'module_presence':modules,'encoding_readiness':'NOT_VERIFIED_even_if_modules_present','datasets':{}}
    for dataset in ['banking77','massive_zh_cn']:
        audit=prepare(dataset,Path(cache)/dataset)
        parts=protocol_rows(read_rows(dataset,Path(cache)/dataset),dataset)
        assert not ({normalize(r['text']) for r in parts['validation']} & {normalize(r['text']) for r in parts['test']})
        result['datasets'][dataset]={'source_revision':audit['source_revision'],'splits':{k:describe(v) for k,v in parts.items()},'validation_test_normalized_overlap':0}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--report',required=True);a=p.parse_args()
    report=check(external(a.cache));out=external(a.report);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'status':report['status'],'module_presence':report['module_presence']}))
