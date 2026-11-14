"""Pinned public-data preparation. No model calls; never writes payloads in checkout.
BANKING uses stdlib; MASSIVE converted parquet requires pyarrow==19.0.1.
"""
from pathlib import Path
import argparse, collections, csv, hashlib, itertools, json, unicodedata, urllib.request
ROOT=Path(__file__).resolve().parents[1]

def external(path):
    p=Path(path).expanduser().resolve()
    if p==ROOT or ROOT in p.parents: raise ValueError('Data/cache/output must be outside the repository')
    return p

def digest(data): return hashlib.sha256(data).hexdigest()
def normalize(text): return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())

def read_rows(dataset,cache):
    if dataset=='banking77':
        return {s:[{'text':r['text'],'label':r['category'],'id':None,'parent':None} for r in csv.DictReader((cache/f'{s}.csv').open(encoding='utf-8'))] for s in ['train','test']}
    import pyarrow.parquet as pq
    result={}
    for split in ['train','validation','test']:
        table=pq.read_table(cache/f'{split}.parquet')
        features=json.loads(table.schema.metadata[b'huggingface'])['info']['features']
        records=table.to_pylist()
        if any(r['locale']!='zh-CN' or r['partition']!=('dev' if split=='validation' else split) for r in records):raise ValueError('Locale or partition mismatch')
        result[split]=[{'text':r['utt'],'label':features['intent']['names'][r['intent']],'id':str(r['id']),'parent':features['scenario']['names'][r['scenario']]} for r in records]
    return result

def audit(rows):
    summary={}; exact={};normalized={};ids={};parents=collections.defaultdict(set)
    for split,items in rows.items():
        texts=[r['text'] for r in items]
        if any(not isinstance(t,str) or not t.strip() for t in texts):raise ValueError('Missing/empty text')
        exact[split]=set(texts);normalized[split]={normalize(t) for t in texts}
        id_values=[r['id'] for r in items if r['id'] is not None];ids[split]=set(id_values)
        for r in items:
            if r['parent'] is not None:parents[r['label']].add(r['parent'])
        summary[split]={'rows':len(items),'classes':len({r['label'] for r in items}),'parents':len({r['parent'] for r in items if r['parent'] is not None}),'exact_duplicate_excess':len(texts)-len(exact[split]),'normalized_duplicate_excess':len(texts)-len(normalized[split]),'id_duplicate_excess':len(id_values)-len(ids[split]),'class_counts':dict(sorted(collections.Counter(r['label'] for r in items).items()))}
    pairs={}
    for a,b in itertools.combinations(rows,2):
        pairs[a+'__'+b]={'exact_shared_texts':len(exact[a]&exact[b]),'normalized_shared_texts':len(normalized[a]&normalized[b]),'shared_ids':len(ids[a]&ids[b])}
    training_labels={r['label'] for r in rows['train']}
    coverage={s:{'missing_training_labels':sorted(training_labels-{r['label'] for r in items}),'labels_not_in_training':sorted({r['label'] for r in items}-training_labels)} for s,items in rows.items()}
    return {'label_coverage':coverage,'splits':summary,'cross_split':pairs,'hierarchy':{'child_count':len(parents),'parent_count':len(set().union(*parents.values())) if parents else 0,'multi_parent_children':sum(len(v)!=1 for v in parents.values())},'normalization':'Unicode NFKC + casefold + whitespace collapse; punctuation preserved','overlap_unit':'distinct text strings, not pairwise row matches; no automatic removal','status':'audit_only_not_benchmark_ready'}

def prepare(dataset,cache,download=False):
    cache=external(cache);cache.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'datasets'/dataset/'source.json').read_text());observed=[]
    for item in manifest['files']:
        name=item.get('name') or item['url'].rsplit('/',1)[1]
        if Path(name).name!=name:raise ValueError('Unsafe filename')
        dest=cache/name
        if not dest.exists():
            if not download:raise FileNotFoundError(f'{name}: supply --download to fetch pinned artifact')
            with urllib.request.urlopen(item['url'],timeout=60) as response:data=response.read()
            if digest(data)!=item['sha256']:raise ValueError(f'{name}: downloaded hash mismatch')
            dest.write_bytes(data)
        data=dest.read_bytes()
        if digest(data)!=item['sha256'] or len(data)!=item['bytes']:raise ValueError(f'{name}: hash/size mismatch')
        observed.append({'name':name,'sha256':digest(data),'bytes':len(data)})
    report=audit(read_rows(dataset,cache));report.update({'dataset':dataset,'source_revision':manifest['revision'],'license':manifest['license'],'artifacts':observed})
    expected=manifest.get('expected_splits',{})
    for split,spec in expected.items():
        for key,value in spec.items():
            if report['splits'][split][key]!=value:raise ValueError(f'{split} {key}: expected {value}, got {report["splits"][split][key]}')
    if any(v['shared_ids'] for v in report['cross_split'].values()) or any(v['id_duplicate_excess'] for v in report['splits'].values()):raise ValueError('ID split leakage')
    if report['hierarchy']['multi_parent_children']:raise ValueError('Intent maps to multiple parents')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['banking77','massive_zh_cn']);p.add_argument('--cache',required=True);p.add_argument('--download',action='store_true');p.add_argument('--report',required=True);a=p.parse_args()
    out=external(a.report);report=prepare(a.dataset,a.cache,a.download);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['splits','artifacts']},ensure_ascii=False));print('Report saved outside repository; inspect overlap before selecting an evaluation protocol.')
