"""Hash-verified UCI calendar preflight, no model or benchmark calls."""
import argparse, collections, datetime as dt, hashlib, json, unicodedata, urllib.request, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'datasets/news_aggregator/source.json'
PROTOCOL=ROOT/'experiments/calendar-preflight-v1/protocol.json'
def digest(b): return hashlib.sha256(b).hexdigest()
def external(path):
    p=Path(path).expanduser().resolve()
    if p==ROOT or ROOT in p.parents:raise ValueError('Cache/output must be outside repository')
    return p
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def stamp(s):return int(dt.datetime.fromisoformat(s).replace(tzinfo=dt.timezone.utc).timestamp()*1000)
def parse(b):
    rows=[]
    for line in b.decode('utf-8',errors='strict').splitlines():
        x=line.split('\t')
        if len(x)!=8 or not all(x[i] for i in (0,1,2,4,5,6,7)) or x[4] not in {'b','t','e','m'}:raise ValueError('Invalid input record')
        rows.append({'id':int(x[0]),'text':x[1],'url':x[2],'publisher':x[3],'category':x[4],'story':x[5],'time':int(x[7])})
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate numeric ID')
    return rows
def take(rows,role,n,protocol):
    return sorted(sorted(rows,key=lambda r:digest(f"{protocol['id']}|{role}|{r['id']}".encode()))[:n],key=lambda r:(r['time'],r['id']))
def split(rows,p):
    a,b,c=map(stamp,(p['warmup_start'],p['test_start'],p['test_end_exclusive']))
    windows=[('warmup',a,b,p['warmup_max_rows'])]
    for left in range(b,c,86400000):
        windows.append((dt.datetime.fromtimestamp(left/1000,dt.timezone.utc).date().isoformat(),left,left+86400000,p['per_scored_day_max_rows']))
    return [(name,left,right,take([r for r in rows if left<=r['time']<right],name,cap,p)) for name,left,right,cap in windows]
def audit(rows,p):
    selected=split(rows,p);seen_titles=set();seen_stories=set();summary=[]
    for name,left,right,items in selected:
        first=[]; prior_titles=set(seen_titles); prior_stories=set(seen_stories)
        for r in items:
            title=norm(r['text'])
            if title not in seen_titles:first.append(r)
            seen_titles.add(title)
        stories={r['story'] for r in items}
        summary.append({'role':name,'start_utc':dt.datetime.fromtimestamp(left/1000,dt.timezone.utc).isoformat(),'end_exclusive_utc':dt.datetime.fromtimestamp(right/1000,dt.timezone.utc).isoformat(),'eligible_rows':sum(left<=r['time']<right for r in rows),'selected_rows':len(items),'first_title_scored_rows':len(first),'repeated_title_rows_vs_prior_windows':sum(norm(r['text']) in prior_titles for r in items),'story_count':len(stories),'recurring_story_count':len(stories&prior_stories),'new_story_count':len(stories-prior_stories),'category_counts':dict(sorted(collections.Counter(r['category'] for r in items).items()))})
        seen_stories.update(stories)
    times=[r['time'] for r in rows]
    return {'status':'preflight_only_no_model_or_benchmark','protocol':p,'source_rows':len(rows),'timestamp_min_utc':dt.datetime.fromtimestamp(min(times)/1000,dt.timezone.utc).isoformat(),'timestamp_max_utc':dt.datetime.fromtimestamp(max(times)/1000,dt.timezone.utc).isoformat(),'rows_after_2014_08_10_utc_day':sum(t>=stamp('2014-08-11') for t in times),'source_order_timestamp_inversions':sum(b<a for a,b in zip(times,times[1:])),'empty_publishers':sum(not r['publisher'] for r in rows),'exact_title_duplicate_excess':len(rows)-len({r['text'] for r in rows}),'normalized_title_duplicate_excess':len(rows)-len({norm(r['text']) for r in rows}),'duplicate_url_excess':len(rows)-len({r['url'] for r in rows}),'story_count':len({r['story'] for r in rows}),'category_counts':dict(sorted(collections.Counter(r['category'] for r in rows).items())),'windows':summary,'selected_scored_rows':sum(w['selected_rows'] for w in summary[1:]),'first_title_scored_rows':sum(w['first_title_scored_rows'] for w in summary[1:])}
def prepare(cache,download=False):
    s=json.loads(SOURCE.read_text());p=json.loads(PROTOCOL.read_text());a=s['archive'];member=s['member']
    cache=external(cache);cache.mkdir(parents=True,exist_ok=True);file=cache/a['name']
    if not file.exists():
        if not download:raise FileNotFoundError('Pinned archive absent; use --download once')
        with urllib.request.urlopen(a['url'],timeout=90) as response:b=response.read(a['bytes']+1)
        if len(b)!=a['bytes'] or digest(b)!=a['sha256']:raise ValueError('Downloaded archive mismatch')
        file.write_bytes(b)
    b=file.read_bytes()
    if len(b)!=a['bytes'] or digest(b)!=a['sha256']:raise ValueError('Archive hash/size mismatch')
    # Read only the pinned member; never extract paths supplied by an archive.
    with zipfile.ZipFile(file) as z:
        if z.getinfo(member['name']).file_size!=member['bytes']:raise ValueError('Member size mismatch')
        b=z.read(member['name'])
    if digest(b)!=member['sha256']:raise ValueError('Member hash mismatch')
    raw=audit(parse(b),p)
    subset_keys={'protocol','windows','selected_scored_rows','first_title_scored_rows'}
    return {'status':raw['status'],'source':s,'source_audit':{k:v for k,v in raw.items() if k not in subset_keys|{'status'}},'replay_subset':{k:raw[k] for k in sorted(subset_keys)},'script_sha256':digest(Path(__file__).read_bytes())}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache',required=True);parser.add_argument('--output',required=True);parser.add_argument('--download',action='store_true');args=parser.parse_args()
    out=external(args.output);result=prepare(args.cache,args.download);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('Aggregate preflight complete; no model or benchmark executed')
