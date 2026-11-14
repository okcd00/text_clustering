"""Prepare external v2 arrays; source parsing and encoder never run in workers."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
import argparse,json,zipfile,time,hashlib
from pathlib import Path
import numpy as np
from prepare_news_calendar import prepare,parse,split,external,norm,SOURCE,PROTOCOL
from benchmark_news_calendar import ROOT,ordered_digest
from prepare_semantic_model import MANIFEST,verify
from prepare_semantic_pilot import MODEL,REVISION
P=ROOT/'experiments/calendar-paired-v2/protocol.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def select(cache):
    prepare(cache)
    source=json.loads(SOURCE.read_text());p=json.loads(P.read_text());sampling={**p,'id':p['sampling_namespace']}
    with zipfile.ZipFile(cache/source['archive']['name']) as z:raw=parse(z.read(source['member']['name']))
    windows=split(raw,sampling);old=split(raw,json.loads(PROTOCOL.read_text()))
    assert [[r['id'] for r in w[3]] for w in windows[:len(old)]]==[[r['id'] for r in w[3]] for w in old]
    return windows,old

def main(cache,out,modeldir):
    out.mkdir(parents=True,exist_ok=True);windows,old=select(cache)
    oldmeta=json.loads((cache/'vectors.meta.json').read_text())
    assert oldmeta['input_digest']==ordered_digest(old) and oldmeta['sha256']==sha(cache/'vectors.npy')
    for name,spec in json.loads(MANIFEST.read_text()).items():verify(modeldir/name,spec)
    assert oldmeta['model_artifacts']==json.loads(MANIFEST.read_text())
    oldz=np.load(cache/'vectors.npy',allow_pickle=False);n_old=sum(len(w[3]) for w in old)
    assert oldz.shape==(n_old,384)
    records=[r for w in windows for r in w[3]];texts=[r['text'] for r in records[n_old:]]
    import torch
    from transformers import AutoModel,AutoTokenizer
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    tok=AutoTokenizer.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False)
    model=AutoModel.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False,use_safetensors=True).eval().float()
    vectors=[oldz];trunc=0;start=time.perf_counter()
    with torch.inference_mode():
        for i in range(0,len(texts),16):
            b=['query: '+x for x in texts[i:i+16]]
            trunc+=sum(len(x)>512 for x in tok(b,truncation=False)['input_ids'])
            inp=tok(b,padding=True,truncation=True,max_length=512,return_tensors='pt')
            h=model(**inp).last_hidden_state;m=inp['attention_mask'].unsqueeze(-1)
            vectors.append(torch.nn.functional.normalize((h*m).sum(1)/m.sum(1),dim=1).numpy())
    z=np.concatenate(vectors);assert z.shape==(len(records),384) and np.isfinite(z).all() and np.allclose(np.linalg.norm(z,axis=1),1,atol=1e-5)
    assert np.array_equal(z[:n_old],oldz)
    np.save(out/'vectors.npy',z,allow_pickle=False)
    for label in ['story','category']:
        codes={v:i for i,v in enumerate(sorted({r[label] for r in records}))}
        np.save(out/(label+'.npy'),np.array([codes[r[label]] for r in records],dtype=np.int32),allow_pickle=False)
    seen=set();mask=[]
    for r in records:
        t=norm(r['text']);mask.append(t not in seen);seen.add(t)
    np.save(out/'first_title.npy',np.array(mask,dtype=np.bool_),allow_pickle=False)
    boundaries=[];offset=0
    for date,left,right,rs in windows:
        boundaries.append({'date':date,'start':offset,'stop':offset+len(rs),'n':len(rs),'first_title_n':sum(mask[offset:offset+len(rs)])});offset+=len(rs)
    report={'protocol_id':'news-calendar-paired-v2','protocol_sha256':sha(P),'model':MODEL,'revision':REVISION,'model_artifacts':json.loads(MANIFEST.read_text()),'windows':boundaries,'shape':list(z.shape),'reused_v1_rows':n_old,'new_encoded_rows':len(texts),'new_encode_seconds':time.perf_counter()-start,'new_truncated_rows':trunc,'prefix_rows_and_vectors_verified':True,'private_input_digest':ordered_digest(windows),'files':{n:{'sha256':sha(out/n),'bytes':(out/n).stat().st_size} for n in ['vectors.npy','story.npy','category.npy','first_title.npy']}}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print('Prepared',len(records),'rows across',len(windows)-1,'days',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True);p.add_argument('--model-dir',required=True);a=p.parse_args();main(external(a.cache),external(a.output),external(a.model_dir))
