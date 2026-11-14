"""Verify unchanged hierarchy split and encode utterances; all row data external."""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
import argparse, hashlib, json, resource, time
from pathlib import Path
import numpy as np
from prepare_hierarchy import construct, run
from prepare_datasets import read_rows, external
from prepare_semantic_model import MANIFEST, verify
ROOT=Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main(cache, modeldir, out):
    expected=json.loads((ROOT/'experiments/hierarchy-preflight-v1/preflight.json').read_text())
    audit=run(cache)
    assert audit==expected, 'Preflight source/split/script provenance mismatch'
    method, scorer, _=construct(read_rows('massive_zh_cn', cache))
    for n,spec in json.loads(MANIFEST.read_text()).items(): verify(modeldir/n,spec)
    import torch
    from transformers import AutoTokenizer, AutoModel
    torch.manual_seed(0);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    tok=AutoTokenizer.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False)
    model=AutoModel.from_pretrained(modeldir,local_files_only=True,trust_remote_code=False,use_safetensors=True).eval().float()
    out.mkdir(parents=True,exist_ok=True);timing={};truncated={};private_digests={}
    roles={'support':[r['text'] for r in method['support']], 'validation':method['validation_texts'],'test':method['test_texts']}
    for role,texts in roles.items():
        vectors=[];trunc=0;t=time.perf_counter()
        with torch.inference_mode():
            for i in range(0,len(texts),16):
                b=['query: '+x for x in texts[i:i+16]]
                trunc+=sum(len(x)>512 for x in tok(b,truncation=False)['input_ids'])
                inp=tok(b,padding=True,truncation=True,max_length=512,return_tensors='pt')
                h=model(**inp).last_hidden_state;m=inp['attention_mask'].unsqueeze(-1)
                vectors.append(torch.nn.functional.normalize((h*m).sum(1)/m.sum(1),dim=1).numpy())
        z=np.concatenate(vectors);assert z.shape==(len(texts),384) and np.isfinite(z).all()
        assert np.allclose(np.linalg.norm(z,axis=1),1,atol=1e-5)
        np.save(out/f'{role}.npy',z,allow_pickle=False);timing[role]=time.perf_counter()-t;truncated[role]=trunc
        private_digests[role]=hashlib.sha256(json.dumps(texts,ensure_ascii=False).encode()).hexdigest()
    # No raw texts needed in the worker. Scorer labels remain a separate file.
    inputs={'support':[{k:v for k,v in r.items() if k!='text'} for r in method['support']], 'nodes':method['nodes']}
    (out/'method.json').write_text(json.dumps(inputs))
    for role,labels in scorer.items(): (out/f'{role}-scorer.json').write_text(json.dumps(labels))
    names=['support.npy','validation.npy','test.npy','method.json','validation-scorer.json','test-scorer.json']
    report={'protocol_sha256':sha(ROOT/'experiments/hierarchy-e5-v1/protocol.json'),'preflight_verified':True,'preflight':audit,
            'files':{n:sha(out/n) for n in names},'private_ordered_input_digests':private_digests,'encode_seconds':timing,'truncated_rows':truncated,
            'encoder_process_highwater_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'model_artifacts':json.loads(MANIFEST.read_text())}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Verified exact preflight; encoded role sizes', {k:len(v) for k,v in roles.items()},flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--model-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    main(external(a.cache),external(a.model_dir),external(a.output))
