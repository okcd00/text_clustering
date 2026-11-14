"""Download only pinned inert model/tokenizer artifacts; verify size and SHA-256."""
import argparse,json,urllib.request,hashlib
from pathlib import Path
from prepare_datasets import external
from prepare_semantic_pilot import MODEL,REVISION
MANIFEST=Path(__file__).resolve().parents[1]/'experiments/semantic-pilot-v1/model-artifacts.json'
def verify(path,spec):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    if path.stat().st_size!=spec['bytes'] or h.hexdigest()!=spec['sha256']:raise ValueError('Artifact mismatch: '+path.name)
def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--download',action='store_true');a=p.parse_args()
    dest=external(a.model_dir);dest.mkdir(parents=True,exist_ok=True)
    for name,spec in json.loads(MANIFEST.read_text()).items():
        path=dest/name
        if not path.exists():
            if not a.download:raise FileNotFoundError('Use --download for '+name)
            with urllib.request.urlopen(f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}',timeout=120) as r,path.open('wb') as f:
                while chunk:=r.read(1024*1024):f.write(chunk)
        verify(path,spec)
    print('Pinned model artifacts: verified')
if __name__=='__main__':main()
