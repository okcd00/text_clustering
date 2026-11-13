"""Fail closed on common accidental public-release payloads; review still required."""
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parents[1]
patterns = [r'alibaba' + r'-inc\.com', r'/Users/' + r'\w+', r'offline-' + r'whale', r'dianchen' + r'_', r'dc_' + r'qwen', r'(?i)(?:api[_-]?key|token|password)\s*[:=]\s*[\"\'][A-Za-z0-9_\-]{8,}[\"\']', r'-----BEGIN ' + r'(?:RSA |OPENSSH )?PRIVATE KEY']
paths = subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard'],cwd=root).decode().split('\0')
errors=[]
for name in set(paths)-{''}:
    p=root/name
    if p.suffix.lower() in {'.pdf','.npy','.npz','.parquet','.pkl','.jsonl'}:
        errors.append((name,'payload prohibited'));continue
    try: text=p.read_text()
    except UnicodeError:
        errors.append((name,'binary needs explicit review'));continue
    if any(re.search(pattern,text) for pattern in patterns):errors.append((name,'sensitive marker'))
    if text.startswith('version https://git-lfs.github.com/spec/'):errors.append((name,'LFS pointer prohibited'))
for name,reason in errors:print(name,reason)
print('Public content audit:', 'FAIL' if errors else 'PASS', f'({len(set(paths)-{chr(0),""})} files)')
raise SystemExit(bool(errors))
