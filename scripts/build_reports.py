"""Deterministic Markdown report pages, no network or third-party dependency.
Curated leads live in survey/reports.json; report bodies always derive from Markdown.
"""
from pathlib import Path
import argparse, hashlib, html, json, re
ROOT=Path(__file__).resolve().parents[1]
BASE='https://github.com/okcd00/text_clustering/blob/main/'
def entries(): return json.loads((ROOT/'survey/reports.json').read_text())
def url(target,source):
    if target.startswith(('https://','http://','#','mailto:')):return target
    path=(source.parent/target.split('#')[0]).resolve().relative_to(ROOT).as_posix()
    # Rendered report bodies link to one another rather than GitHub markdown.
    match=next((x for x in entries() if x['source']==path),None)
    return (match['id']+'.html') if match and '#' not in target else BASE+path+('#'+target.split('#',1)[1] if '#' in target else '')
def inline(s,source):
    tokens=[]
    def keep(value):tokens.append(value);return f'\x00{len(tokens)-1}\x00'
    s=re.sub(r'`([^`]+)`',lambda m:keep('<code>'+html.escape(m[1])+'</code>'),s)
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:keep('<a href="'+html.escape(url(m[2],source),quote=True)+'">'+html.escape(m[1])+'</a>'),s)
    s=html.escape(s);s=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',s)
    return re.sub(r'\x00(\d+)\x00',lambda m:tokens[int(m[1])],s)
def markdown(source):
    lines=source.read_text().splitlines();out=[];i=0;n=0;toc=[]
    while i<len(lines):
        line=lines[i]
        if line.startswith('# '):pass # title is shown in original-source label below
        elif line.startswith('```'):
            block=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):block.append(lines[i]);i+=1
            out.append('<pre><code>'+html.escape('\n'.join(block))+'</code></pre>')
        elif line.startswith('|') and i+1<len(lines) and re.match(r'^\|[\s:|\-]+\|?$',lines[i+1]):
            cells=lambda x:[c.strip() for c in x.strip().strip('|').split('|')]
            out.append('<div class="tablewrap" tabindex="0" aria-label="实验原表，可横向滚动"><table><thead><tr>'+''.join('<th scope="col">'+inline(c,source)+'</th>' for c in cells(line))+'</tr></thead><tbody>');i+=2
            while i<len(lines) and lines[i].startswith('|'):
                out.append('<tr>'+''.join('<td>'+inline(c,source)+'</td>' for c in cells(lines[i]))+'</tr>');i+=1
            out.append('</tbody></table></div>');continue
        elif line.startswith(('## ','### ','#### ')):
            level=len(line)-len(line.lstrip('#'));title=line[level+1:];n+=1
            out.append(f'<h{level} id="section-{n}">'+inline(title,source)+f'</h{level}>');toc.append((n,title))
        elif line=='<details>':out.append('<details class="report-disclosure">')
        elif line=='</details>':out.append(line)
        elif line.startswith('<summary>') and line.endswith('</summary>'):out.append('<summary>'+inline(line[9:-10],source)+'</summary>')
        elif line.startswith('<!--'):pass
        elif re.match(r'^(\d+\. |[-*] )',line):out.append('<p class="report-list">'+inline(line,source)+'</p>')
        elif line.strip():out.append('<p>'+inline(line,source)+'</p>')
        i+=1
    return '\n'.join(out),toc

def shell(title,body):
    digest=hashlib.sha256((ROOT/'docs/report.css').read_bytes()).hexdigest()[:12]
    return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+' · Text Clustering Lab</title><link rel="stylesheet" href="../style.css"><link rel="stylesheet" href="../report.css?v='+digest+'"></head><body><main>'+body+'</main></body></html>\n'
def card(x):return '<article class="report-card"><span class="report-status">'+html.escape(x['status'])+'</span><h2><a href="'+x['id']+'.html">'+html.escape(x['title'])+'</a></h2><p>'+html.escape(x['conclusion'])+'</p><small>'+html.escape(x['limit'])+'</small></article>'
def report(x):
    source=ROOT/x['source'];body,toc=markdown(source)
    back='../survey.html#'+x['anchor']
    if x['id']=='routing-v2':back='../survey.html?view=routing&amp;dataset=massive_zh_cn#compare'
    lead='<header><a href="index.html">← 实验报告目录</a><a href="'+back+'">返回 Survey 证据</a></header><section class="report-lead"><p class="eyebrow">'+html.escape(x['status'])+' / '+x['id']+'</p><h1>'+html.escape(x['title'])+'</h1><p class="report-conclusion">'+html.escape(x['conclusion'])+'</p><p class="report-limit">'+html.escape(x['limit'])+'</p><div class="report-actions"><a href="'+BASE+x['source']+'">Markdown 原文 ↗</a><a href="'+BASE+x['reproduction']+'">复现说明 ↗</a></div></section>'
    provenance='<details class="report-provenance"><summary>ⓘ 摘要与原文来源</summary><p>上方为编辑摘要，来源于以下报告，不是新增实验。正文从仓库 Markdown 自动生成，不维护第二份结果。</p><p><code>'+x['source']+'</code><br>源文件 SHA256：<code>'+hashlib.sha256(source.read_bytes()).hexdigest()+'</code></p></details>'
    contents='<details class="report-toc"><summary>报告目录</summary>'+''.join(f'<a href="#section-{n}">'+html.escape(t)+'</a>' for n,t in toc)+'</details>'
    original=source.read_text().splitlines()[0].lstrip('# ')
    return shell(x['title'],lead+provenance+contents+'<section class="report-body" aria-label="Markdown 报告正文"><p class="original-title">原报告：'+html.escape(original)+'</p>'+body+'</section><footer><a href="index.html">← 所有实验报告</a><a href="'+back+'">返回对应证据 →</a></footer>')
def build_all():
    es=entries();out={x['id']+'.html':report(x) for x in es}
    main=[x for x in es if x['status'] not in ['仅数据预检','早期基线']]
    rest=[x for x in es if x not in main]
    out['index.html']=shell('实验报告', '<header><a href="../index.html">← 首页场景结论</a><a href="../survey.html">Survey 证据</a></header><section class="report-lead"><p class="eyebrow">EXPERIMENT NOTEBOOK</p><h1>每次实验，独立读懂。</h1><p>先看结论与边界，再读完整报告、原始记录和复现步骤。不同协议不合成总榜。</p></section><div class="report-grid">'+''.join(card(x) for x in main)+'</div><h2>预检与早期基线</h2><div class="report-grid">'+''.join(card(x) for x in rest)+'</div>')
    return out
def write(check=False):
    dest=ROOT/'docs/reports';out=build_all()
    for name,s in out.items():
        p=dest/name
        if check:
            if not p.exists() or p.read_text()!=s:raise SystemExit('Stale report: '+name)
        else:dest.mkdir(exist_ok=True);p.write_text(s)
    print('Report source/page parity: PASS' if check else f'Built {len(out) - 1} reports and directory')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');write(p.parse_args().check)
