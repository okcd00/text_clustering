"""Build the public survey from reviewed Markdown; no network or dependencies.
Run --check in CI to reject source/page drift. Supports this survey's MD subset.
"""
from pathlib import Path
import argparse, html, re
ROOT=Path(__file__).resolve().parents[1]
BASE='https://github.com/okcd00/text_clustering/blob/main/'

def inline(text, source):
    def link(m):
        label,url=m.groups()
        if not url.startswith(('https://','http://','#')):
            path=(source.parent/url).resolve().relative_to(ROOT)
            url=BASE+path.as_posix()
        return '<a href="'+html.escape(url,quote=True)+'">'+html.escape(label)+'</a>'
    text=html.escape(text)
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,text)
    text=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',text)
    return re.sub(r'`([^`]+)`',r'<code>\1</code>',text)

def render(source,prefix):
    lines=source.read_text().splitlines(); out=[]; i=0; heading=0
    while i<len(lines):
        line=lines[i]
        if line.startswith('```'):
            block=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):block.append(lines[i]);i+=1
            out.append('<pre>'+html.escape('\n'.join(block))+'</pre>')
        elif line.startswith('|'):
            headers=[x.strip() for x in line.strip('|').split('|')];i+=2;out.append('<div class="survey-grid">')
            while i<len(lines) and lines[i].startswith('|'):
                cells=[x.strip() for x in lines[i].strip('|').split('|')]
                out.append('<article class="evidence-card"><h3>'+inline(cells[0],source)+'</h3><dl>')
                for title,value in zip(headers[1:],cells[1:]):out.append('<dt>'+inline(title,source)+'</dt><dd>'+inline(value,source)+'</dd>')
                out.append('</dl></article>');i+=1
            out.append('</div>');continue
        elif line.startswith('# '): pass
        elif line.startswith('## '):
            heading+=1;out.append(f'<h2 id="{prefix}-{heading}">'+inline(line[3:],source)+f' <a class="permalink" href="#{prefix}-{heading}" aria-label="链接到此节">#</a></h2>')
        elif re.match(r'^(\d+\. |\- )',line):out.append('<p class="list-item">'+inline(line,source)+'</p>')
        elif line.strip():out.append('<p>'+inline(line,source)+'</p>')
        i+=1
    return '\n'.join(out)

def build():
    sections=[('overview','范围与证据','README.md'),('methods','方法地图','methods.md'),('datasets','数据与协议','datasets.md'),('roadmap','研究路线','roadmap.md')]
    tabs=''.join(f'<a href="#{key}" data-section="{key}">{label}</a>' for key,label,_ in sections)
    body=''.join(f'<section id="{key}" class="survey-panel" aria-label="{label}"><div class="section-heading"><span>RESEARCH NOTE / {idx+1:02}</span><a href="{BASE}survey/{file}">Markdown 原文 ↗</a></div>'+render(ROOT/'survey'/file,key)+'</section>' for idx,(key,label,file) in enumerate(sections))
    template=(ROOT/'scripts/survey_template.html').read_text()
    return template.replace('{{TABS}}',tabs).replace('{{BODY}}',body)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();dest=ROOT/'docs/survey.html';result=build()
    if a.check:
        if not dest.exists() or dest.read_text()!=result:raise SystemExit('Survey page is stale: run python scripts/build_survey.py')
        print('Survey source/page parity: PASS')
    else:dest.write_text(result);print('Built docs/survey.html')
