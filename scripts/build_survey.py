"""Build the public survey from reviewed Markdown; no network or dependencies.
Run --check in CI to reject source/page drift. Supports this survey's MD subset.
"""
from pathlib import Path
import argparse, hashlib, html, re, json
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
    lines=source.read_text().splitlines(); out=[]; i=0; heading=0; native_table=False
    while i<len(lines):
        line=lines[i]
        if line == '<!-- native-table -->':
            native_table=True
        elif line == '<details>':out.append('<details class="metric-help">')
        elif line == '</details>':out.append('</details>')
        elif line.startswith('<summary>') and line.endswith('</summary>'):
            out.append('<summary>'+inline(line[9:-10],source)+'</summary>')
        elif line.startswith('```'):
            block=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):block.append(lines[i]);i+=1
            out.append('<pre>'+html.escape('\n'.join(block))+'</pre>')
        elif line.startswith('|') and native_table:
            headers=[x.strip() for x in line.strip('|').split('|')];i+=2
            out.append('<div class="tablewrap" tabindex="0" aria-label="独立协议结果表，可横向滚动"><table><thead><tr>'+''.join('<th scope="col">'+inline(h,source)+'</th>' for h in headers)+'</tr></thead><tbody>')
            while i<len(lines) and lines[i].startswith('|'):
                cells=[x.strip() for x in lines[i].strip('|').split('|')]
                out.append('<tr>'+''.join(('<th scope="row">' if j==0 else '<td>')+inline(v,source)+('</th>' if j==0 else '</td>') for j,v in enumerate(cells))+'</tr>');i+=1
            out.append('</tbody></table></div>');native_table=False;continue
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

def atlas():
    data=json.loads((ROOT/'survey/atlas.json').read_text())
    esc=html.escape
    branches=''.join(f'<details class="bone"><summary><span>{esc(x["title"])}</span><strong>{esc(x["subtitle"])}</strong><small>{esc(x["examples"])}</small></summary><p>{esc(x["note"])}</p></details>' for x in data['branches'])
    cards=''.join(f'<article class="evidence-card method-card" id="profiles-{esc(x["id"],quote=True)}"><span class="method-track">{esc(x["track"])}</span><h3>{esc(x["name"])}</h3><p class="method-highlight">{esc(x["highlight"])}</p><dl><dt>信息预算</dt><dd>{esc(x["budget"])}</dd><dt>代价与边界</dt><dd>{esc(x["tradeoff"])}</dd></dl></article>' for x in data['methods'])
    return '<details class="concept-map"><summary>展开方法脉络 · 六个设计问题</summary><section class="atlas-map" aria-labelledby="map-heading"><div class="map-heading"><div><p class="eyebrow">ONE WORKFLOW · DIFFERENT QUESTIONS</p><h2 id="map-heading">同一条主干，不同的分叉。</h2></div><details class="map-info"><summary aria-label="阅读方法地图">ⓘ</summary><p>鱼骨图表示方法设计维度，并非先后依赖或算法优劣。点击每个分叉查看比较时须控制的变量。</p></details></div><div class="fishbone">'+branches+'</div><div class="spine-label">文本 → 表示 → 成簇 / 归属 → 评估</div></section></details><section id="profiles" class="survey-panel" aria-label="算法画像"><h2>算法画像</h2><p>先了解设计取舍，再看同协议实测；下面不是性能排名。</p><div class="survey-grid">'+cards+'</div></section>'

def comparison():
    return '''<section id="compare" class="survey-panel" aria-label="统一实测"><div class="section-heading"><span>REPRODUCIBLE / MATCHED INPUTS</span><a href="unified-results.json">下载本轮 JSON ↗</a></div><h2>同场景，同输入，逐项看。</h2><p>本轮对齐输入，分别比较静态聚类与路由；质量需配合覆盖率与成本阅读。</p><div class="report-entry-links"><a href="reports/unified-static-v1.html">静态聚类：结论与报告 →</a><a href="reports/routing-v2.html">路由拒识：结论与报告 →</a><a href="reports/index.html">全部实验报告 →</a></div><details id="comparison-workbench" class="comparison-workbench"><summary>展开交互对比 · 质量、覆盖率、成本与逐次记录</summary><div class="compare-controls"><label>比较组<select id="cohort"></select></label><label>聚焦维度 · 明细表同步<select id="metric"><option value="ari_all_noise_as_one_cluster">全样本 ARI ↑</option><option value="nmi_all_noise_as_one_cluster">全样本 NMI ↑</option><option value="coverage">覆盖率 ↑</option><option value="ari_covered">接收样本 ARI ↑</option><option value="fit_predict_seconds">聚类耗时 / 秒 ↓</option><option value="process_peak_rss_bytes">进程峰值内存 / MiB ↓</option></select></label></div><p id="cohort-context"></p><p id="metric-protocol" class="metric-protocol"></p><p id="compare-status" role="status">正在读取统一评估…</p><p class="interval-legend"><span class="legend-band" aria-hidden="true"></span>填色为 min–max · <span class="legend-dot" aria-hidden="true"></span>点为均值 · n 为有效运行数；不是置信区间</p><div id="comparison-chart" class="metric-card-grid" aria-label="同一比较组的分维度区间卡片"></div><details class="comparison-details"><summary>展开当前维度的原值与逐次记录</summary><div class="tablewrap" tabindex="0" aria-label="结果原值，可横向滚动"><table><thead><tr><th>方法</th><th>均值</th><th>种子范围</th><th>有效运行</th></tr></thead><tbody id="comparison-rows"></tbody></table></div><div class="tablewrap" tabindex="0" aria-label="逐次运行原值，可横向滚动"><table><thead><tr><th>方法</th><th>seed</th><th>原值</th><th>状态</th></tr></thead><tbody id="comparison-runs"></tbody></table></div></details><details class="metric-help"><summary>03 / 复现与细节 · 协议、记录和指标口径</summary><p><a href="https://github.com/okcd00/text_clustering/blob/main/results/unified-static-v1/README.md">静态聚类复现说明 ↗</a> · <a href="https://github.com/okcd00/text_clustering/blob/main/results/routing-v2/README.md">路由复现说明 ↗</a> · <a href="routing.html">路由专页与逐次明细 ↗</a></p><p>ARI/NMI 的全样本口径把噪声视为一簇；接收样本 ARI 必须配合覆盖率阅读。种子范围不是统计置信区间；不算综合分。已知 K 与未知 K 同轴分组展示，算法不同不能归因为 K 增益；路由独立。峰值内存是整个独立进程的 RSS，包含导入及输入，不是算法净增内存。耗时不含上游表示计算；可能混有 JIT 编译与缓存状态，仅是本轮运行成本，不作为稳态速度排名。原始逐种子记录与参数见 JSON。</p></details><div id="deferred-methods"></div></details><details class="archive-links"><summary>历史记录与其他场景</summary><p><a href="legacy.html">历史批量评估</a> · <a href="scenarios.html">历史路由、层级与持续更新</a>。旧记录原样保留，不参与当前统一对照。</p></details></section>'''

def roadmap_summary(source):
    # Keep all original heading IDs; detailed result bodies belong to their reports.
    from build_reports import entries
    reports={x['id']:x for x in entries()}
    chunks=re.split(r'(?=<h2 id="roadmap-)',render(source,'roadmap'))
    mapping={4:['calendar-paired-v1'],5:['calendar-paired-v2','calendar-gc-diagnostic-v1','esc-crossday-v1'],6:['hierarchy-e5-v1']}
    out=[]
    for chunk in chunks:
        m=re.match(r'<h2 id="roadmap-(\d+)".*?</h2>',chunk)
        if not m:out.append(chunk);continue
        number=int(m[1])
        if number not in mapping:
            if number==1:chunk += '<p><a href="reports/hierarchy-sibling-preflight-v1.html">同父双隐藏叶：数据预检报告（未评分） →</a></p>'
            out.append(chunk);continue
        cards=[]
        for key in mapping[number]:
            x=reports[key]
            cards.append('<article class="evidence-card report-summary"><span class="method-track">'+html.escape(x['status'])+'</span><h3>'+html.escape(x['title'])+'</h3><p>'+html.escape(x['conclusion'])+'</p><small>'+html.escape(x['limit'])+'</small><p><a href="reports/'+key+'.html">阅读独立实验报告 →</a></p></article>')
        out.append(m[0]+'<div class="survey-grid">'+''.join(cards)+'</div>')
    return ''.join(out)

def build():
    sections=[('overview','范围与证据','README.md'),('methods','方法地图','methods.md'),('datasets','数据与协议','datasets.md'),('roadmap','研究路线','roadmap.md')]
    tabs='<a href="#compare" data-section="compare">统一实测</a><a href="#profiles" data-section="profiles">算法画像</a>'+''.join(f'<a href="#{key}" data-section="{key}">{label}</a>' for key,label,_ in sections)
    body=''.join(f'<section id="{key}" class="survey-panel" aria-label="{label}"><div class="section-heading"><span>RESEARCH NOTE / {idx+1:02}</span><a href="{BASE}survey/{file}">Markdown 原文 ↗</a></div>'+(roadmap_summary(ROOT/'survey'/file) if key=='roadmap' else render(ROOT/'survey'/file,key))+'</section>' for idx,(key,label,file) in enumerate(sections))
    template=(ROOT/'scripts/survey_template.html').read_text()
    # These assets and generated markup must refresh together on cached Pages clients.
    for asset in ('survey.css', 'unified.js', 'survey.js'):
        digest=hashlib.sha256((ROOT/'docs'/asset).read_bytes()).hexdigest()[:12]
        template=template.replace(f'"{asset}"',f'"{asset}?v={digest}"')
    result=template.replace('{{TABS}}',tabs).replace('{{ATLAS}}',atlas()).replace('{{COMPARE}}',comparison()).replace('{{BODY}}',body)
    from build_reports import entries
    for report in entries():
        result=result.replace(BASE+report['source'], 'reports/'+report['id']+'.html')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();dest=ROOT/'docs/survey.html';result=build()
    if a.check:
        if not dest.exists() or dest.read_text()!=result:raise SystemExit('Survey page is stale: run python scripts/build_survey.py')
        print('Survey source/page parity: PASS')
    else:dest.write_text(result);print('Built docs/survey.html')

    from build_reports import write as write_reports
    write_reports(a.check)
