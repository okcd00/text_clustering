'use strict';
const $=id=>document.getElementById(id);
const labels={onepass:'新文本路由',hierarchical:'层级导航',evolution:'随机顺序持续更新',evolution_temporal:'日历时间流',evolution_temporal_capacity:'容量与维护'};
const route=[['known_accuracy_all','已知类准确率 ↑'],['known_coverage','已知类覆盖率 ↑'],['hidden_false_accept_rate','隐藏类误接收 ↓'],['novel_f1','新类检测 F1 ↑'],['novel_and_parent_recall','新类且父级正确 ↑'],['fit_predict_seconds','拟合+测试 / s ↓'],['query_seconds','查询 / s ↓'],['process_peak_rss_bytes','进程峰值 RSS / MiB ↓']];
const stream=[['window_weighted_acc_all_noise_error','窗口加权 ACC ↑'],['window_weighted_ari_assigned','窗口加权 ARI ↑'],['coverage','覆盖率 ↑'],['max_window_predicted_clusters','最大窗口输出簇数'],['last_window_predicted_clusters','末窗口输出簇数'],['update_seconds','更新 / s ↓'],['process_job_seconds','完整作业 / s ↓'],['process_peak_rss_bytes','进程峰值 RSS / MiB ↓']];
let data;
function options(el,items){el.replaceChildren(...items.map(([v,t])=>{const o=document.createElement('option');o.value=v;o.textContent=t;return o;}));}
function format(v,k=''){if(v===null||v===undefined)return '—';if(typeof v==='object'&&'mean'in v){if(v.mean===null)return '—';return format(v.mean,k)+(v.sd==null?'':` ± ${format(v.sd,k)}`);}if(typeof v==='number')return (k.includes('bytes')?v/1048576:v).toLocaleString('en-US',{maximumFractionDigits:4});return String(v);}
function draw(){const g=data.groups.find(g=>g.id===$('group').value);const rows=g.records.filter(r=>r.view===$('view').value);$('heading').textContent=g.title+' · '+g.protocol;$('scope').textContent=JSON.stringify(rows[0]?.dataset||{});$('takeaway').textContent=g.task.includes('temporal')?'质量按日历窗口计算；raw 与 first_title 复用学习和成本，不能重复计费。维护耗时与 native fit 有重叠，不能相加。':g.task==='evolution'?'本组比较冻结与持续更新。窗口输出簇数不等于模型微簇状态大小；状态和学习资源以明细为准。':'隐藏类误接收必须结合已知类准确率与覆盖率判断。新类检测不等于完成新节点发现、命名和插入。';
let cols=g.task==='onepass'||g.task==='hierarchical'?route:stream;
const value=(r,k)=>{
 if(k==='resource_native_seconds')return r.resources?.native_seconds;
 if(k==='resource_maintenance_seconds')return r.resources?.maintenance_seconds;
 if(k==='resource_process_seconds')return r.resources?.process_seconds??r.resources?.supervisor?.process_job_seconds;
 if(k==='resource_peak_bytes')return r.resources?.peak_rss_bytes??r.resources?.worker?.process_peak_rss_bytes;
 if(k==='terminal_microclusters')return r.terminal_state?.n_micro_clusters;
 return r.summary[k];
};
if(g.task.includes('temporal'))cols=[...cols,['resource_native_seconds','纯算法 / s ↓'],['resource_maintenance_seconds','维护 / s（重叠）'],['resource_process_seconds','进程总时间 / s ↓'],['resource_peak_bytes','进程 RSS / MiB ↓'],['terminal_microclusters','终态微簇数']];
cols=cols.filter(([k])=>rows.some(r=>value(r,k)!==undefined));$('head').replaceChildren();const hr=document.createElement('tr');['方法 / 观测次数',...cols.map(c=>c[1])].forEach(t=>{const th=document.createElement('th');th.textContent=t;hr.append(th);});$('head').append(hr);$('body').replaceChildren();
for(const r of rows){const tr=document.createElement('tr');[r.algorithm+' / '+(r.seeds.length||'单次时间流'),...cols.map(([k])=>format(value(r,k),k))].forEach(t=>{const td=document.createElement('td');td.textContent=t;tr.append(td);});$('body').append(tr);}
$('caveats').replaceChildren(...g.caveats.map(t=>{const li=document.createElement('li');li.textContent=t;return li;}));$('settings').textContent=JSON.stringify(g.settings,null,2);$('details').replaceChildren(...rows.map(r=>{const d=document.createElement('details');const s=document.createElement('summary');s.textContent=r.algorithm;const p=document.createElement('pre');p.style.cssText='white-space:pre-wrap;overflow-wrap:anywhere';p.textContent=JSON.stringify({summary:r.summary,resources:r.resources,terminal_state:r.terminal_state},null,2);d.append(s,p);return d;}));}
function views(){const g=data.groups.find(g=>g.id===$('group').value);options($('view'),[...new Set(g.records.map(r=>r.view))].map(v=>[v,v]));draw();}
function groups(){options($('group'),data.groups.filter(g=>g.task===$('task').value).map(g=>[g.id,g.id+' · '+g.protocol+' · '+[...new Set(g.records.map(r=>r.algorithm))].join(' / ')]));views();}
fetch('scenarios.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json();}).then(d=>{data=d;options($('task'),Object.entries(labels));const task=new URLSearchParams(location.search).get('task');if(task in labels)$('task').value=task;groups();$('task').onchange=groups;$('group').onchange=views;$('view').onchange=draw;}).catch(e=>{$('scope').textContent='结果读取失败：'+e.message+'。请通过 HTTP 服务访问。';});
