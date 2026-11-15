(() => {
 const $=id=>document.getElementById(id); let data, views;
 const finite=x=>typeof x==='number'&&Number.isFinite(x);
 const mean=xs=>xs.reduce((a,b)=>a+b,0)/xs.length;
 const metricLists={static:[['ari_all_noise_as_one_cluster','全样本 ARI','↑ 越高越好'],['nmi_all_noise_as_one_cluster','全样本 NMI','↑ 越高越好'],['coverage','覆盖率','↑ 越高越好'],['ari_covered','接收样本 ARI','↑ 须配合覆盖率'],['fit_predict_seconds','聚类耗时','↓ 秒'],['process_peak_rss_bytes','进程峰值 RSS','↓ MiB']],routing:[['known_coverage','已知类覆盖率','↑ 全部已知样本'],['known_accepted_accuracy','接收准确率','↑ 已接收已知样本'],['known_correct_acceptance','正确接收率','↑ 全部已知样本'],['hidden_false_acceptance','隐藏类误接收率','↓ 全部隐藏样本'],['test_routing_seconds','测试路由耗时','↓ 秒']]};
 const text=(tag,value,cls)=>{const e=document.createElement(tag);e.textContent=value;if(cls)e.className=cls;return e;};
 const rate=key=>['coverage','known_coverage','known_accepted_accuracy','known_correct_acceptance','hidden_false_acceptance'].includes(key);
 const cost=key=>key.endsWith('_seconds')||key==='process_peak_rss_bytes';
 const value=(r,key)=>key in r?r[key]:r.metrics?.[key];
 const format=(x,key)=>!finite(x)?'未提供':rate(key)?`${(x*100).toFixed(1)}%`:key==='process_peak_rss_bytes'?(x/1048576).toFixed(1):x.toFixed(3);
 const niceCeiling=x=>{if(x<=0)return 1;const p=10**Math.floor(Math.log10(x));return [1,2,5,10].find(n=>n*p>=x)*p;};
 const choices=c=>metricLists[c.scenario==='routing'?'routing':'static'];
 const trackLabel=c=>({'known-k':'已知 K · oracle 信息','unknown-k-fixed':'未知 K · 固定参数',routing:'路由 · 支持集与校准集'}[c.scenario]||c.scenario);
 function groupedViews(cohorts){
  const out=[], staticGroups=new Map();
  cohorts.forEach(c=>{
   if(!['known-k','unknown-k-fixed'].includes(c.scenario)){out.push({...c,source_cohorts:[c]});return;}
   const key=JSON.stringify([c.dataset,c.representation,c.n,c.dimensions]);
   if(!staticGroups.has(key)){const view={...c,id:`static-${out.length}`,scenario:'static-compare',source_cohorts:[]};staticGroups.set(key,view);out.push(view);}
   staticGroups.get(key).source_cohorts.push(c);
  });return out;
 }
 function summaries(c,key){return c.source_cohorts.map(source=>({...summarize(source,key),source}));}
 function seedsFor(c,runs){
  // The exported v1 grid is locked to 0/1/2: even a seed missing for ALL methods must fail closed.
  return c.expected_seeds||data.expected_seeds||(data.protocol==='survey-comparison-v1'?[0,1,2]:[...new Set(runs.map(r=>r.seed))]);
 }
 function summarize(c,key){
  const runs=data.runs.filter(r=>r.cohort_id===c.id),seeds=seedsFor(c,runs);
  const ids=[...new Set([...(data.methods||[]).filter(m=>m.track===c.scenario).map(m=>m.id),...runs.map(r=>r.method)])];
  const rows=ids.map(id=>{const rs=runs.filter(r=>r.method===id),xs=rs.filter(r=>r.status==='ok').map(r=>value(r,key)).filter(finite);const validN=seeds.filter(seed=>{const found=rs.filter(x=>x.seed===seed);return found.length===1&&found[0].status==='ok'&&finite(value(found[0],key));}).length;return{id,name:data.methods?.find(m=>m.id===id)?.title||id,rs,xs,validN};});
  const comparable=seeds.length>0&&rows.length>0&&rows.every(r=>r.rs.length===seeds.length&&seeds.every(seed=>{const found=r.rs.filter(x=>x.seed===seed);return found.length===1&&found[0].status==='ok'&&finite(value(found[0],key));}));
  return {rows,seeds,comparable};
 }
 function metricCard(c,entry){
  const [key,title,direction]=entry,groups=summaries(c,key),comparable=groups.every(g=>g.comparable);
  const rows=groups.flatMap(g=>g.rows.map(r=>({...r,source:g.source,seeds:g.seeds,comparable:g.comparable})));
  const card=text('section','','interval-card');card.dataset.metric=key;card.setAttribute('aria-label',`${title}，${direction}`);
  const head=text('div','','interval-heading');head.append(text('h3',title),text('span',direction,'metric-direction'));card.append(head);
  card.append(text('p',`${c.dataset} · ${c.scenario==='static-compare'?'两种 K 信息预算 · 共用坐标轴':trackLabel(c)}`,'card-cohort'));
  if(!comparable)card.append(text('p','部分信息预算组暂停：失败、缺失、重复或无有效指标。完整组保留；不平均不同 seed 子集。','run-warning'));
  const values=rows.filter(r=>r.comparable).flatMap(r=>r.xs),lo=key.startsWith('ari_')?-1:0;
  const hi=cost(key)?niceCeiling(Math.max(0,...values)/(key==='process_peak_rss_bytes'?1048576:1))*(key==='process_peak_rss_bytes'?1048576:1):1;
  let lastGroup;
  rows.forEach(r=>{
   if(lastGroup!==r.source.id){const badge=text('p',trackLabel(r.source)+(r.comparable?'':' · 比较暂停'),'interval-group');badge.dataset.budget=r.source.scenario;card.append(badge);lastGroup=r.source.id;}
   const row=text('div','','interval-method');row.dataset.budget=r.source.scenario;const label=text('div','','interval-label');label.append(text('span',r.name),text('strong',r.comparable?format(mean(r.xs),key):'—'));row.append(label);
   if(r.comparable){
    const min=Math.min(...r.xs),max=Math.max(...r.xs),avg=mean(r.xs),pos=x=>`${100*(x-lo)/(hi-lo)}%`;
    const plot=text('div','','interval-track');plot.setAttribute('aria-hidden','true');
    const zero=text('i','','interval-zero');zero.style.left=pos(0);plot.append(zero);
    const band=text('i','','interval-band');band.style.left=pos(min);band.style.width=`${100*(max-min)/(hi-lo)}%`;plot.append(band);
    const dot=text('i','','interval-mean');dot.style.left=pos(avg);plot.append(dot);row.append(plot);
    const desc=r.xs.length===1?'单次运行，无重复区间':min===max?'各次同值，区间退化为点':`范围 ${format(min,key)}–${format(max,key)}`;
    row.append(text('p',`${desc} · n=${r.xs.length}/${r.seeds.length}`,'interval-values'));
   }else{
    const bad=r.rs.filter(x=>x.status!=='ok').map(x=>x.error||x.status);
    row.append(text('p',`有效指标 n=${r.validN}/${r.seeds.length}${bad.length?' · '+bad.join('；'):''}`,'interval-values'));
   }
   card.append(row);
  });
  if(rows.some(r=>r.comparable)){const axis=text('div','','interval-axis');axis.append(text('span',format(lo,key)),text('span',format(hi,key)));card.append(axis);}
  card.append(text('p',cost(key)?'仅本组同轴；进程/JIT/缓存等成本口径见说明。':'固定指标尺度；不合成总分，不将算法差异归因于 K。','card-footnote'));
  return card;
 }
 function drawDetails(c){
  const key=$('metric').value,groups=summaries(c,key);
  const rows=groups.flatMap(g=>g.rows.map(r=>({...r,seeds:g.seeds,comparable:g.comparable,source:g.source})));
  $('comparison-rows').replaceChildren();$('comparison-runs').replaceChildren();
  rows.forEach(r=>{
   const tr=text('tr','');tr.append(text('th',`${r.name} · ${trackLabel(r.source)}`),text('td',r.comparable?format(mean(r.xs),key):'比较暂停'),text('td',r.comparable?`${format(Math.min(...r.xs),key)}–${format(Math.max(...r.xs),key)}`:'—'),text('td',`${r.validN} / ${r.seeds.length}`));$('comparison-rows').append(tr);
   const actualSeeds=[...new Set([...r.seeds,...r.rs.map(x=>x.seed)])];
   actualSeeds.forEach(seed=>{
    const matches=r.rs.filter(x=>x.seed===seed);
    (matches.length?matches:[null]).forEach(run=>{
     const line=text('tr','');let state=!run?'缺失记录':matches.length>1?'重复记录':run.status!=='ok'?(run.error||run.status):!finite(value(run,key))?'指标未提供':'完成';
     if(run&&c.scenario==='routing')state+=` · known=${run.metrics?.known_n??'—'}, accepted=${run.metrics?.known_accepted_n??'—'}, hidden=${run.metrics?.hidden_n??'—'}`;
     line.append(text('th',`${r.name} · ${trackLabel(r.source)}`),text('td',String(seed)),text('td',run?format(value(run,key),key):'—'),text('td',state));$('comparison-runs').append(line);
    });
   });
  });
  [...$('comparison-chart').children].forEach(card=>{card.classList.toggle('is-focused',card.dataset.metric===key);});
 }
 function draw(){
  const c=views.find(x=>x.id===$('cohort').value);$('comparison-chart').replaceChildren();if(!c)return;
  $('cohort-context').textContent=`${c.dataset} · ${c.representation} · ${c.n} 条 · ${c.dimensions} 维 · ${c.scenario==='static-compare'?'已知 K 与未知 K 同轴对照；其他输入相同':trackLabel(c)+' · '+(typeof c.information_budget==='string'?c.information_budget:JSON.stringify(c.information_budget))}`;
  $('metric-protocol').textContent=c.scenario==='routing'?'路由单独比较：接收准确率须同时看覆盖率和正确接收率；隐藏类误接收率越低越好。阈值仅由校准集选择，不混入静态 ARI。':'静态聚类：圆点＝已知 K（oracle），方点＝未知 K。两组同轴便于观察，但算法不同，不是配对消融；当前没有经协议验证的同算法双 K 配对，不能据此估计 K 增益。全样本 ARI/NMI 把噪声视为一簇，接收样本 ARI 须配合覆盖率。';
  const metrics=choices(c);metrics.forEach(entry=>$('comparison-chart').append(metricCard(c,entry)));
  const ready=metrics.filter(([key])=>summaries(c,key).every(g=>g.comparable)).length;
  $('compare-status').className=ready<metrics.length||data.status==='partial'?'run-warning':'';
  $('compare-status').textContent=`${ready} / ${metrics.length} 个维度完整可比较 · 填色是各次实际值的 min–max，点是均值，n 是有效运行数。不是置信区间，也不是总分。${ready<metrics.length?' 不完整维度暂停比较，原始状态仍保留。':''}${data.status==='partial'?' 本轮整体部分完成。':''}`;
  drawDetails(c);
 }
 function selectMetrics(){const c=views.find(x=>x.id===$('cohort').value);if(!c)return;const previous=$('metric').value;$('metric').replaceChildren();choices(c).forEach(([value,label,direction])=>{const o=text('option',`${label} · ${direction}`);o.value=value;$('metric').append(o)});$('metric').value=choices(c).some(x=>x[0]===previous)?previous:choices(c)[0][0];draw();}
 fetch('unified-results.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json()}).then(d=>{
  data=d;if(!['executed','partial'].includes(d.status)||!d.cohorts?.length){$('compare-status').textContent='本轮统一评估尚未完成。暂不展示历史数字替代实测。';return;}
  views=groupedViews(d.cohorts);
  views.forEach(c=>{const o=text('option',`${c.dataset} / ${c.representation} / ${c.scenario==='static-compare'?'已知 K + 未知 K 同轴':trackLabel(c)}`);o.value=c.id;$('cohort').append(o)});
  (d.deferred||[]).forEach(m=>$('deferred-methods').append(text('p',typeof m==='string'?m:JSON.stringify(m),'run-warning')));
  $('cohort').onchange=selectMetrics;$('metric').onchange=()=>{drawDetails(views.find(c=>c.id===$('cohort').value));[...$('comparison-chart').children].find(card=>card.dataset.metric===$('metric').value)?.scrollIntoView({block:'start',behavior:'instant'});};selectMetrics();
 }).catch(()=>{$('compare-status').textContent='统一评估结果暂时无法读取，请刷新或下载仓库中的结果；不会以历史数字替代。';});
})();
