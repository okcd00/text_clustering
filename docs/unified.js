(() => {
 const $=id=>document.getElementById(id); let data;
 const number=x=>typeof x==='number'&&Number.isFinite(x);
 const mean=xs=>xs.reduce((a,b)=>a+b,0)/xs.length;
 const metricLists={static:[['ari_all_noise_as_one_cluster','全样本 ARI ↑'],['nmi_all_noise_as_one_cluster','全样本 NMI ↑'],['coverage','覆盖率 ↑'],['ari_covered','接收样本 ARI ↑'],['fit_predict_seconds','聚类耗时 / 秒 ↓'],['process_peak_rss_bytes','进程峰值内存 / MiB ↓']],routing:[['known_coverage','已知类覆盖率 ↑'],['known_accepted_accuracy','已接收已知类准确率 ↑'],['known_correct_acceptance','已知类正确接收率 ↑'],['hidden_false_acceptance','隐藏类误接收率 ↓'],['test_routing_seconds','测试路由耗时 / 秒 ↓']]};
 function selectMetrics(){const c=data.cohorts.find(x=>x.id===$('cohort').value);if(!c)return;const choices=metricLists[c.scenario==='routing'?'routing':'static'];const previous=$('metric').value;$('metric').replaceChildren();choices.forEach(([value,label])=>{const o=text('option',label);o.value=value;$('metric').append(o)});$('metric').value=choices.some(x=>x[0]===previous)?previous:choices[0][0];draw();}
 const text=(tag,value,cls)=>{const e=document.createElement(tag);e.textContent=value;if(cls)e.className=cls;return e;};
 const format=(x,key)=>number(x)?(key==='process_peak_rss_bytes'?x/1048576:x).toFixed(3):'未提供';
 function draw(){
  const c=data.cohorts.find(x=>x.id===$('cohort').value),key=$('metric').value;
  $('comparison-chart').replaceChildren();$('comparison-rows').replaceChildren();if(!c)return;
  $('cohort-context').textContent=`${c.dataset} · ${c.representation} · ${c.n} 条 · ${c.dimensions} 维 · ${c.scenario} · ${typeof c.information_budget==='string'?c.information_budget:JSON.stringify(c.information_budget)}`;
  $('metric-protocol').textContent=c.scenario==='routing'?'路由独立协议：已知类覆盖率与正确接收率以全部已知样本为分母；接收准确率以已接收已知样本为分母；隐藏类误接收率以全部隐藏样本为分母。阈值只从校准集选择，不混入静态聚类 ARI。':'静态聚类：全样本 ARI/NMI 把噪声视为一簇，接收样本 ARI 须与覆盖率一起看。';
  const runs=data.runs.filter(r=>r.cohort_id===c.id);const ids=[...new Set([...(data.methods||[]).filter(m=>m.track===c.scenario).map(m=>m.id),...runs.map(r=>r.method)])];
  const rows=ids.map(id=>{const rs=runs.filter(r=>r.method===id),good=rs.filter(r=>r.status==='ok'),xs=good.map(r=>key in r?r[key]:r.metrics?.[key]).filter(number);return {id,rs,good,xs,avg:xs.length?mean(xs):null};});
  // Fail closed: never average different successful/finite seed subsets.
  const seeds=[...new Set(runs.map(r=>r.seed))];
  const comparable=seeds.length>0&&rows.length>0&&rows.every(r=>
   r.rs.length===seeds.length&&seeds.every(seed=>{
    const matches=r.rs.filter(x=>x.seed===seed);
    return matches.length===1&&matches[0].status==='ok'&&number(key in matches[0]?matches[0][key]:matches[0].metrics?.[key]);
   }));
  if(!comparable)rows.forEach(r=>{r.avg=null;r.xs=[];});
  const range=rows.flatMap(r=>r.xs);const lo=Math.min(0,...range),hi=Math.max(key.endsWith('_seconds')||key==='process_peak_rss_bytes'?0:1,...range,0.001);
  rows.forEach(r=>{
   const m=data.methods?.find(x=>x.id===r.id);const name=m?.title||r.id;
   const row=text('tr','');row.append(text('th',name),text('td',format(r.avg,key)),text('td',r.xs.length?`${format(Math.min(...r.xs),key)} – ${format(Math.max(...r.xs),key)}`:'—'),text('td',`${r.good.length} / ${r.rs.length}`));$('comparison-rows').append(row);
   const card=text('div','','metric-row');const label=text('div',name,'metric-label');label.append(text('strong',format(r.avg,key)));card.append(label);
   if(number(r.avg)){const track=text('div','','metric-track');track.setAttribute('aria-hidden','true');const zero=text('i','','metric-zero');zero.style.left=`${100*(0-lo)/(hi-lo)}%`;track.append(zero);r.xs.forEach(x=>{const dot=text('i','','metric-dot');dot.style.left=`${100*(x-lo)/(hi-lo)}%`;track.append(dot)});card.append(track)}else card.append(text('p','比较暂停：存在失败、缺失记录或指标；不对不同 seed 子集求均值。','run-warning'));
   if(c.scenario==='routing'){const counts=r.good.map(x=>`seed ${x.seed}: known=${x.metrics?.known_n??'—'}, accepted=${x.metrics?.known_accepted_n??'—'}, hidden=${x.metrics?.hidden_n??'—'}`).join('；');const d=text('details','','routing-counts');d.append(text('summary','查看分母'),text('p',counts));card.append(d);}
   $('comparison-chart').append(card);
   if(r.good.length<r.rs.length) card.append(text('p',`${r.rs.length-r.good.length} 次未完成：`+r.rs.filter(x=>x.status!=='ok').map(x=>x.error||x.status).join('；'),'run-warning'));
  });
  if(comparable){const axis=text('div','','metric-axis');axis.append(text('span',format(lo,key)),text('span',format(hi,key)));$('comparison-chart').append(axis);}
  $('compare-status').className=data.status==='partial'||!comparable?'run-warning':'';
  $('compare-status').textContent=(data.status==='partial'?'本轮评估部分完成。':'')+(!comparable?'当前比较组或指标不完整，已暂停均值及图表比较；有效运行数与失败原因仍保留。':rows.length?`${rows.length} 个方法 · 点表示各 seed 的实际值；数值是均值，不是总分。`:'该组尚无完成的结果。');
 }
 fetch('unified-results.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json()}).then(d=>{
  data=d;if(!['executed','partial'].includes(d.status)||!d.cohorts?.length){$('compare-status').textContent='本轮统一评估尚未完成。暂不展示历史数字替代实测。';return;}
  d.cohorts.forEach(c=>{const o=text('option',`${c.dataset} / ${c.representation} / ${c.scenario}`);o.value=c.id;$('cohort').append(o)});
  (d.deferred||[]).forEach(m=>$('deferred-methods').append(text('p',typeof m==='string'?m:JSON.stringify(m),'run-warning')));
  $('cohort').onchange=selectMetrics;$('metric').onchange=draw;selectMetrics();
 }).catch(()=>{$('compare-status').textContent='统一评估结果暂时无法读取，请刷新或下载仓库中的结果；不会以历史数字替代。';});
})();
