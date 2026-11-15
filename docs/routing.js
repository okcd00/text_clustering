fetch('unified-results.json').then(r=>{if(!r.ok)throw Error('读取失败');return r.json()}).then(data=>{
 const $=id=>document.querySelector('#'+id);
 if(!['executed','partial'].includes(data.status))throw Error('没有实测结果');
 const sel=$('dataset'), names={'prototype':'Prototype · 类原型','class-max-1nn':'Class-max 1NN · 类别最大近邻'};
 const keys=['known_coverage','known_accepted_accuracy','known_correct_acceptance','hidden_false_acceptance'];
 function render(){
  const cohort=data.cohorts.find(c=>c.dataset===sel.value&&c.scenario==='routing');
  if(!cohort)throw Error('没有实测结果');
  const back=$('back-evidence');if(back)back.href='survey.html?view=routing&dataset='+encodeURIComponent(cohort.dataset)+'#compare';
  const all=data.runs.filter(r=>r.cohort_id===cohort.id),seeds=[...new Set(all.map(r=>r.seed))];
  const counts=all.find(r=>r.status==='ok')?.metrics;
  $('counts').textContent=counts?`相同测试样本：${cohort.n} 条，其中已知 ${counts.known_n} 条，隐藏 ${counts.hidden_n} 条。`:'尚无有效运行。';
  // A missing/null metric suppresses that column for every method.
  const comparable=keys.map(k=>seeds.length>0&&Object.keys(names).every(method=>{
   const rs=all.filter(r=>r.method===method);
   return rs.length===seeds.length&&seeds.every(seed=>{
    const matches=rs.filter(r=>r.seed===seed);
    return matches.length===1&&matches[0].status==='ok'&&typeof matches[0].metrics?.[k]==='number'&&Number.isFinite(matches[0].metrics[k]);
   });
  }));
  const body=$('rows');body.replaceChildren();
  for(const method of Object.keys(names)){
   const rs=all.filter(r=>r.method===method),tr=document.createElement('tr'),title=document.createElement('td');
   title.textContent=names[method];tr.append(title);
   keys.forEach((k,i)=>{
    const td=document.createElement('td');
    td.textContent=comparable[i]?`${(100*rs.reduce((s,r)=>s+r.metrics[k],0)/rs.length).toFixed(1)}%`:'比较暂停';
    tr.append(td);
   });body.append(tr);
  }
  const errors=all.filter(r=>r.status!=='ok').map(r=>`${r.method} seed ${r.seed}: ${r.error||r.status}`);
  $('status').textContent=(data.status==='partial'?'本轮评估部分完成。':'')+
   (comparable.every(Boolean)?'实测记录 · routing-v2':'存在失败、缺失记录或指标，相应指标已暂停比较，避免不同 seed 子集的均值。')+
   ` 有效运行 ${all.filter(r=>r.status==='ok').length} / ${all.length}。`+
   (errors.length?' '+errors.join('；'):'');
 }
 sel.addEventListener('change',render);render();
}).catch(()=>{document.querySelector('#status').textContent='结果暂时无法载入，请查看完整协议或稍后重试。'});
