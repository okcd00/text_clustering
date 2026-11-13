/* Curated choices, numerical evidence loaded from the published snapshot. */
'use strict';
const scenarioChoices=[
 {id:'scenario-01',title:'新文本路由 · 多原型优先，单原型省时',scope:'WOS · 静态开放集 · 5 seeds',names:['单原型','三原型'],why:'三原型略提高已知类准确率并降低误接收，但隐藏类误接收仍高，不能作为可靠自动接纳新类的方案。',route:true},
 {id:'scenario-02',title:'层级导航 · 多原型小幅改善，单原型轻量',scope:'WOS · 层级开放集 · 5 seeds',names:['单原型','三原型'],why:'父级与新类联合判断仍弱。适合作为导航候选，实际新节点发现、命名和插入尚未验证。',route:true},
 {id:'scenario-06',title:'已知 K 持续更新 · MiniBatch 候选',scope:'BANKING77 · 随机顺序 · 固定 K · 5 seeds',names:['冻结 MiniBatch','更新 MiniBatch'],why:'此组更新提高窗口质量，状态由固定 K 控制；不能据此断言优于未知 K 方法，也不代表能追踪真实时间漂移。'},
 {id:'scenario-11',title:'未知 K 持续更新 · BIRCH 候选',scope:'BANKING77 · 随机顺序 · 5 seeds',names:['冻结 BIRCH','更新 BIRCH'],why:'此组质量增益较小，作业时间与内存增加。允许状态增长时再考虑持续更新，不应仅因支持增量就默认开启。'},
 {id:'scenario-18',title:'时间流与容量 · 预算优先选容量冻结',scope:'新闻日历流 · raw 视图 · 单次回放',names:['无上限 BIRCH','容量冻结','滚动重建'],why:'容量冻结在此回放保留较多质量且降低资源；无上限质量更高但状态增长，滚动重建状态更小。银标签、近似发布时间；已查看测试集的描述性比较，不是泛化胜出。',capacity:true}
];
function recommendationNode(tag,text,cls){const n=document.createElement(tag);n.textContent=text;if(cls)n.className=cls;return n;}
function recommendationValue(r,key){return r.summary[key]?.mean;}
function recommendationFormat(v,percent=false){return !Number.isFinite(v)?'未测':(percent?(v*100).toFixed(1)+'%':v.toFixed(2));}
fetch('scenarios.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json();}).then(data=>{
 const container=document.getElementById('scenario-choices');
 for(const c of scenarioChoices){const g=data.groups.find(g=>g.id===c.id);if(!g)throw Error('Missing evidence '+c.id);
 const rows=g.records.filter(r=>!c.capacity||r.view==='raw');const card=recommendationNode('article','');card.append(recommendationNode('b',c.scope),recommendationNode('h3',c.title));
 const metrics=recommendationNode('div','','choice-metrics');
 rows.forEach((r,i)=>{const item=recommendationNode('div','','choice-row');item.append(recommendationNode('strong',c.names[i]));
 const get=k=>recommendationValue(r,k);let text;
 if(c.route)text=`已知准确率 ${recommendationFormat(get('known_accuracy_all'),true)} · 隐藏误接收 ${recommendationFormat(get('hidden_false_accept_rate'),true)} ↓ · 拟合+测试 ${recommendationFormat(get('fit_predict_seconds'))} s`;
 else if(c.capacity)text=`窗口 ACC ${recommendationFormat(get('window_weighted_acc_all_noise_error'),true)} · 进程 ${recommendationFormat(r.resources?.process_seconds)} s · RSS ${recommendationFormat(r.resources?.peak_rss_bytes/1048576)} MiB · 终态 ${r.terminal_state?.n_micro_clusters??'未测'} 微簇`;
 else text=`窗口 ACC ${recommendationFormat(get('window_weighted_acc_all_noise_error'),true)} · 作业 ${recommendationFormat(get('process_job_seconds'))} s · RSS ${recommendationFormat(get('process_peak_rss_bytes')/1048576)} MiB`;
 item.append(recommendationNode('span',text));metrics.append(item);});card.append(metrics,recommendationNode('p',c.why));
 const link=recommendationNode('a','查看本组完整指标与协议 →');link.href=`scenarios.html?task=${g.task}&group=${g.id}`;card.append(link);container.append(card);
 }
 document.getElementById('choice-status').textContent='卡片展示本组均值；标准差、覆盖率及完整资源口径见详情。时间为整次运行，不是单文本延迟；不含历史 embedding 获取。各卡片独立解读，不构成跨协议排名。';
}).catch(()=>{document.getElementById('choice-status').textContent='推荐证据读取失败，请通过 HTTP 访问或打开场景评测页。';});
