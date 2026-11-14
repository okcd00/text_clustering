const fs=require('fs'),vm=require('vm'),assert=require('assert');
class N{constructor(tag){this.tag=tag;this.children=[];this.style={};this.value='';}append(...xs){this.children.push(...xs);if(this.tag==='select'&&!this.value&&xs[0])this.value=xs[0].value;}replaceChildren(...xs){this.children=xs;}setAttribute(){}}
async function run(data){
 const nodes={};['cohort','metric','cohort-context','metric-protocol','compare-status','comparison-chart','comparison-rows','deferred-methods'].forEach(id=>nodes[id]=new N(['cohort','metric'].includes(id)?'select':'div'));nodes.metric.value='ari_all_noise_as_one_cluster';
 const context={document:{createElement:t=>new N(t),getElementById:id=>nodes[id]},fetch:async()=>({ok:true,json:async()=>data})};vm.runInNewContext(fs.readFileSync('docs/unified.js','utf8'),context);await new Promise(r=>setImmediate(r));return nodes;
}
(async()=>{
 let n=await run({status:'not_run',cohorts:[],runs:[]});assert(n['compare-status'].textContent.includes('尚未完成'));assert.equal(n['comparison-rows'].children.length,0);
 n=await run({status:'executed',cohorts:[{id:'x',dataset:'fixture',representation:'fixed',n:5,dimensions:2,scenario:'known-k',information_budget:'oracle K'}],methods:[],runs:[{cohort_id:'x',method:'a',seed:0,status:'ok',metrics:{ari_all_noise_as_one_cluster:-.1},fit_predict_seconds:2},{cohort_id:'x',method:'a',seed:1,status:'ok',metrics:{ari_all_noise_as_one_cluster:.3},fit_predict_seconds:4},{cohort_id:'x',method:'b',seed:0,status:'ok',metrics:{ari_all_noise_as_one_cluster:.2},fit_predict_seconds:1},{cohort_id:'x',method:'b',seed:1,status:'ok',metrics:{ari_all_noise_as_one_cluster:.4},fit_predict_seconds:3}]});
 assert.equal(n['comparison-rows'].children.length,2);assert.equal(n['comparison-rows'].children[0].children[1].textContent,'0.100');assert.equal(n['comparison-rows'].children[1].children[1].textContent,'0.300');n.metric.value='fit_predict_seconds';n.metric.onchange();assert.equal(n['comparison-rows'].children[0].children[1].textContent,'3.000');assert(!JSON.stringify(n).includes('NaN'));n=await run({status:'executed',cohorts:[{id:'route',scenario:'routing',information_budget:'calibration only'}],runs:[{cohort_id:'route',method:'prototype',seed:0,status:'ok',metrics:{known_coverage:.8,known_accepted_accuracy:.9,known_correct_acceptance:.72,hidden_false_acceptance:.2,known_n:100,known_accepted_n:80,hidden_n:50},test_routing_seconds:.5}]});assert.equal(n.metric.value,'known_coverage');assert.equal(n.metric.children.length,5);assert.equal(n['comparison-rows'].children[0].children[1].textContent,'0.800');assert(n['metric-protocol'].textContent.includes('校准'));n.metric.value='hidden_false_acceptance';n.metric.onchange();assert.equal(n['comparison-rows'].children[0].children[1].textContent,'0.200');

 const base={status:'partial',cohorts:[{id:'x',scenario:'known-k'}],methods:[{id:'a',track:'known-k'},{id:'b',track:'known-k'}],runs:[{cohort_id:'x',method:'a',seed:0,status:'ok',metrics:{ari_all_noise_as_one_cluster:.1}},{cohort_id:'x',method:'a',seed:1,status:'ok',metrics:{ari_all_noise_as_one_cluster:.9}},{cohort_id:'x',method:'b',seed:0,status:'ok',metrics:{ari_all_noise_as_one_cluster:.5}},{cohort_id:'x',method:'b',seed:1,status:'failed',error:'OOM'}]};
 for(const mutate of [d=>d,d=>{d.runs[3]=Object.assign({},d.runs[2]);return d},d=>{d.runs.pop();return d},d=>{d.runs[3].status='ok';d.runs[3].metrics={ari_all_noise_as_one_cluster:null};return d},d=>{d.runs=d.runs.filter(r=>r.method==='a');return d}]){
  n=await run(mutate(JSON.parse(JSON.stringify(base))));
  assert(n['compare-status'].textContent.includes('部分完成'));
  assert(n['compare-status'].textContent.includes('暂停'));
  assert(n['comparison-rows'].children.every(r=>r.children[1].textContent==='未提供'));
  assert(!JSON.stringify(n).includes('metric-dot'));
 }
 n=await run(base);assert(JSON.stringify(n).includes('OOM'));assert.equal(n['comparison-rows'].children[1].children[3].textContent,'1 / 2');
 const actual=JSON.parse(fs.readFileSync('docs/unified-results.json'));
 n=await run(actual);
 for(const cohort of actual.cohorts){
  n.cohort.value=cohort.id;n.cohort.onchange();
  assert(!n['compare-status'].textContent.includes('暂停'));
  for(const row of n['comparison-rows'].children){
   const method=actual.methods.find(m=>(m.title||m.id)===row.children[0].textContent);
   const rs=actual.runs.filter(r=>r.cohort_id===cohort.id&&r.method===method.id);
   const k=n.metric.value,mean=rs.reduce((s,r)=>s+(k in r?r[k]:r.metrics[k]),0)/rs.length;
   assert.equal(row.children[1].textContent,mean.toFixed(3));
  }
 }
 console.log('Unified renderer: PASS (negative ARI, aggregation, blocked != zero, metric changes, empty state)');
})();
