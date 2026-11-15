const fs=require('fs'),vm=require('vm'),assert=require('assert');
class N{
 constructor(tag='div'){this.tag=tag;this.children=[];this.value='';this.textContent='';this.dataset={};this.style={};this.attrs={};this.className='';this.classList={toggle:(c,on)=>{const classes=new Set(this.className.split(' ').filter(Boolean));on?classes.add(c):classes.delete(c);this.className=[...classes].join(' ');}};}
 append(...nodes){this.children.push(...nodes);if(this.tag==='select'&&!this.value&&nodes.length)this.value=nodes[0].value;}
 replaceChildren(){this.children=[];if(this.tag==='select')this.value='';}
 setAttribute(k,v){this.attrs[k]=v;} scrollIntoView(){}
}
const ids=['cohort','metric','comparison-chart','comparison-rows','comparison-runs','cohort-context','metric-protocol','compare-status','deferred-methods'];
async function render(data){const nodes=Object.fromEntries(ids.map(id=>[id,new N(['cohort','metric'].includes(id)?'select':'div')]));vm.runInNewContext(fs.readFileSync('docs/unified.js','utf8'),{document:{getElementById:id=>nodes[id],createElement:tag=>new N(tag)},fetch:async()=>({ok:true,json:async()=>data})});await new Promise(r=>setImmediate(r));return nodes;}
const all=(n,p)=>[...(p(n)?[n]:[]),...n.children.flatMap(c=>all(c,p))];
const cls=(n,c)=>all(n,x=>x.className.split(' ').includes(c));
const txt=n=>n.textContent+' '+n.children.map(txt).join(' ');
const data=()=>JSON.parse(fs.readFileSync('docs/unified-results.json'));
(async()=>{
 let d=data(),n=await render(d);assert.equal(n.cohort.children.length,4);assert.equal(n['comparison-chart'].children.length,6);
 let ari=n['comparison-chart'].children[0];assert.equal(cls(ari,'interval-group').length,2);assert.equal(cls(ari,'interval-axis').length,1);assert.equal(cls(ari,'interval-mean').length,5);assert(txt(n['metric-protocol']).includes('不能据此估计 K 增益'));
 const chartrows=cls(ari,'interval-method');for(const row of chartrows){const source=d.cohorts.find(c=>c.dataset==='banking77'&&c.scenario===row.dataset.budget);const method=cls(row,'interval-label')[0].children[0].textContent;const id=d.methods.find(m=>m.title===method).id;const xs=d.runs.filter(r=>r.cohort_id===source.id&&r.method===id).map(r=>r.metrics.ari_all_noise_as_one_cluster);const avg=xs.reduce((a,b)=>a+b)/xs.length;assert.equal(cls(row,'interval-label')[0].children[1].textContent,avg.toFixed(3));assert(Math.abs(parseFloat(cls(row,'interval-band')[0].style.width)-50*(Math.max(...xs)-Math.min(...xs)))<1e-10);}
 n.metric.value='process_peak_rss_bytes';n.metric.onchange();assert(txt(n['comparison-rows']).includes('KMeans'));assert.equal(n['comparison-runs'].children.length,15);
 n.cohort.value='banking77__train-lexical__routing';n.cohort.onchange();assert.equal(n['comparison-chart'].children.length,5);assert(txt(n['comparison-runs']).includes('known='));assert(!txt(n['comparison-chart']).includes('oracle'));
 // A globally missing declared seed must not be mistaken for n=2 complete.
 d=data();d.runs=d.runs.filter(r=>r.seed!==2);n=await render(d);assert.equal(cls(n['comparison-chart'],'interval-mean').length,0);assert(txt(n['comparison-runs']).includes('缺失记录'));
 d=data();const target=d.runs.find(r=>r.cohort_id==='banking77__e5-small-pinned__unknown-k-fixed');target.status='failed';target.error='fixture OOM';n=await render(d);assert.equal(cls(n['comparison-chart'].children[0],'interval-mean').length,2);assert(txt(n['comparison-chart']).includes('fixture OOM'));
 d=data();d.runs.push({...d.runs[0]});n=await render(d);assert.equal(cls(n['comparison-chart'].children[0],'interval-mean').length,3);assert(txt(n['comparison-runs']).includes('重复记录'));
 d=data();d.runs[0].metrics.ari_all_noise_as_one_cluster=null;n=await render(d);assert.equal(cls(n['comparison-chart'].children[0],'interval-mean').length,3);assert.equal(cls(n['comparison-chart'].children[1],'interval-mean').length,5);
 d=data();delete d.protocol;d.runs=d.runs.filter(r=>r.seed===0);n=await render(d);assert(txt(n['comparison-chart']).includes('单次运行，无重复区间'));
 d=data();d.runs.forEach(r=>r.metrics.coverage=1);n=await render(d);assert(txt(n['comparison-chart']).includes('各次同值'));
 console.log('Comparison renderer: PASS (shared K axes, source-value parity, routing, missing/duplicate/null/failure, single/degenerate)');
})();
