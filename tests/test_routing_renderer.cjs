const fs=require('fs'),vm=require('vm'),assert=require('assert');
class N{constructor(){this.children=[];this.value='banking77'}append(...xs){this.children.push(...xs)}replaceChildren(){this.children=[]}addEventListener(){}}
async function run(data){const nodes={};for(const id of ['dataset','rows','counts','status','back-evidence'])nodes[id]=new N();vm.runInNewContext(fs.readFileSync('docs/routing.js','utf8'),{document:{querySelector:s=>nodes[s.slice(1)],createElement:()=>new N()},fetch:async()=>({ok:true,json:async()=>data})});await new Promise(r=>setImmediate(r));return nodes}
(async()=>{
 const data=JSON.parse(fs.readFileSync('docs/unified-results.json'));
 let n=await run(data);assert(n['back-evidence'].href.includes('dataset=banking77')); assert(n.rows.children.every(r=>r.children.slice(1).every(c=>c.textContent.endsWith('%'))));
 const row=data.runs.find(r=>r.method==='prototype'&&r.cohort_id.startsWith('banking77'));
 row.metrics.known_accepted_accuracy=null;data.status='partial';n=await run(data);
 assert(n.status.textContent.includes('部分完成'));
 assert(n.rows.children.every(r=>r.children[2].textContent==='比较暂停'));
 assert(n.rows.children.every(r=>r.children[1].textContent.endsWith('%')));
 row.status='failed';row.error='OOM';n=await run(data);
 assert(n.rows.children.every(r=>r.children.slice(1).every(c=>c.textContent==='比较暂停')));
 assert(n.status.textContent.includes('OOM'));
 data.runs=data.runs.filter(r=>!(r.method==='class-max-1nn'&&r.cohort_id.startsWith('banking77')));
 n=await run(data);assert(n.rows.children.every(r=>r.children.slice(1).every(c=>c.textContent==='比较暂停')));
 console.log('Routing renderer: PASS (complete, null metric, failure, missing method)');
})();
