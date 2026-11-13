const fs=require('fs'),vm=require('vm'),assert=require('assert');
class N{constructor(tag){this.tag=tag;this.children=[];}append(...x){this.children.push(...x);}}
const nodes={'scenario-choices':new N('div'),'choice-status':new N('p')};
const context={document:{createElement:t=>new N(t),getElementById:id=>nodes[id]},fetch:async()=>({ok:true,json:async()=>JSON.parse(fs.readFileSync('docs/scenarios.json'))})};
vm.runInNewContext(fs.readFileSync('docs/recommendations.js','utf8'),context);
setTimeout(()=>{assert.equal(nodes['scenario-choices'].children.length,5);const s=JSON.stringify(nodes);assert(!s.includes('NaN'));assert(s.includes('61.2%'));assert(s.includes('2048'));assert(s.includes('group=scenario-18'));console.log('Home recommendation render: PASS (5 cards, numeric evidence, deep links)');},20);
