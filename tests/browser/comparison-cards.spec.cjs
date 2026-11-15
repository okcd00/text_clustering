const {test,expect}=require('@playwright/test');
function fixture(seeds=[0,1,2]){
 const cohorts=['known-k','unknown-k-fixed'].map((scenario,i)=>({id:`c${i}`,dataset:'Paired inputs',representation:'fixed',scenario,n:10,dimensions:4,information_budget:scenario}));
 const methods=[{id:'a',title:'Known method',track:'known-k'},{id:'b',title:'Unknown method',track:'unknown-k-fixed'}];
 const runs=methods.flatMap((m,i)=>seeds.map(seed=>({cohort_id:`c${i}`,method:m.id,seed,status:'ok',fit_predict_seconds:seed+1,process_peak_rss_bytes:1048576*(seed+1),metrics:{ari_all_noise_as_one_cluster:(i?-.1:.2)+seed*.1,nmi_all_noise_as_one_cluster:.5,ari_covered:.5,coverage:1}})));
 return {status:'executed',protocol:'survey-comparison-v1',cohorts,methods,runs,deferred:[]};
}
async function load(page,data){await page.route('**/unified-results.json',r=>r.fulfill({json:data}));await page.goto('/survey.html#compare');await expect(page.locator('.interval-card')).toHaveCount(6);}
test('same-axis K groups, ranges, mean, raw values and mobile layout',async({page})=>{
 const errors=[];page.on('pageerror',e=>errors.push(e.message));await load(page,fixture());
 await expect(page.locator('#cohort option')).toHaveCount(1);
 const ari=page.locator('.interval-card').first();await expect(ari.locator('.interval-group')).toHaveCount(2);
 await expect(ari.locator('.interval-axis')).toHaveCount(1);await expect(ari).toContainText('-1.000');
 await expect(ari).toContainText('0.300');await expect(ari).toContainText('范围 -0.100–0.100');
 await expect(ari.locator('.interval-method[data-budget="known-k"] .interval-band')).toHaveCSS('background-color','rgb(227, 198, 159)');
 const width=await ari.locator('.interval-band').first().evaluate(e=>parseFloat(e.style.width));expect(width).toBeCloseTo(10);
 await expect(page.locator('#metric-protocol')).toContainText('不能据此估计 K 增益');
 await expect(page.locator('.interval-card[data-metric="coverage"]')).toContainText('各次同值');
 await page.locator('#metric').selectOption('fit_predict_seconds');await expect(page.locator('.interval-card[data-metric="fit_predict_seconds"]')).toHaveClass(/is-focused/);
 await page.locator('.comparison-details>summary').click();await expect(page.locator('#comparison-rows')).toBeVisible();await expect(page.locator('#comparison-runs tr')).toHaveCount(6);await expect(page.locator('#comparison-runs')).toContainText('3.000');
 expect(await page.evaluate(()=>document.documentElement.scrollWidth-innerWidth)).toBeLessThanOrEqual(1);expect(errors).toEqual([]);
});
test('missing seed across ALL methods and duplicate records fail closed',async({page})=>{
 await load(page,fixture([0,1]));await expect(page.locator('.interval-mean')).toHaveCount(0);await expect(page.locator('#comparison-runs')).toContainText('缺失记录');
});
test('failed group pauses only that group; finite subset never averaged',async({page})=>{
 const d=fixture();d.runs.find(r=>r.method==='b'&&r.seed===1).status='failed';d.status='partial';await load(page,d);
 await expect(page.locator('.interval-card').first().locator('.interval-mean')).toHaveCount(1);
 await expect(page.locator('.interval-card').first()).toContainText('比较暂停');await expect(page.locator('#comparison-rows')).toContainText('比较暂停');
});
test('null metric, duplicate, single run and fetch failure stay honest',async({page})=>{
 let d=fixture();d.runs[0].metrics.ari_all_noise_as_one_cluster=null;d.runs.push({...d.runs[3]});await load(page,d);
 await expect(page.locator('.interval-card').first().locator('.interval-mean')).toHaveCount(0);
 await page.unroute('**/unified-results.json');d=fixture([0]);delete d.protocol;await page.route('**/unified-results.json',r=>r.fulfill({json:d}));await page.reload();await expect(page.locator('.interval-card').first()).toContainText('单次运行，无重复区间');
 await page.unroute('**/unified-results.json');await page.route('**/unified-results.json',r=>r.abort());await page.reload();await expect(page.locator('#compare-status')).toContainText('暂时无法读取');await expect(page.locator('.interval-card')).toHaveCount(0);
});
test('real published groups and routing remain independently selectable',async({page})=>{
 await page.goto('/survey.html#compare');await expect(page.locator('#cohort option')).toHaveCount(4);
 await expect(page.locator('.interval-card')).toHaveCount(6);await expect(page.locator('.interval-card').first().locator('.interval-method')).toHaveCount(5);
 await page.locator('#cohort').selectOption('banking77__train-lexical__routing');await expect(page.locator('.interval-card')).toHaveCount(5);await expect(page.locator('#metric-protocol')).toContainText('路由单独比较');await expect(page.locator('.interval-card').first()).not.toContainText('oracle');
 await page.locator('.comparison-details>summary').click();await expect(page.locator('#comparison-runs')).toContainText('known=');
 await page.locator('#cohort').selectOption('static-0');await expect(page.locator('.interval-card')).toHaveCount(6);expect(await page.evaluate(()=>document.documentElement.scrollWidth-innerWidth)).toBeLessThanOrEqual(1);
});
test('stable routing deep link survives reload, selection and Back',async({page})=>{
 await page.goto('/survey.html?view=routing&dataset=massive_zh_cn#compare');
 await expect(page.locator('.interval-card')).toHaveCount(5);
 await expect(page.locator('#cohort-context')).toContainText('massive_zh_cn');
 await page.reload();await expect(page.locator('.interval-card')).toHaveCount(5);
 await page.locator('#cohort').selectOption({index:0});await expect(page).toHaveURL(/view=static/);
 await page.goBack();await expect(page.locator('.interval-card')).toHaveCount(5);
 await expect(page.locator('#cohort-context')).toContainText('massive_zh_cn');
 await page.goto('/survey.html?view=routing&dataset=bad#compare');
 await expect(page.locator('#compare-status')).toContainText('已显示默认组');
});
