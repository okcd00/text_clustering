const {test,expect}=require('@playwright/test');
test('light overview keeps deep-link comparison and report return paths',async({page})=>{
 await page.goto('/survey.html');
 await expect(page.locator('#comparison-workbench')).not.toHaveAttribute('open','');
 await page.getByRole('link',{name:'静态聚类：结论与报告 →',exact:true}).click();
 await expect(page.locator('.report-conclusion')).toBeVisible();
 await expect(page.locator('.report-body')).toBeVisible();
 expect(await page.evaluate(()=>document.documentElement.scrollWidth-innerWidth)).toBeLessThanOrEqual(1);
 await page.getByRole('link',{name:'返回 Survey 证据',exact:true}).click();
 await expect(page.locator('#comparison-workbench')).toHaveAttribute('open','');
 await expect(page.locator('.interval-card')).toHaveCount(6);
 await page.goto('/survey.html?view=routing&dataset=massive_zh_cn#compare');
 await expect(page.locator('.interval-card')).toHaveCount(5);
 await expect(page.locator('#cohort-context')).toContainText('massive_zh_cn');
});
test('old roadmap anchors lead to independent conclusion-first reports',async({page})=>{
 for(const [anchor,slug] of [[4,'calendar-paired-v1'],[5,'calendar-paired-v2'],[6,'hierarchy-e5-v1']]){
  await page.goto('/survey.html#roadmap-'+anchor);
  await expect(page.locator('#roadmap-'+anchor)).toBeVisible();
  await page.locator('#roadmap a[href="reports/'+slug+'.html"]').first().click();
  await expect(page.locator('.report-conclusion')).toBeVisible();
  await expect(page.getByRole('link',{name:'Markdown 原文 ↗',exact:true})).toBeVisible();
  await page.getByRole('link',{name:'返回 Survey 证据',exact:true}).click();
  await expect(page.locator('#roadmap-'+anchor)).toBeVisible();
 }
 await page.goto('/reports/hierarchy-sibling-preflight-v1.html');
 await expect(page.locator('.report-lead')).toContainText('仅数据预检');
 await expect(page.locator('.report-limit')).toContainText('没有模型或聚类分数');
});
