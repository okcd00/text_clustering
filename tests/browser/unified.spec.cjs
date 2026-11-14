const {test,expect}=require('@playwright/test');
const fixture={status:'executed',cohorts:[{id:'test',dataset:'Fixture only',representation:'fixed',scenario:'known-k',n:10,dimensions:4,information_budget:'oracle K'}],methods:[{id:'m',title:'Method A'}],runs:[0,1,2].map(seed=>({cohort_id:'test',method:'m',seed,status:'ok',metrics:{ari_all_noise_as_one_cluster:.2+seed*.1,coverage:1},fit_predict_seconds:seed+1,process_peak_rss_bytes:1048576})),deferred:[]};
test('default, archive hash, fishbone and honest unified metric controls',async({page})=>{
 await page.route('**/unified-results.json',r=>r.fulfill({json:fixture}));
 await page.goto('/index.html');await expect(page.locator('#compare')).toBeVisible();
 await expect(page.locator('#comparison-rows')).toContainText('0.300');
 await page.locator('#metric').selectOption('fit_predict_seconds');await expect(page.locator('#comparison-rows')).toContainText('2.000');
 await page.locator('.bone summary').first().click();await expect(page.locator('.bone').first()).toHaveAttribute('open','');
 await page.getByRole('link',{name:'算法画像',exact:true}).click();await expect(page.locator('#profiles')).toBeVisible();
 expect(await page.evaluate(()=>document.documentElement.scrollWidth-innerWidth)).toBeLessThanOrEqual(1);
 await page.goto('/index.html#leaderboard');await expect(page).toHaveURL(/legacy.html#leaderboard/);await expect(page.locator('#leaderboard')).toBeVisible();
});
test('empty result never substitutes historical numbers',async({page})=>{
 await page.route('**/unified-results.json',r=>r.fulfill({json:{status:'not_run',cohorts:[],runs:[]}}));
 await page.goto('/survey.html#compare');await expect(page.locator('#compare-status')).toContainText('尚未完成');await expect(page.locator('#comparison-rows tr')).toHaveCount(0);
});
