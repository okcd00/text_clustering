const {test,expect}=require('@playwright/test');
test('home pathway links and mobile-safe illustration',async({page})=>{
  await page.goto('/index.html');
  await expect(page.getByRole('heading',{name:'同一条主干，不同的分叉。'})).toBeVisible();
  await expect(page.locator('.path-branch')).toHaveCount(6);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBeTruthy();
  await page.locator('.pathway-info summary').click();
  await expect(page.locator('.pathway-info p')).toBeVisible();
  await page.locator('.pathway-info summary').click();
  await page.locator('.branch-4').click();
  await expect(page).toHaveURL(/survey.html#roadmap-6$/);
  await expect(page.locator('#roadmap')).toBeVisible();
  await page.goBack();
  await expect(page.locator('.path-branch')).toHaveCount(6);
});
test('legacy index hashes still resolve',async({page})=>{
  await page.goto('/index.html#compare');await expect(page).toHaveURL(/survey.html#compare$/);
  await page.goto('/index.html#leaderboard');await expect(page).toHaveURL(/legacy.html#leaderboard$/);
});
