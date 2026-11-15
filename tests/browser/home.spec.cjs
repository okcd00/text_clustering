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
  await page.goto('/index.html#compare');await expect(page).toHaveURL(/survey.html(?:\?[^#]+)?#compare$/);
  await page.goto('/index.html#leaderboard');await expect(page).toHaveURL(/legacy.html#leaderboard$/);
});
test('scene first, evidence second, detail third',async({page})=>{
 await page.goto('/index.html');
 await page.locator('.primary-link').click();
 await expect(page).toHaveURL(/index.html#scenes$/);
 await expect(page.locator('.scene-card')).toHaveCount(4);
 await page.reload();await expect(page.locator('#scene-title')).toBeVisible();
 await page.locator('.scene-card > a').first().click();
 await expect(page).toHaveURL(/survey.html(?:\?[^#]+)?#compare$/);
 await expect(page.locator('#compare')).toBeVisible();
 await expect(page.locator('.page-links')).toHaveCount(0);
 await page.getByRole('link',{name:'算法画像',exact:true}).click();
 await expect(page.locator('#profiles')).toBeVisible();
 await page.locator('.concept-map>summary').click();
 await expect(page.locator('.fishbone')).toBeVisible();
});

test('generated icons load and scene methods deep-link to profiles',async({page})=>{
  await page.goto('/index.html');
  const css=await page.locator('.branch-icon').evaluateAll(els=>els.map(e=>getComputedStyle(e).backgroundImage));
  expect(css).toHaveLength(6);
  expect(css.every(src=>src.startsWith('url("data:image/webp;base64,'))).toBeTruthy();
  await page.getByRole('link',{name:'KMeans',exact:true}).click();
  await expect(page).toHaveURL(/survey.html#profiles-kmeans$/);
  await expect(page.locator('#profiles-kmeans')).toBeVisible();
});
