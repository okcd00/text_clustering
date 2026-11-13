const { test, expect } = require('@playwright/test');
const evidence = require('../../docs/scenarios.json');
const destinations = [
  ['新文本', 'onepass', 'scenario-01'],
  ['层级', 'hierarchical', 'scenario-02'],
  ['MiniBatch', 'evolution', 'scenario-06'],
  ['BIRCH', 'evolution', 'scenario-11'],
  ['时间流', 'evolution_temporal_capacity', 'scenario-18'],
];
async function noOverflow(page) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(1);
}
async function scrollTable(page) {
  const table = page.locator('.tablewrap:visible');
  const sizes = await table.evaluate(el => ({ client: el.clientWidth, scroll: el.scrollWidth }));
  if (page.viewportSize().width <= 390) expect(sizes.scroll).toBeGreaterThan(sizes.client);
  if (sizes.scroll > sizes.client) {
    await table.scrollIntoViewIfNeeded();
    await table.focus();
    await expect(table).toBeFocused();
    await page.keyboard.press('ArrowRight');
    await expect.poll(() => table.evaluate(el => el.scrollLeft)).toBeGreaterThan(0);
  }
  await noOverflow(page);
}
test.beforeEach(async ({ page }) => {
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('requestfailed', r => errors.push(r.url()));
  page.on('response', r => { if (r.status() >= 400) errors.push(`${r.status()} ${r.url()}`); });
  page.__dashboardErrors = errors;
});
test.afterEach(async ({ page }) => { expect(page.__dashboardErrors).toEqual([]); });

test('recommendation cards link to their own protocol and actual records', async ({ page }) => {
  for (const [label, task, id] of destinations) {
    await page.goto('/index.html');
    const cards = page.locator('#scenario-choices article');
    await expect(cards).toHaveCount(5);
    const card = cards.nth(destinations.findIndex(d => d[2] === id));
    await expect(card.locator('h3')).toContainText(label);
    await noOverflow(page);
    const link = card.getByRole('link');
    await expect(link).toHaveAttribute('href', `scenarios.html?task=${task}&group=${id}`);
    await link.click();
    await expect(page.locator('#task')).toHaveValue(task);
    await expect(page.locator('#group')).toHaveValue(id);
    const group = evidence.groups.find(g => g.id === id);
    await expect(page.locator('#heading')).toHaveText(`${group.title} · ${group.protocol}`);
    const view = await page.locator('#view').inputValue();
    const records = group.records.filter(r => r.view === view);
    await expect(page.locator('#body tr')).toHaveCount(records.length);
    for (const [i, record] of records.entries()) {
      const row = page.locator('#body tr').nth(i);
      await expect(row.locator('td').first()).toContainText(record.algorithm);
      const key = task === 'onepass' || task === 'hierarchical' ? 'known_accuracy_all' : 'window_weighted_acc_all_noise_error';
      const value = record.summary[key].mean;
      await expect(row.locator('td').nth(1)).toContainText(value.toLocaleString('en-US', { maximumFractionDigits: 4 }));
    }
    await scrollTable(page);
    await page.getByRole('link', { name: '← 选型主页' }).click();
    await expect(page.locator('#scenario-choices article')).toHaveCount(5);
  }
});

test('leaderboard, radar and algorithm dialog fit the viewport', async ({ page }) => {
  await page.goto('/index.html');
  await page.getByRole('button', { name: '效果排行榜', exact: true }).click();
  await expect(page.locator('#rows tr').first()).toBeVisible();
  for (const run of ['known-k-001', 'unknown-k-001']) {
    await page.locator('#run').selectOption(run);
    await expect(page.locator('#rows tr').first()).toBeVisible();
    await scrollTable(page);
    const svg = page.locator('#radar');
    await expect(svg).toBeVisible();
    const bounds = await svg.boundingBox();
    expect(bounds.x).toBeGreaterThanOrEqual(0);
    expect(bounds.x + bounds.width).toBeLessThanOrEqual(page.viewportSize().width + 1);
    // Text must also stay inside the SVG viewBox, not merely its CSS box.
    expect(await svg.locator('text').count()).toBeGreaterThan(0);
    expect(await svg.evaluate(el => [...el.querySelectorAll('text')].every(t => {
      const b = t.getBBox(), v = el.viewBox.baseVal;
      return b.x >= v.x && b.y >= v.y && b.x + b.width <= v.width && b.y + b.height <= v.height;
    }))).toBe(true);
    const buttons = page.locator('#rows button');
    for (let i = 0; i < await buttons.count(); i++) {
      const method = (await buttons.nth(i).getAttribute('aria-label')).replace(' 方法详情', '');
      await buttons.nth(i).click();
      const dialog = page.locator('#info');
      await expect(dialog).toBeVisible();
      await expect(page.locator('#title')).toHaveText(method);
      await expect(page.locator('#description')).not.toBeEmpty();
      const box = await dialog.boundingBox();
      expect(box.x).toBeGreaterThanOrEqual(0);
      expect(box.x + box.width).toBeLessThanOrEqual(page.viewportSize().width + 1);
      expect(box.y + box.height).toBeLessThanOrEqual(page.viewportSize().height + 1);
      await page.getByRole('button', { name: '关闭详情' }).click();
      await expect(dialog).not.toBeVisible();
    }
  }
});
