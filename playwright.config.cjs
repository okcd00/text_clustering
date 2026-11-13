const { defineConfig } = require('@playwright/test');
module.exports = defineConfig({
  testDir: './tests/browser',
  fullyParallel: true,
  workers: 2,
  retries: 0,
  use: { baseURL: 'http://127.0.0.1:8897', trace: 'retain-on-failure', screenshot: 'only-on-failure' },
  projects: [320, 390, 1280].map(width => ({ name: `chromium-${width}`, use: { browserName: 'chromium', ...(process.env.DASHBOARD_BROWSER_CHANNEL ? { channel: process.env.DASHBOARD_BROWSER_CHANNEL } : {}), viewport: { width, height: 900 } } })),
  webServer: { command: 'python3 -m http.server 8897 --bind 127.0.0.1 --directory docs', url: 'http://127.0.0.1:8897/index.html', reuseExistingServer: false },
});
