# 看板浏览器回归

使用真实 Chromium，在 320、390、1280 CSS 像素宽度下检查公开看板。无需数据集、embedding 或外部服务。

```sh
npm ci
npx playwright install chromium
npm run test:dashboard
```

需要 Node.js 20+、Python 3（用于静态 HTTP 服务）。Linux 首次运行可使用 `npx playwright install --with-deps chromium` 安装浏览器系统依赖。测试独占本机 8897 端口；端口已占用时失败，不复用可能过时的服务。

若已经安装 Google Chrome，可跳过浏览器下载，用 `DASHBOARD_BROWSER_CHANNEL=chrome npm run test:dashboard` 启动独立的无头测试浏览器（不使用个人浏览器资料）。默认使用 Playwright 配套 Chromium。

## 自动检查什么

- 五张场景推荐卡片真实点击后的 task/group、标题、算法行与实际 JSON 指标一致；返回主页有效。
- 手机与桌面不发生页面级横向溢出；手机指标表保留局部滚动，键盘右键能够滚动。
- 已知 K / 未知 K 排行榜、雷达图边界及 SVG 文字边界正常。
- 各方法详情弹窗可打开、内容非空、边界在视口内，并可关闭。
- 页面脚本异常、失败请求及 HTTP 错误使测试失败。

`playwright.config.cjs` 管理独立服务、视口与失败产物。失败截图和 trace 在忽略提交的 `test-results/`；使用 `npx playwright show-trace <trace.zip>` 排查。数值检查读取当前公开证据，链接归属使用独立固定映射；改变推荐组时需一起审核测试映射。没有修改实验数据，也不通过截图基线自动接受布局变动。

这套检查验证 Chromium 的几何布局与交互，不替代 Safari/真机触摸体验或人工审美检查。CI 接入示例见 `docs/ci-example.yml`，当前尚未安装 GitHub Actions 工作流。
