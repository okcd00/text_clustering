# 文本聚类方法与开源生态 Survey · v1

**问题导向的阅读与选型地图，而不是跨任务总榜。** 更新日期：2026-11-14。当前版本以仓库既有参考库为起点，连接批量聚类、已有类别路由、层级导航和持续更新；尚不是覆盖所有会议的系统综述。

## 阅读路线

1. [方法地图与证据比较](methods.md)：先确定输入、输出、监督预算和失败模式，再选择论文。
2. [数据集选型与公平比较协议](datasets.md)：先选数据和切分，再谈榜单；真实时间与合成流分开。
3. [复现优先级与覆盖缺口](roadmap.md)：哪些值得先做，哪些暂不应投入。
4. [原始参考库](../references/readme.md)：逐篇来源与 BibTeX；[开源生态快照](../references/ecosystem_2026-11-13.md)：2026-11-13 的 stars、许可证和实现限制。
5. 新增独立可运行 [lexical-v1 基线](../results/lexical-v1/README.md)：两套完整 test、3 seeds、oracle-K/未知 K 和重复排除敏感性，明确不冒充语义模型比较。
6. 本仓库实测证据：[批量协议](../docs/protocols.md)、[场景协议](../docs/scenario-protocols.md)。**下文的论文结果不是本仓库复现结果。**

## 范围与检索边界

- 核心：短文本/意图聚类、冻结语义表示上的聚类、LLM 辅助约束与解释、拒识后发现新类、类别树维护、文本流演化。
- 邻接：主题模型、概念 taxonomy、表格聚类，用于借鉴或单独实验，不把它们的指标直接混入文本聚类榜。
- 文献层：以 2024–2026 近期工作为重点，保留早期基础方法；正式会议、Findings、演示、预印本和软件分别标注。题目或 arXiv 年份不等于录用证明。
- 软件层：优先有可检查代码、明确任务接口和实际采用度的项目。stars 是带日期的可见度快照，不是算法质量，不宣称“全网最高”。
- 来源顺序：正式论文集/论文正文 → 作者官方仓库及配置 → 数据发布页。二手列表只用于发现线索，不作为结果证据。

本轮审阅入口包括 ACL Anthology 的文本/意图聚类、topic modeling、taxonomy、continual intent discovery；ICLR/NeurIPS 原始论文入口；作者 GitHub 及仓库现有引用链。该过程是**目的性筛选**，没有宣称所有数据库已检索穷尽，也不虚构 PRISMA 检索数量。可复用关键词：`text clustering`, `intent discovery`, `LLM clustering constraints`, `topic modeling`, `hierarchical taxonomy`, `stream clustering`。

## 证据分层

| 标记 | 含义 | 允许得出的结论 |
|---|---|---|
| 本仓库测量 | 已发布协议、配置及逐 seed 汇总 | 仅在相同组内比较，遵守历史编码器和数据限制 |
| 论文测量 | 原文表格/实验设置 | 该论文的设置下有效，不等于在本仓库有效 |
| 实现检查 | README、配置或源码静态检查 | 接口/默认值/依赖条件，不等于运行成功 |
| 待验证假设 | 迁移建议或独立改编 | 必须标注未执行，不能写成收益 |

每次扩充先补“方法—数据—协议—成本—证据—限制”，再补引用数量。没有可靠版本/许可/完整实现的条目仍可保留为阅读材料，但不进入可复现候选榜。

## 可视化与维护

[Pages 交互阅读版](https://dian.host/text_clustering/survey.html)。内容以本目录四份 Markdown 为单一来源，运行 `python scripts/build_survey.py` 生成网页；`python scripts/build_survey.py --check` 检查漂移。页面外观在 `scripts/survey_template.html`、`docs/survey.css`，筛选逻辑在 `docs/survey.js`。本页中提及的本地路径为维护说明，不是 Pages 相对文件链接。

## 本地语义对照已执行

[semantic-pilot-v1](../results/semantic-pilot-v1/README.md)：固定 E5-small 与词面表示在相同 512 条样本上对照，分别给出 oracle-K、固定未知 K 和有标签验证集选参的未知-test-K。完整记录环境、权重哈希、耗时与覆盖率，不将小样本结论推广成通用排名。
