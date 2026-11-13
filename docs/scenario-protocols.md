# 路由、层级和持续更新：公开证据

本次仅从既有结果投影脱敏字段，没有重新运行实验或调用模型。18 个独立比较组、51 个方法/视图配置、185 条逐 seed 记录；另有 10 个单次日历流方法/视图配置，**不能算成 10 次独立实验**。它们与原有 17 个批量配置、85 条逐 seed 记录分开存储。

入口：[场景看板](https://dian.host/text_clustering/scenarios.html)。机器可读证据为 `scenarios.json`，每组保留固定配置、输入摘要、样本数、均值/SD、逐 seed 数值及源快照 digest。原始服务信息、环境路径、逐文本输出和内部 run ID 不发布。

## 协议分组

| 组 | 比较 | 解释要点 |
|---|---|---|
| 01 / 02 | WOS 单原型、三原型；平面路由 / 层级导航 | 81 已知叶类、20 验证隐藏类、20 测试隐藏类；17,115 训练支持；6,053 测试 query（2,895 known + 3,158 hidden）。阈值仅用 validation、known coverage ≥0.8。不是无监督或零样本分类。 |
| 03 / 04 | 单原型与 top-1 NLI 验证 | 同协议 seed=0 配对，只有一次测量；不可借用五 seed 方差。NLI 使用可见类别描述，不能将差异全部归因于蕴含机制。 |
| 05 | DBSTREAM 冻结 / 更新 | BANKING77、CLINC intent/domain；随机顺序窗口，不是真实时间漂移。 |
| 06–09 | MiniBatch 冻结 / 更新 | BANKING77、CLINC intent/domain、WOS；显式已知 K，不能和 unknown-K 当作同预算实验。 |
| 10 | TextClust 有效微簇快照 | token unigram/bigram TF，不是同一 dense embedding 表示。保留独立组，CLINC 两视图共享学习成本。 |
| 11–14 | BIRCH 冻结 / 更新 | unknown-K CF leaf；BANKING77、CLINC 两视图和 WOS。仅同组配对，不因窗口相同就认定跨方法准入成立。 |
| 15 | WOS DBSTREAM 冻结 / 更新 | 独立准入与质量记录，不与不同 K/表示预算混榜。 |
| 16 | 单原型、三原型、ELiDECiDe 冻结向量核心改编 | 已查看 test 上的探索性分析；known validation 校准；不是新的盲测，不是论文完整复现。 |
| 17 | 新闻日历流 BIRCH 冻结 / 更新 | 单次按近似发布时间回放；story 标签是 silver labels，不是已证明真实漂移。raw / first_title 共用学习和成本。 |
| 18 | BIRCH 无界、容量冻结、滚动重建 | 固定工程预算，无 test 搜参；已查看 test 的描述性复用。max_cf=2048 / window_size=2048、rebuild_every=2048。 |

## 指标与成本

- Known accuracy 的分母是全部 known query（拒识算错），known coverage 是 known 中接收的比例。Hidden false accept 的分母是全部 hidden query，越低越好；必须同时看 known accuracy/coverage，防止全部拒收。
- Novel precision/recall/F1 评价拒识；novel-and-parent recall 需要同时拒识且父级正确。**没有测量新增类的聚合、命名及实际插入**。
- 持续更新采用窗口内独立标签对齐，按样本数加权；不代表簇 ID 跨窗口稳定。随机顺序种子不是独立真实时间序列。
- ACC 全量把噪声计错；ARI/NMI 是接收子集指标，结合 coverage 阅读。窗口输出簇数不是内存状态微簇数。
- 表格时间维持原字段：fit+test 不包括验证、校准和模型加载；stream 作业时间包含输入、学习、持久化等。没有把不同字段改成统一“推理延迟”。全部明细保留源值。
- 日历容量组的 native、fingerprint、snapshot、persistence、maintenance 等成本单列；maintenance 与 native fit 重叠，不重复累加。RSS 是进程高水位，不是模型独占内存或设备显存。
- 原型重复计时不等于独立质量重复；单次 SD 为 null；缺测保留 null。不同视图共用学习时不相加计费。
- 历史 embedding checkpoint、原始账单未知，未分发向量。NLI 公共 checkpoint 和 revision 见各组 settings；本地计算费用未知，不把零新增 API 费写成零总成本。

## 当前结果提示

WOS 平面路由的配对 NLI 实验中，隐藏类误接收率从 **64.00% 升到 69.85%**；已知准确率从 58.62% 到 58.96%。这项配置不能推荐为“更好的隐藏类拒识器”。层级配对中误接收率从 70.52% 到 71.15%，已知准确率从 53.13% 到 51.81%。这些是单次配对结果，不作统计显著性或一般 NLI 优劣结论。

完整组内限制随数据展示。当前不补发历史已失败尝试的所有运行日志，也不把旧不完整 TextClust 快照提升为最新结果。公开证据便于选型，不承诺原始数据与完整实验环境已经可一键重建。
