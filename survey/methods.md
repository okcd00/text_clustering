# 方法地图：从任务而不是模型名称出发

[Survey 首页](README.md) · [数据协议](datasets.md) · [优先级](roadmap.md)

## 1. 按系统环节分组

| 环节 | 最小基线 | 值得对照的路线 | 必须隔离的变量 |
|---|---|---|---|
| 静态、已知 K | KMeans / MiniBatchKMeans | 对比表示学习、LLM 约束、[Bag-of-Texts](../references/lin2026bagoftexts/readme.md)、[Summaries as Centroids](../references/diazrodriguez2026summariescentroids/readme.md) | K 来源、训练标签、表示/生成预算、测试时可见信息 |
| 静态、已知 K 的规模化 | Lloyd / MiniBatchKMeans | [Faiss e-means](../references/bessa2026emeans/readme.md)（2026 预印本，未实测） | 同硬件和向量测耗时、峰值内存与失真；失真与语义 ARI 分开，计入最终分配时间 |
| 静态、未知 K | HDBSCAN / BERTopic 核心管线 | [Top2Vec](../references/angelov2020top2vec/readme.md)、[TopicMerge](../references/raffel2026topicmerge/readme.md) | 噪声、全覆盖再分配、初始微簇数、合并决策成本 |
| 主题解释 | c-TF-IDF 关键词 | [CTM](../references/bianchi2021ctm/readme.md)、[FASTopic](../references/wu2024fastopic/readme.md)、[LLM-ITL](../references/yang2025llmitl/readme.md) | soft topics ≠ hard clusters；coherence ≠ ARI；K 与词表需一致 |
| 新文本路由/拒识 | 原型 / 多原型 / kNN | 选择性语义验证、拒识缓冲区再聚类；[NILC](../references/wang2025nilc/readme.md) 为阅读对照 | known coverage 与 hidden false accept 同报；校准只能用 validation |
| 类别树维护 | 现有分层导航与 NodeProposal | [SPARROW](../references/zhang2026sparrow/readme.md)、[PEEL](../references/kim2026peel/readme.md)、[TaxoAgent](../references/lin2026taxoagent/readme.md) | 概念词表/金标准边是额外信息；树与 DAG、归属与构树分开 |
| 持续演化 | frozen vs MiniBatch/BIRCH 更新 | [COSID](../references/chen2026cosid/readme.md) 阶段协议、[RAILS](../references/oliya2026rails/readme.md) 增量路线 | 阶段流≠真实时间；预测后学习；遗忘、簇 ID、内存增长须单测 |
| 邻接表格方法 | 同矩阵几何基线 | [TabClustPFN](../references/zhao2026tabclustpfn/comparison.md) | 文本适用性未证实；维度、K、批次归一化与非商业许可限制 |

以上是路线定位，非各方法已实现清单。详读级证据如下；仅有书目记录的既有条目不自动提升为已复现。

## 2. 近期重点工作的证据卡

| 工作 / 出版状态 | 可借鉴之处与原始证据 | 限制 / 下一步 |
|---|---|---|
| [TopicMerge](https://aclanthology.org/2026.findings-emnlp.2248/)，EMNLP Findings 2026 | 过分割微簇后做局部 LLM 合并；Table 1 Bills ARI .42、Wiki .71；Table 2 为 645/562 次 prompts | 基线调用数是估计下界，不是实测延迟节省；只有两个数据集；记录时作者代码不可访问。先几何合并 vs 语义合并配对 |
| [Dial-In LLM](https://aclanthology.org/2025.emnlp-main.300/)，EMNLP main 2025 | 对话角色/上下文与迭代意图发现；Table 4 NMI 从 KMeans .7899 到 context-free .8208，再到 role merge .8826 | 不把角色输入的收益全归为聚类器；完整训练集和微调权重可用性未建立。对话不是点击行为流 |
| [SPARROW](https://aclanthology.org/2026.emnlp-main.1502/)，EMNLP main 2026 | 完整概念集合上分块建树/融合；1K GPT-5 设置下 Product ancestor F1 .745 vs LLMscorer .453 | 同设置 Edge F1 .579 vs .588，不能说全面提升。已有概念名、单父树，不是开放世界文本发现 |
| [PEEL](https://aclanthology.org/2026.emnlp-main.1305/)，EMNLP main 2026 | 冻结 embedding 上的轻量 containment probe；有助研究父子关系信号 | 使用来源域金标准边与给定概念词表，不是无监督；98K 参数不代表整体系统零成本 |
| [LLM-ITL](https://aclanthology.org/2025.acl-long.70/)，ACL main 2025 | LLM 改善 topic words，置信加权 OT 训练 | 保留为主题解释独立轨道；训练、生成、外部知识偏置均需核算 |
| [COSID / CIDA](https://aclanthology.org/2026.emnlp-main.705/)，EMNLP main 2026 | ClincCL α=.5，CIDA USR 6.22 vs VBCGCD 21.82，但 OSR .73 vs .47；明确新类漏检/旧类误拒的权衡 | 默认配置 `ESTIMATE_K=False` 使用真实新类数。先借协议，不直接引入逐样本 LLM；完整定义见[参考卡](../references/chen2026cosid/readme.md) |

各论文数据/表示/监督不一致，上表数字不能纵向排序。新旧类准确率、层级质量、聚类质量与业务推荐增量是不同终点。

## 3. 开源项目怎么比较

[带 stars 与许可的原始快照](../references/ecosystem_2026-11-13.md)保留原日期，不以新文档日期冒充重新查询。

- **BERTopic / Top2Vec**：优先真实工程基线。仓库的 `bertopic_pipeline` 只是降维聚类核心，不等于完整 BERTopic。Top2Vec classic 将所有文档重新归到最近主题，不能把它的覆盖率直接与保留 noise 的 HDBSCAN 比；加一个“相同 HDBSCAN + 最近质心再分配”控制组。
- **CTM / FASTopic**：给定 K 的训练型 soft-topic 路线。若用 argmax 转硬标签，报告转换和覆盖规则；另报告主题质量。训练时间与冻结 encoder 聚类成本分开。
- **TopMost**：组织主题模型实验的工具箱，而非一种单独算法。先用作参考适配，不替换仓库既有协议。
- **Contextual-Top2Vec**：同 Top2Vec 仓库，不能重复计 stars；上下文版本为 beta 且 encoder 支持受限。
- **论文附属代码**：区分 README 存在、完整代码/权重可获取、许可明确、成功运行、完整复现五个等级。缺失许可不是默认许可。

## 4. 当前本地证据对选型的约束

既有 WOS 单次配对里 top-1 NLI 把 hidden false accept 从 64.00% 提到 69.85%，known accuracy 仅 58.62%→58.96%；不能默认“加一次 LLM/NLI 就更准”。见[完整口径](../docs/scenario-protocols.md)。先找模型成本可控且能改善失败 case 的方案，再扩大离线 seed/数据范围；只有明确离线增量，才讨论下游小流量验证。
