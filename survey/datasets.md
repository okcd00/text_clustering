# 数据集选型与公平比较协议

[Survey 首页](README.md) · [方法地图](methods.md) · [数据源目录](../datasets/readme.md)

核对日期：2026-11-14。下表区分**官方数据规格**、**仓库历史使用子集**与**计划协议**；未重新下载/运行数据。规模是出处中的规格，不冒充本轮 artifact 计数。数据不随仓库分发。原始文本、逐样本预测和缓存不进入公开仓库。

## 1. 可优先进入新基准的数据

| 数据 / 官方来源 | 语言、规模、标签 | 版本与权限 | 适合的公平比较 / 主要陷阱 |
|---|---|---|---|
| [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) | 英文；77 意图；train 10,003 / test 3,080 | 官方 train/test；[许可](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE) CC BY 4.0；运行时 pin commit 与文件 hash | 金融细粒度意图、known-K/unknown-K。无原生 OOS 或真实时间；需从训练集建立独立 validation；隐藏类是人工构造协议 |
| [CLINC150](https://github.com/clinc/oos-eval) | 英文众包单意图；150 意图/10 domains。`data_full.json` 每意图 train100/val20/test30；OOS train100/val100/test1000 | [许可](https://github.com/clinc/oos-eval/blob/master/LICENSE) CC BY **3.0**；不得与 small/imbalanced 版本混用 | intent/domain 两种粒度分别报。现仓库 4,500 test 不含 OOS；原生 OOS 可测拒识，但混合 OOS 标签不是一个可发现语义类。10 hidden intents 的 test 最多 300 条，拒识后缓冲区更小 |
| [WOS v2](https://data.mendeley.com/datasets/9rw3vkcfy4/2) | 英文科学摘要；完整发布规格 46,985 / 134 子类 / 7 父类 | Mendeley v2 页面 CC BY 4.0；描述另含 MIT 式 permission notice，保留两者，不简化成“MIT 数据” | 层级导航、粗/细类聚类。历史 numeric-consistent 子集尚无验证过的公共重建，不能用完整数据伪称复现旧分数。先审计 numeric path 与类名冲突。2017 是发布年，不是样本事件时间 |
| [MASSIVE zh-CN](https://github.com/alexa/massive) / [数据卡](https://huggingface.co/datasets/AmazonScience/massive) | 中文；60 intents / 18 scenarios；卡中 train11,514 / validation2,033 / test2,974 | CC BY 4.0；冻结 release/commit 与 locale。官方 1.1 增加 Catalan，其他语言不变 | 优先补中文语义意图；保留原始划分，用 `utt` 而非带 slot 标记的 `annot_utt`。跨语言按对齐 utterance ID 分组，防翻译泄漏。数据卡总量文字与 split 合计不一致，必须实数核对。不是自然中文电商行为 |
| [UCI News Aggregator](https://archive.ics.uci.edu/dataset/359/news+aggregator) | 英文新闻标题；422,937；4 大类及 STORY 聚合标识；2014-03-10 至 08-10 | CC BY 4.0；冻结下载版本/hash；TIMESTAMP 是 approximate publication time | 可做时间戳 silver-label 回放，不是人工金标意图漂移；story 标签与 4 大类是两任务。转载/重复 URL/标题及时间边界重复故事需单列；2pageSessions 不等于完整真实用户行为流 |

MASSIVE 是新增候选，不是已完成的适配/测试。BANKING、CLINC、WOS 原有出处文件位于 `datasets/`，历史结果保持不动。

## 2. 暂不承诺开箱可复现的覆盖候选

- **中文新闻主题**：[C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB)、[ThuNewsClusteringP2P.v2](https://huggingface.co/datasets/mteb/ThuNewsClusteringP2P.v2)。它们补中文 topic，不替代中文 intent；原版嵌套任务行数不能当文档数，v2 不是同一协议。数据卡许可 unknown 时，不能套用框架许可证。先核对实际数据权利与 task revision，再纳入。
- **人工标注新闻流**：[Miranda et al., EMNLP 2018](https://aclanthology.org/D18-1483/)、[官方代码](https://github.com/Priberam/news-clustering)。旧 FTP 数据入口当前可用性未验证；官方说明原始部分组件专有，公开的是复刻/预提取特征等。BSD-3 软件许可不等于新闻文本许可。列为待准入，不宣称可完整复现。
- **COSID 分阶段意图**：[论文及官方实现](../references/chen2026cosid/readme.md)。这是在 BANKING/CLINC 上构造的协议，不是新自然流数据集；随机、语义调度、旧类复现率/Zipf 不平衡分别记录。借 split 的低成本实验与完整 CIDA 复现分开。

## 3. 固定六条实验轨道，禁止总榜

| Track | 输入/切分 | 核心指标 | 不能得出的结论 |
|---|---|---|---|
| B-K：批量已知 K | 同一文档集合，同一明确 K | 全量 ACC、ARI/NMI、时间/内存、多 seed | oracle K 优势不能冒充未知 K 能力 |
| B-U：批量未知 K | 禁止 test label 选 K/阈值 | discovered K、coverage、全量噪声计错 ACC、接收子集 ARI/NMI | 高接收子集分数不等于全量好；密度 noise 不等于校准拒识 |
| R：已有类路由/拒识 | train known support，独立 validation/test hidden classes | known acc/coverage、hidden false accept、novel P/R/F1、校准曲线 | CLINC 原生 OOS 与 held-out intent 不是同一未知分布 |
| H：层级 | 固定树/父子标签，叶类隐藏协议 | leaf/parent、novel-and-parent recall、edge/ancestor F1、拒识 | 导航不等于从零构树；给 gold edges 的监督模型不是无监督 |
| S：合成阶段流 | 固定引入顺序、复现率、类不平衡；阶段内先预测后更新 | old/new accuracy、OSR/USR、buffer、簇稳定性、成本 | 随机排列不是自然漂移；阶段窗口不是逐事件在线 |
| T：时间戳回放 | 稳定时间排序、未来隔离、重复故事显式处理 | 每窗口质量、recurring/new story、时间/内存/延迟、维护成本 | approximate timestamps 与 silver labels 不证明真实行为意图漂移 |

## 4. 每次比较必须冻结的 manifest

新实验必须记录以下字段；无法公开具体数据时，仅发布安全汇总与摘要，但不能声称完全可复现：

```yaml
# Schema example, not a completed run
source: canonical dataset URL
revision: release or immutable commit
artifact_sha256: required before execution
license: dataset-specific license and notices
language: locale
text_field: exact field and preprocessing version
label_view: intent / domain / leaf_path / story
counts: raw, deduplicated, train, validation, test, known, hidden
split: official or derived; seed; held-out classes; grouping key
time: none / synthetic_stage / approximate_timestamp / verified_event_time
encoder: model, immutable revision, instruction, pooling, truncation, dtype
matrix: dimensions, preprocessing_fit_split, normalization, hash
protocol: track, K source, noise/reassignment policy, update timing
budget: tuning trials, labeled support, LLM tokens/calls, hardware
```

1. **先锁分割再编码**：按 document/utterance/story ID 去泄漏；若研究重复出现，明确它是目标而非意外泄漏。PCA/词表/归一化参数只在允许的训练/calibration 集拟合。
2. **双层对照**：同矩阵后端比较回答“聚类器是否更好”；完整 native pipeline 比较回答“系统是否更好”。两个问题分栏，不能混因果。
3. **相同监督与调参预算**：oracle K、已知类名、标注 pairs、gold hierarchy、LLM 外部知识均计入预算。跨语言 encoder/指令差异单列。
4. **同一噪声口径**：同时报告接收和全量。Top2Vec 的全量再分配需配“相同密度聚类+最近质心”控制，不能把 assignment policy 当成表示突破。
5. **同一统计单位**：种子配对；报告均值/SD、逐 seed 与样本数。共享向量的 CLINC domain/intent 不是独立训练；窗口不是独立数据集。仅单次结果不附伪造方差/显著性。
6. **同一成本边界**：区分 encoder、fit、校准、预测、LLM 生成、维护、模型加载；报告峰值内存、吞吐/延迟及硬件。无法观测的账单写未知，不写零。
7. **留出盲测**：旧 test 已被反复查看，只能做描述性配对；新改进须留新验证/测试或预登记配置。已有历史 checkpoint 不可获取时，建立新 pinned encoder 基线，不回填旧分数。

## 5. 建议第一轮（未执行）

先用 BANKING77、CLINC150、MASSIVE zh-CN 建立英文细粒度/跨域/中文意图三格；每格相同 frozen encoder 下做 KMeans 与一种 unknown-K 基线，已知 K 和未知 K 独立。CLINC 同时设原生 OOS 与 held-out intents 两个拒识组。层级先解决 WOS 重建和标签一致性，再跑；流式先借 COSID split 测旧新类干扰，再进入 UCI 时间回放。

这轮目标是验证数据与协议可重复、找到稳定差异，不追求一次跑完所有论文，更不以英文分类标签上的 ARI 推断真实电商推荐增益。
