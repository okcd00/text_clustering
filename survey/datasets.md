# 数据集选型与公平比较协议

[Survey 首页](README.md) · [方法地图](methods.md) · [数据源目录](../datasets/readme.md)

核对日期：2026-11-14。下表区分**官方数据规格**、**仓库实测**与**计划协议**；规模来自官方规格。BANKING77 和 MASSIVE zh-CN 的本地聚合审计与后续实测分别列出。规模是出处中的规格，不冒充本轮 artifact 计数。数据不随仓库分发。原始文本、逐样本预测和缓存不进入公开仓库。

## 1. 可优先进入新基准的数据

| 数据 / 官方来源 | 语言、规模、标签 | 版本与权限 | 适合的公平比较 / 主要陷阱 |
|---|---|---|---|
| [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) | 英文；77 意图；train 10,003 / test 3,080 | 官方 train/test；[许可](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE) CC BY 4.0；运行时 pin commit 与文件 hash | 金融细粒度意图、known-K/unknown-K。无原生 OOS 或真实时间；需从训练集建立独立 validation；隐藏类是人工构造协议 |
| [CLINC150](https://github.com/clinc/oos-eval) | 英文众包单意图；150 意图/10 domains。`data_full.json` 每意图 train100/val20/test30；OOS train100/val100/test1000 | [许可](https://github.com/clinc/oos-eval/blob/master/LICENSE) CC BY **3.0**；不得与 small/imbalanced 版本混用 | intent/domain 两种粒度分别报。现仓库 4,500 test 不含 OOS；原生 OOS 可测拒识，但混合 OOS 标签不是一个可发现语义类。10 hidden intents 的 test 最多 300 条，拒识后缓冲区更小 |
| [WOS v2](https://data.mendeley.com/datasets/9rw3vkcfy4/2) | 英文科学摘要；完整发布规格 46,985 / 134 子类 / 7 父类 | Mendeley v2 页面 CC BY 4.0；描述另含 MIT 式 permission notice，保留两者，不简化成“MIT 数据” | 层级导航、粗/细类聚类。历史 numeric-consistent 子集尚无验证过的公共重建，不能用完整数据伪称复现旧分数。先审计 numeric path 与类名冲突。2017 是发布年，不是样本事件时间 |
| [MASSIVE zh-CN](https://github.com/alexa/massive) / [数据卡](https://huggingface.co/datasets/AmazonScience/massive) | 中文；60 intents / 18 scenarios；卡中 train11,514 / validation2,033 / test2,974 | CC BY 4.0；冻结 release/commit 与 locale。官方 1.1 增加 Catalan，其他语言不变 | 优先补中文语义意图；保留原始划分，用 `utt` 而非带 slot 标记的 `annot_utt`。跨语言按对齐 utterance ID 分组，防翻译泄漏。常规三 split 合计16,521；19,521口径另含竞赛3,000条/locale holdout，不能混入本基准。不是自然中文电商行为 |
| [UCI News Aggregator](https://archive.ics.uci.edu/dataset/359/news+aggregator) | 英文新闻标题；422,937；4 大类及 STORY 聚合标识；官方介绍写到 2014-08-10，当前官方压缩包实测 UTC 日期为 2014-03-10 至 08-28 | CC BY 4.0；TIMESTAMP 是粗略发布时间 | 可做时间戳 silver-label 回放，不是人工金标意图漂移；story 标签与 4 大类是两任务。转载/重复 URL/标题及时间边界重复故事需单列；2pageSessions 不等于完整真实用户行为流 |

**BANKING77 与 MASSIVE zh-CN 已实测。** 两者都有[全量官方 test 词面基线](../results/lexical-v1/README.md)、[同样本词面与 E5 小对照](../results/semantic-pilot-v1/README.md)、[统一静态评估](../results/unified-static-v1/README.md)和[开放集路由](../results/routing-v2/README.md)。全量词面、512 条静态语义与带标签支持集的路由是不同协议，不能直接混排；这两套数据均未用于真实时间流评估。CLINC 和 WOS 的历史结果不动。

**UCI 时间回放完成数据校验，尚未跑算法。** [预检与复跑命令](../experiments/calendar-preflight-v1/README.md)固定了官方包的 SHA-256、UTC 热身和七个评分日期，审计了重复标题与跨窗同故事。原始新闻标题和 URL 不入库。

## 2. 暂不承诺开箱可复现的覆盖候选

- **中文新闻主题**：[C-MTEB](https://github.com/FlagOpen/FlagEmbedding/tree/master/C_MTEB)、[ThuNewsClusteringP2P.v2](https://huggingface.co/datasets/mteb/ThuNewsClusteringP2P.v2)。它们补中文 topic，不替代中文 intent；原版嵌套任务行数不能当文档数，v2 不是同一协议。数据卡许可 unknown 时，不能套用框架许可证。先核对实际数据权利与 task revision，再纳入。
- **人工标注新闻流**：[Miranda et al., EMNLP 2018](https://aclanthology.org/D18-1483/)、[官方代码](https://github.com/Priberam/news-clustering)。旧 FTP 数据入口当前可用性未验证；官方说明原始部分组件专有，公开的是复刻/预提取特征等。BSD-3 软件许可不等于新闻文本许可。列为待准入，不宣称可完整复现。
- **COSID 分阶段意图**：[论文及官方实现](../references/chen2026cosid/readme.md)。这是在 BANKING/CLINC 上构造的协议，不是新自然流数据集；随机、语义调度、旧类复现率/Zipf 不平衡分别记录。借 split 的低成本实验与完整 CIDA 复现分开。

- **EB-NeRD 新闻曝光流（仅作协议参考）**：[官方字段](https://recsys.eb.dk/dataset/)给出 impression 时间、会话、记录为可见的文章 ID 与点击 ID；可见列表已打乱，不能恢复排位；公开说明中的 test 不含点击 ID，有标签的 train/dev 才适合上述回放。它来自筛选过的活跃读者，不是商品意图金标，也不能用点击缺失推断用户已无兴趣。仅在非商业推荐研究、符合[原许可](https://recsys.eb.dk/assets/pdf/general_license_terms.pdf)并走官方申请后考虑；许可还限制用途、存放与使用期限。本仓库未申请、下载或运行，也不将它纳入默认一键评估。
  - 可证伪的问题：以预先固定的主题分组和观察窗，区分「无可见访问信号」「观察到访问但无同主题曝光」「有同主题曝光但未点击」。下一窗口仅在确有同主题曝光的机会中，比对时间衰减和基于机会的衰减对点击的预测，配平上次互动间隔、既往活跃度和曝光次数。比较是观察性的，不是去偏的策略因果估计。特征在当前截点冻结；排除未来阅读、滚动及含未来的文章统计，文章快照可能有后续编辑，不能验证时点版本时不把正文当作时点可见输入。默认文本聚类时间回放仍先选 UCI。

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

## 5. 仍待补齐的协议

BANKING77 与 MASSIVE zh-CN 已分别给出英文细粒度与中文意图小样本结果。跨域的 CLINC150 可补相同 frozen encoder 下的静态聚类对照，并将原生 OOS 与 held-out intents 分设两个拒识组。层级先解决 WOS 重建和标签一致性，再跑；流式先借 COSID split 测旧新类干扰，再进入 UCI 时间回放。

后续目标是验证新增数据与协议可重复、找到稳定差异，不追求一次跑完所有论文，更不以英文分类标签上的 ARI 推断真实电商推荐增益。

## 6. 数据准备实证：先把比较的地基做实

[可执行准备流程与完整审计口径](../datasets/preparation.md)。数据只下载到仓库外；没有模型调用，也没有向外部服务发送文本。旧榜单不变。

| 数据 | 已固定并实测 | 发现的比较风险 | 下一步门槛 |
|---|---|---|---|
| BANKING77 | 既有 revision 与三文件 SHA-256 全通过；train/test 10,003/3,080，均77类 | 跨split原文重合0，但规范化文本有7个共享字符串；无原生validation | 固定训练内validation与去重敏感性方案；不能只查精确重复 |
| MASSIVE zh-CN | 固定官方账号 HF Parquet 转换 revision；11,514/2,033/2,974；60/59/59 intents，18 scenarios | train/dev、train/test、dev/test 分别132/187/46个共享原文；ID无交叉不代表文本无交叉 | 官方split与重复控制敏感性轨道分开；test实际59类不能机械视为60类 |

MASSIVE validation 缺 `audio_volume_other`，test 缺 `cooking_query`。已验证60意图到18场景的单父映射；这是数据审计，不授予模型读取隐藏类别或测试标签的权限。完整 hash、公共标签计数和缺失类在 [BANKING 审计](../datasets/banking77/audit_2026-11-14.json)、[MASSIVE 审计](../datasets/massive_zh_cn/audit_2026-11-14.json)。共享数按不同字符串计，不是重合行对数；尚未测量它对模型分数的影响。

关于 MASSIVE 总量：[MMNLU-22 组织者报告 §4.1/4.2.1](https://arxiv.org/html/2212.06346)说明常规划分之外，另有每语言3,000条人工改写的竞赛留出集。因此19,521与这里三split实数16,521属于不同口径；本次仅审计固定HF转换版本的常规三split，未下载竞赛holdout，也未验证原始tarball字节。


## 首个可运行基线：lexical-v1

**本仓库新实测，独立于历史结果。** 在 BANKING77 / MASSIVE zh-CN 全量官方 test 上，冻结字符 TF-IDF → 64 维 SVD → L2 表示，以 3 个随机种子比较 oracle-K KMeans、固定参数的未知 K HDBSCAN，以及全归一簇的 sanity control。无需模型、GPU 或外部 API；是词面基线，不是语义方法排名。

[完整执行协议与逐 seed 结果](../results/lexical-v1/README.md)。已完成独立的重复排除敏感性实验：BANKING77 从 3,080 到 3,071 条；MASSIVE zh-CN 从 2,974 到 2,696 条，均未丢失 test 中已有标签。删除所有 test 重复组及 train/dev 同文交集，不挑选“正确标签”的代表。每个新语料重新拟合表示，因此不能把分数变化解释为泄漏的因果效应。

| 官方 test 实测 | Oracle-K KMeans 的 ARI（3 seed 均值） | 固定 HDBSCAN 的 ARI / 覆盖率 | 允许的结论 |
|---|---|---|---|
| BANKING77，77 个 test-present 意图 | 0.2934 | 0.0155 / 42.0% | 此词面表示和固定参数下，大量样本被密度法拒收；不能据此否定 HDBSCAN 或未知 K 方法 |
| MASSIVE zh-CN，59 个 test-present 意图 | 0.1790 | 0.0387 / 62.9% | 相同配置不等于相同任务难度；不能把两个数据集的绝对分数当作语言优劣 |

这里的全量 ARI 明确将噪声视为同一簇；原始 JSON 同时报告 covered-only ARI、覆盖率、簇数与 NMI，避免只展示易样本分数。KMeans 获得 test-present oracle K，是不同信息预算，不宣布“公平击败”未知 K。全归一簇控制覆盖率 100% 而 ARI/NMI 为 0，说明覆盖率也不能单独排名。3 seeds 是稳定性检查，不是置信区间。

下一步先固定多语言语义模型的 revision、prefix、池化和归一化，再在同一矩阵上比较；另设验证集选参，不能反复按这里的 test 分数调参。此轮没有验证路由、层级、流式或业务推荐增量。
