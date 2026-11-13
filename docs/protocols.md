# 公开评测口径

## 首版范围

这是历史实验的显式字段白名单导出，不是全部内部实验的镜像。公开 `results.json` 保留17组配置的汇总及85次逐种子数字，固定随机种子0–4。未包含的路由、层级和持续更新结果标为“未公开”，不代表算法没有实现，也不代表已经全面评测。

- **known-k-001**：KMeans / MiniBatchKMeans；BANKING77 official test intent、CLINC150 in-scope test intent/domain、WOS numeric-consistent all numeric paths。K来自各视图标签基数，所以是已知K实验，不证明能发现类别数。训练不输入类别标签。
- **unknown-k-001**：HDBSCAN / BERTopic核心管线 / EVoC；BANKING77和CLINC150上述test视图，固定预设，不用测试标签挑层或调参。同视图使用相同已缓存向量；CLINC intent/domain复用向量。
- 不跨数据集、标签粒度或协议排名。WOS的全部样本分区实验不等同于held-out分类泛化。

## 指标

- ACC assigned：只在非噪声样本上做 Hungarian 一对一最优标签对齐；不能独立于覆盖率解读。
- ACC all, noise error：噪声拒收视为错，分母为全部样本。
- Coverage：非噪声样本 / 全部样本。
- NMI assigned：非噪声样本上，互信息除以真实与预测标签熵的算术平均。
- ARI assigned：非噪声样本上，经随机一致性修正的 Rand index。
- 耗时：fit+predict墙钟时间，不含编码；RSS为进程峰值，不能等同于算法增量内存。汇总为均值和标准差，逐seed值保存在JSON。
- 缺失值显示“未提供”，绝不补0。已知K结果未记录覆盖率字段时不推断填值。

## 数值复现与费用

历史实验使用自部署1024维语义编码器。公开版本不包含其业务标识、访问地址、权重、原始文本、向量或逐文本预测；checkpoint revision未知，因此当前不能从本仓库精确重现历史数值。向量摘要只用于同输入身份检查。费用未知，不写成免费或零成本。公共工具可用自己的向量运行；字符哈希示例仅验证调用链，不进入语义效果榜。

这些结果只支持对应固定协议下的局部选择，不作为普遍SOTA声明。最新内部研究结论不自动成为公开结论；后续逐批审核新证据再更新。
