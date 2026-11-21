# ESC v1.0 跨日故事延续 · 固定协议

[实测结果](../../results/esc-crossday-v1/README.md) · [数据选型](../../survey/datasets.md)

问题：在同一套词面特征和分配规则下，收盘更新已存在的原型，会提升跨日同主题链接，还是把不同主题并在一起？这是小规模 Track T 诊断，与 UCI 日历流独立。

- 数据：[格罗宁根大学数据页](https://research.rug.nl/en/datasets/the-event-storyline-corpus-v10/)；[作者 ESC v1.0 与内置 ECB+](https://github.com/tommasoc80/EventStoryLine/tree/46edefee5e82e0917b823abe0a18bf8c7770f15c)。ESC 仓和 ECB+ 标注均声明 CC BY 3.0；按各自来源署名。不随本仓分发新闻原文或逐行预测。使用 ESC 原划分，只读 `annotated_data/v1.0/{dev,test}`，根部主题副本不重复计入。每个分区校验有序相对路径和文件 SHA-256 的汇总。
- 时间：取唯一可解析的 ISO 文档创建日（DCT）；排除无该日期、非 ISO 或日期冲突的文档。不推定时区和日内先后；空白天没有观测。划分按主题，不是时间前后分割；这是离线、主题隔离的回放，不是面向未来的部署验证。
- 特征：CAT-XML 原始 token 文本，整个句 0 排除以免 URL 泄漏。Unicode 字母小写、词 unigram/bigram、(1+\log(tf))、L2；不拟合词典、不读取标注字段作特征，不调用模型/API。日期只排序，行标识只打破并列。开发、测试各自空状态启动。
- 预测：每个文档同前一日结束时可见的所有簇原型比较余弦，相似度达到阈值就分配给最接近的簇，否则各自提议新簇。同日查询全部出预测才一起入库，两臂都可新增簇。冻结臂始终保留簇的首个文档原型；更新臂对归入旧簇的归一向量求和，再以余弦使用新质心。没有衰减、删除、重分配或合并操作；多个主题后续归进同一个簇也会视为误合并。
- 校准：共享余弦阈值，网格 `0.15, 0.25, 0.35, 0.45, 0.55, 0.65`，仅开发集两臂跨日同主题 pair F1 的等权均值最大者，并列取更高阈值；锁定后才读 test。
- 评估：只计跨日文档对；同 ECB+ 主题目录作主题金标，不代表单一事件。独立事件诊断使用 ECB+ 人工标注的 `ACT*` 跨文档共指链，链须连到有 token 锚点的事件 mention，且两份 XML 的有序文本和 token ID 必须一致。报告模型同簇捕获的已标注正例数；一个文档可有多个链，文档对只计一次，未标同链不当作事件负例，不报事件精度。预测阶段不读主题或事件标签。

## 复跑

Python 3.9+，只依赖标准库。从 Survey 仓库根目录执行，原始数据放在仓库外：

```bash
git clone https://github.com/tommasoc80/EventStoryLine.git /tmp/esc-v1-source
git -C /tmp/esc-v1-source checkout 46edefee5e82e0917b823abe0a18bf8c7770f15c
python scripts/benchmark_esc_crossday.py /tmp/esc-v1-source > /tmp/esc-crossday-v1.json
cmp /tmp/esc-crossday-v1.json results/esc-crossday-v1/results.json
```

适用结论限定在预选的跨日故事与这两个词面更新策略。下一步可在新盲测中验证更新门槛、时间门控或原型防漂移；不要用本轮 test 反调参数后声称盲测。
