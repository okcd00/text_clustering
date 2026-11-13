# Text Clustering Lab

**从文本集合到可维护的类别体系：可切换的算法工具 + 同协议评测。**

[打开能力与效果看板](https://dian.host/text_clustering/) · [API 与选型](docs/usage.md) · [评测口径](docs/protocols.md) · [参考文献](references/readme.md)

| 需求 | 起步方案 | 选择要点 |
|---|---|---|
| 已知类别数，整理一批文本 | KMeans / MiniBatchKMeans | 对比质量与速度；K 需要由业务提供 |
| 未知类别数，发现主题 | HDBSCAN / BERTopic 核心管线 / EVoC | 同时看准确率与覆盖率，不能只看被接收样本 |
| 新文本归入已有类别 | 原型路由 / kNN 路由 | 在独立验证集设拒识阈值 |
| 层级归属与新节点建议 | Hierarchical + 可插拔导航策略 | 节点新增建议需要业务审核 |
| 持续到来的文本流 | BIRCH / MiniBatch / 密度微簇 | 同时监测质量、状态增长与维护开销 |

看板首版公开 **17 组配置、85 次逐种子结果**：已知 K 与未知 K 两个独立协议，不混排总榜。其他场景已提供代码入口，其评测结果尚未纳入此公开快照。

## 五分钟开始

推荐 Python 3.10；部分可选算法需要 Python 3.11 或更高版本。

```bash
git clone https://github.com/okcd00/text_clustering.git
cd text_clustering
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python examples/offline_demo.py
python scripts/serve_dashboard.py
```

最小调用（演示用字符哈希；真实工作请替换为语义 embedding）：

```python
from text_clustering import Document
from text_clustering.embeddings import HashingEmbedder
from text_clustering.factory import build_tool

tool = build_tool("partitioning", HashingEmbedder(),
                  algorithm={"name": "kmeans", "n_clusters": 2, "seed": 0})
result = tool.fit([Document(str(i), text) for i, text in enumerate(
    ["refund my payment", "cancel payment", "reset password", "forgot password"])])
print(result.partition.labels)
```

真实使用可注入具有 `identity` 和 `encode(texts)` 的 embedder，或用已缓存的矩阵调用 `fit(..., vectors=X, encoder_identity=embedder.identity)`。本仓库不附带内部服务、密钥或 embedding 缓存。

## 文件导航

- `src/text_clustering/`：工具类、算法后端、共享类型及研究改编实现。
- `examples/`：无需网络或模型下载的最小端到端示例。
- `docs/`：静态评测看板、逐种子公开结果、调用与评测说明。
- `datasets/`：数据集出处、下载入口及引用，不附数据。
- `references/`：82 篇参考工作的来源和 BibTeX，不附论文 PDF。
- `tests/`：公共 API 的离线验收。
- `scripts/`：看板启动与发布审核。

论文原方法、核心改编和工程基线需区别解读，例如 `bertopic_pipeline` 是降维与聚类核心，**不包含完整 BERTopic 主题表示产品**。当前公开结果使用的历史编码器未提供可下载 checkpoint，数值复现条件见[评测说明](docs/protocols.md)。

## 公开协作

只在此公开仓库中提交可公开代码和汇总结果。内部数据、派生向量、逐条预测、内部服务配置和工作历史不得同步。发布流程见[公开发布约定](docs/publication.md)。参考论文与第三方代码遵循各自权利和许可证；此版本不额外授予整个项目的统一开源许可证。

安装验收与浏览器检查状态见[验证记录](docs/verification.md)。
