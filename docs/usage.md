# API 与选型

统一工厂：`text_clustering.factory.build_tool`。算法由独立后端实现；切换算法不会自动替你选择 K、阈值或时间流顺序。

## 批量聚类

```python
from text_clustering.factory import build_tool
from text_clustering.embeddings import HashingEmbedder
embedder = HashingEmbedder()  # 只作 smoke；生产注入真实语义模型
pipeline = build_tool('partitioning', embedder,
    algorithm={'name': 'minibatch_kmeans', 'n_clusters': 10, 'seed': 0})
# result = pipeline.fit(documents)
# result = pipeline.fit(documents, vectors=X, encoder_identity=embedder.identity)
```

核心可用：`kmeans`、`minibatch_kmeans`、`spherical_kmeans`、`agglomerative_cosine`、`hdbscan`、`gpkmeans`。约束聚类需提供业务 must-link / cannot-link，而非把测试标签作为约束。

可选安装：

```bash
python -m pip install '.[discovery]'   # BERTopic 核心、EVoC
python -m pip install '.[finch]'       # FINCH
python -m pip install '.[classix]'     # CLASSIX
python -m pip install '.[streaming]'   # River，需要 Python >= 3.11
python -m pip install '.[models]'      # SentenceTransformerEmbedder
```

可选依赖与版本见 `pyproject.toml`。首版验收覆盖核心安装和离线三任务示例，不等于所有可选组合均已在干净环境验证。

## 路由与层级

完整可执行过程见 `examples/offline_demo.py`：聚类获得 index，再构建 routing 工具；导航由 Taxonomy / Node 表达层级。路由与导航有不同协议，不把聚类准确率当成拒识能力。

## 持续更新

`build_tool('evolution', algorithm={...}, config={...})` 接受预计算向量和显式时间顺序，调用 `run(vectors, order, on_prediction=...)`。后端包括 `sklearn_birch_stream`、`sklearn_minibatch_stream`、`birch_capacity_freeze`、`birch_rolling_rebuild` 等。`on_prediction` 是评估回调，不得使用未来窗口标签调参。完整参数和回调签名见 `src/text_clustering/workflows/evolution.py` 及对应后端 docstring。
