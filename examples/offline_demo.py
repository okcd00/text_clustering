"""No network, no model download. Demonstrates contracts, not semantic quality."""
from dataclasses import asdict
import json
from text_clustering import Document, Node, Taxonomy
from text_clustering.embeddings import HashingEmbedder
from text_clustering.factory import build_tool


def main():
    encoder = HashingEmbedder()
    docs = [Document(str(i), text) for i, text in enumerate([
        "退款申请 退款申请", "申请退款", "退款申请进度", "账户密码重置", "重置密码", "账户密码修改"])]
    cold = build_tool('partitioning', encoder,
                      algorithm={'name': 'kmeans', 'n_clusters': 2, 'seed': 0}).fit(docs)
    router = build_tool('routing', encoder,
                        config={'min_cosine': 0.4, 'min_margin': 0.05})
    decision, updated = router.update(Document("new", "申请退款"), cold.index, expected_revision=0)
    taxonomy = Taxonomy([Node("root", "业务", None), Node("refund", "退款申请", "root"),
                         Node("password", "重置密码", "root")])
    navigation = build_tool('navigation', encoder).predict(Document("query", "退款申请"), taxonomy)
    print(json.dumps({"disclaimer": "character hashing smoke test, not model evaluation",
        "coldstart_labels": cold.partition.labels.tolist(), "onepass": asdict(decision),
        "index_revisions": [cold.index.revision, updated.revision],
        "hierarchical": asdict(navigation)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
