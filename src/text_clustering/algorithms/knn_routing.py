"""Exact cosine kNN engineering baseline, not a named-paper reproduction.

Uniform vote; ties use sorted class ID. Rejection score is kth-neighbor cosine,
not class probability. Memory grows with all training support. No query fitting.
"""
from numbers import Integral
import numpy as np
from ..vectors import unit_rows
from ..workflows.prototype_routing import _vectors, _support_ids, _gates


class KNNRouting:
    def __init__(self, k=5, threshold=0.55):
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 1:
            raise ValueError('k must be a positive integer')
        self.k = int(k)
        self.threshold, _ = _gates(threshold, 0)

    def fit(self, vectors, leaf_ids, parent_ids=None):
        x = unit_rows(_vectors(vectors)).astype(np.float32)
        labels = _support_ids(leaf_ids, len(x), 'leaf_ids')
        if len(x) < self.k:
            raise ValueError('insufficient training support for k')
        if parent_ids is not None:
            _support_ids(parent_ids, len(x), 'parent_ids')
        self.classes, self.owners = np.unique(labels, return_inverse=True)
        self.support = x
        return self

    def score(self, vectors, batch_size=256):
        if not hasattr(self, 'support'):
            raise ValueError('fit training support first')
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError('batch_size must be positive integer')
        values = _vectors(vectors, self.support.shape[1], allow_empty=True, copy=False)
        labels, confidence = [], []
        for start in range(0, len(values), batch_size):
            similarities = unit_rows(values[start:start+batch_size]).astype(np.float32) @ self.support.T
            # Stable support-position tie break, including ties at kth boundary.
            nearest = np.argsort(-similarities, axis=1, kind='stable')[:, :self.k]
            for row, neighbors in zip(similarities, nearest):
                votes = np.bincount(self.owners[neighbors], minlength=len(self.classes))
                labels.append(self.classes[int(np.argmax(votes))])
                confidence.append(row[neighbors[-1]])
        return np.asarray(labels, dtype=self.classes.dtype), np.asarray(confidence, dtype=np.float32)

    def predict(self, vectors, ids, *, batch_size=256):
        ids = list(ids)
        if len(ids) != len(vectors) or len(set(ids)) != len(ids) or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError('unique nonempty string IDs must align with queries')
        threshold, _ = _gates(self.threshold, 0)
        labels, scores = self.score(vectors, batch_size)
        return {i: dict(node_id=str(label) if score >= threshold else None,
                        novel=bool(score < threshold), score=float(score))
                for i, label, score in zip(ids, labels, scores)}
