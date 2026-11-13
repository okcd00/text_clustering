"""Train-support cosine routing: flat or parent-first; no query fitting.

Engineering baselines, not a reproduction of a named paper. Multi-prototype
centers use ordinary sklearn KMeans, then cosine-normalize the centers.
"""
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from sklearn.cluster import KMeans
from ..vectors import unit_rows


@dataclass
class RoutingScores:
    leaves: np.ndarray
    parents: np.ndarray
    confidence: np.ndarray
    margin: np.ndarray


class PrototypeRouter:
    def __init__(self, prototypes=1, seed=0, hierarchical=False):
        if prototypes not in (1, 3):
            raise ValueError('supported prototype counts: 1, 3')
        self.prototypes, self.seed, self.hierarchical = prototypes, seed, hierarchical

    def fit(self, vectors, leaf_ids, parent_ids):
        x = unit_rows(vectors).astype(np.float32)
        leaf_ids, parent_ids = np.asarray(leaf_ids), np.asarray(parent_ids)
        if len(x) != len(leaf_ids) or len(x) != len(parent_ids):
            raise ValueError('unaligned training support')
        self.leaves = np.unique(leaf_ids)
        self.parents = np.unique(parent_ids)
        if len(self.leaves) < 2:
            raise ValueError('at least two known leaves required')
        self.leaf_parents, centers, owners = [], [], []
        for i, leaf in enumerate(self.leaves):
            mask = leaf_ids == leaf
            parents = np.unique(parent_ids[mask])
            if len(parents) != 1:
                raise ValueError('leaf must have one parent')
            self.leaf_parents.append(parents[0])
            members = x[mask]
            if self.prototypes == 1:
                c = members.mean(axis=0, keepdims=True)
            else:
                if len(members) < self.prototypes:
                    raise ValueError('insufficient prototype support')
                c = KMeans(n_clusters=self.prototypes, n_init=10,
                           random_state=self.seed, algorithm='lloyd').fit(members).cluster_centers_
            centers.extend(c)
            owners.extend([i] * len(c))
        self.leaf_parents = np.asarray(self.leaf_parents)
        self.centers = unit_rows(centers).astype(np.float32)
        self.owners = np.asarray(owners)
        parent_means = np.asarray([x[parent_ids == p].mean(axis=0) for p in self.parents])
        defined = np.max(np.abs(parent_means), axis=1) > 1e-12
        if self.hierarchical and not defined.all():
            raise ValueError('parent-first routing requires nonzero parent centers')
        # Opposing children can cancel without invalidating their leaf centers.
        # Exact zero is an unavailable parent proposal, never a cosine score.
        self.parent_centers = np.zeros_like(parent_means, dtype=np.float32)
        self.parent_centers[defined] = unit_rows(parent_means[defined]).astype(np.float32)
        return self

    def score(self, vectors, batch_size=None):
        """Score without fitting query data; optionally bound temporary row count.

        None preserves the original full-matrix path. Batching bounds normalized
        queries and similarity matrices, not input storage, model size, or the
        returned O(n) arrays. BLAS may round differently across batch shapes;
        applications near decision thresholds must validate their operating point.
        """
        if batch_size is None:
            return self._score_normalized(unit_rows(vectors).astype(np.float32))
        if (isinstance(batch_size, bool) or not isinstance(batch_size, Integral)
                or batch_size < 1):
            raise ValueError('batch_size must be a positive integer or None')
        values = np.asarray(vectors)
        if values.ndim != 2:
            raise ValueError('expected matrix [n, d]')
        if len(values) == 0:
            return self._score_normalized(unit_rows(values).astype(np.float32))
        fields = ('leaves', 'parents', 'confidence', 'margin')
        output = None
        for start in range(0, len(values), int(batch_size)):
            end = min(start + int(batch_size), len(values))
            block = self._score_normalized(unit_rows(values[start:end]).astype(np.float32))
            if output is None:
                output = {name: np.empty(len(values), dtype=getattr(block, name).dtype)
                          for name in fields}
            for name in fields:
                output[name][start:end] = getattr(block, name)
        return RoutingScores(**output)

    def _score_normalized(self, x):
        similarity = x @ self.centers.T
        scores = np.stack([similarity[:, self.owners == i].max(axis=1)
                           for i in range(len(self.leaves))], axis=1)
        defined = np.any(self.parent_centers != 0, axis=1)
        if defined.any():
            parent_scores = x @ self.parent_centers.T
            parent_scores[:, ~defined] = -np.inf
            parents = self.parents[parent_scores.argmax(axis=1)]
        else:
            parents = np.full(len(x), None, dtype=object)
        if self.hierarchical:
            scores = np.where(self.leaf_parents[None, :] == parents[:, None], scores, -np.inf)
        order = np.argsort(-scores, axis=1, kind='stable')[:, :2]
        a, b = np.take_along_axis(scores, order, axis=1).T
        # Singleton child branches have no competing child; cosine range is [-1,1].
        margin = np.where(np.isfinite(b), a-b, 2.0)
        return RoutingScores(self.leaves[order[:, 0]], parents, a, margin)

    @staticmethod
    def predict(scores, ids, threshold, min_margin):
        if len(ids) != len(scores.leaves):
            raise ValueError('unaligned queries')
        result = {}
        for i, identity in enumerate(ids):
            novel = bool(scores.confidence[i] < threshold)
            accept = not novel and scores.margin[i] >= min_margin
            result[identity] = dict(node_id=str(scores.leaves[i]) if accept else None,
                                    novel=novel, parent_id=(None if scores.parents[i] is None
                                                           else str(scores.parents[i])),
                                    status='novel' if novel else 'assigned' if accept else 'ambiguous')
        return result


def tune(scores, gold, min_known_coverage=0.8):
    """Validation-query only. Novelty=low similarity; low margin=abstention.

    Fixed score/margin quantiles plus all-accept/all-reject endpoints. Evaluation
    cost is identical per gate (precomputed scores), so its tie break is constant.
    """
    known = np.array([g['role'] == 'known' for g in gold])
    correct = scores.leaves == np.array([g['leaf'] for g in gold])
    if not known.any() or known.all():
        raise ValueError('validation needs known and hidden queries')
    quantiles = np.linspace(0, 1, 21)
    thresholds = np.unique(np.r_[-1.000001, np.quantile(scores.confidence, quantiles), 1.000001])
    margins = np.unique(np.r_[0., np.quantile(scores.margin, quantiles), 2.000001])
    best, trials, feasible = None, 0, 0
    for threshold in thresholds:
        novel = scores.confidence < threshold
        tp = (novel & ~known).sum()
        fp = (novel & known).sum()
        f1 = float(2*tp / ((~known).sum()+tp+fp))
        for margin in margins:
            trials += 1
            accepted = ~novel & (scores.margin >= margin)
            coverage = float((accepted & known).sum()/known.sum())
            if coverage < min_known_coverage:
                continue
            feasible += 1
            accuracy = float((accepted & known & correct).sum()/known.sum())
            key = (-f1, -accuracy, float(threshold), float(margin))
            if best is None or key < best[0]:
                best = (key, dict(threshold=float(threshold), min_margin=float(margin),
                                 validation_novel_f1=f1, validation_known_accuracy=accuracy,
                                 validation_known_coverage=coverage))
    if best is None:
        raise ValueError('no feasible validation gate; do not relax guardrail')
    return dict(best[1], trials=trials, feasible=feasible, min_known_coverage=min_known_coverage)
