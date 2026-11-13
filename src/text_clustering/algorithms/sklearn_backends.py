"""Official sklearn backends. These are baselines, not paper reproductions."""
from sklearn.cluster import HDBSCAN, KMeans, MiniBatchKMeans
from ..types import Partition


class KMeansBackend:
    def __init__(self, n_clusters, seed=0, minibatch=True, batch_size=1024):
        self.n_clusters, self.seed = n_clusters, seed
        self.minibatch, self.batch_size = minibatch, batch_size

    def fit(self, vectors):
        if not 1 <= self.n_clusters <= len(vectors):
            raise ValueError("require 1 <= K <= n")
        kwargs = dict(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        model = (MiniBatchKMeans(batch_size=self.batch_size, **kwargs) if self.minibatch
                 else KMeans(**kwargs))
        labels = model.fit_predict(vectors)
        return Partition(labels, {"algorithm": type(model).__name__, "seed": self.seed,
                                  "inertia": float(model.inertia_),
                                  "params": model.get_params(), "n_iter": int(model.n_iter_)})


class HDBSCANBackend:
    """Unknown K; retains -1 noise. Full-data backend, not a large-N guarantee."""
    def __init__(self, min_cluster_size=5, min_samples=None):
        self.min_cluster_size, self.min_samples = min_cluster_size, min_samples

    def fit(self, vectors):
        model = HDBSCAN(min_cluster_size=self.min_cluster_size,
                       min_samples=self.min_samples, metric="euclidean")
        return Partition(model.fit_predict(vectors), {"algorithm": "HDBSCAN",
            "min_cluster_size": self.min_cluster_size, "min_samples": self.min_samples})


class StreamingMiniBatchKMeans:
    """Two-pass coldstart backend: fit incremental batches, then predict batches.

    Memory for embeddings is bounded by max(batch size, K) rows. This differs
    from sklearn.fit's multiple mini-batch passes; epochs are explicit in caller.
    """
    def __init__(self, n_clusters, seed=0):
        if n_clusters < 1:
            raise ValueError("n_clusters must be positive")
        self.n_clusters, self.seed = n_clusters, seed

    def fit_stream(self, vector_batches):
        import numpy as np
        model = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=self.seed,
                               n_init=1, reassignment_ratio=0.0)
        initial, size, initialized = [], 0, False
        for x in vector_batches():
            if not initialized:
                initial.append(x)
                size += len(x)
                if size < self.n_clusters:
                    continue
                model.partial_fit(np.concatenate(initial))
                initial = []
                initialized = True
            else:
                model.partial_fit(x)
        if not initialized:
            raise ValueError("stream has fewer documents than K")
        return model
