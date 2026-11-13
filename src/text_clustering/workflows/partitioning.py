"""Task orchestration independent of the imported clustering algorithm."""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.sparse import issparse
from ..index import ClusterIndex
from ..types import ClusterAlgorithm, Embedder, Partition
from ..vectors import unit_rows


@dataclass
class ColdStartResult:
    partition: Partition
    index: ClusterIndex


class ColdStart:
    def __init__(self, embedder: Embedder, algorithm: ClusterAlgorithm):
        self.embedder, self.algorithm = embedder, algorithm

    def fit(self, documents, vectors=None, encoder_identity=None, *, preprocessing="l2"):
        """Fit on L2-normalized float64 rows, or a dtype-preserving raw copy.

        Both policies build cosine index statistics from normalized float64 rows.
        Audit hashes describe C-order array bytes before the backend runs (not
        cache-file hashes); dtype and shape are recorded separately. The metadata
        key ``workflow_preprocessing`` is reserved for this workflow.
        """
        if not isinstance(preprocessing, str) or preprocessing not in ("l2", "none"):
            raise ValueError("preprocessing must be 'l2' or 'none'")
        documents = list(documents)
        ids = [d.id for d in documents]
        if not documents or len(set(ids)) != len(ids):
            raise ValueError("need nonempty documents with unique IDs")
        if vectors is not None and encoder_identity != self.embedder.identity:
            raise ValueError("precomputed vectors must identify the same encoder")
        source = self.embedder.encode([d.text for d in documents]) if vectors is None else vectors
        if issparse(source) or np.ma.isMaskedArray(source):
            raise ValueError("expected a dense real numeric matrix")
        source = np.asarray(source)
        if (source.ndim != 2 or source.shape[1] == 0 or
                source.dtype.kind not in "iuf" or not np.isfinite(source).all()):
            raise ValueError("expected a finite dense real numeric matrix [n, d], d > 0")
        if len(source) != len(documents):
            raise ValueError("document/vector count mismatch")
        x = unit_rows(source)
        fit_vectors = (x if preprocessing == "l2" else source).copy()
        audit = {
            "policy": preprocessing,
            "source": "embedding" if vectors is None else "precomputed",
            "shape": list(source.shape),
            "source_dtype": str(source.dtype),
            "source_sha256": hashlib.sha256(source.tobytes(order="C")).hexdigest(),
            "fit_dtype": str(fit_vectors.dtype),
            "fit_sha256": hashlib.sha256(fit_vectors.tobytes(order="C")).hexdigest(),
        }
        result = self.algorithm.fit(fit_vectors)
        if "workflow_preprocessing" in result.metadata:
            raise ValueError("backend metadata uses reserved key workflow_preprocessing")
        result.metadata = {**result.metadata, "workflow_preprocessing": audit}
        labels = np.asarray(result.labels)
        if labels.shape != (len(x),) or not np.issubdtype(labels.dtype, np.integer) or (labels < -1).any():
            raise ValueError("algorithm must return one integer label >= -1 per document")
        clusters = sorted(set(labels) - {-1})
        sums = np.array([x[labels == c].sum(axis=0) for c in clusters]).reshape(-1, x.shape[1])
        counts = np.array([np.sum(labels == c) for c in clusters], dtype=int)
        index = ClusterIndex(tuple(str(c) for c in clusters), sums, counts,
                             self.embedder.identity, {}, set(ids))
        return ColdStartResult(result, index)

    def fit_stream(self, batch_factory):
        """Replayable document batches, two encoder passes, no full N*d matrix.

        The caller supplies a fresh iterator each time. IDs/text/order are checked
        by a digest across passes. Labels and seen IDs still require O(N) memory.
        Use StreamingMiniBatchKMeans; group/density backends are full-data only.
        """
        import hashlib
        import json
        if not hasattr(self.algorithm, "fit_stream"):
            raise TypeError("algorithm does not implement streaming fit")
        first_digest, second_digest = hashlib.sha256(), hashlib.sha256()
        seen, dimension = set(), [None]

        def encoded_batches():
            for docs in batch_factory():
                docs = list(docs)
                if not docs:
                    continue
                for d in docs:
                    if d.id in seen:
                        raise ValueError("duplicate document ID")
                    seen.add(d.id)
                    first_digest.update(json.dumps([d.id, d.text], ensure_ascii=False).encode() + b"\n")
                x = unit_rows(self.embedder.encode([d.text for d in docs]))
                if len(x) != len(docs) or dimension[0] not in (None, x.shape[1]):
                    raise ValueError("inconsistent embedding batch")
                dimension[0] = x.shape[1]
                yield x
        model = self.algorithm.fit_stream(encoded_batches)
        sums = np.zeros((model.n_clusters, dimension[0]))
        counts = np.zeros(model.n_clusters, dtype=int)
        labels = []
        for docs in batch_factory():
            docs = list(docs)
            if not docs:
                continue
            for d in docs:
                second_digest.update(json.dumps([d.id, d.text], ensure_ascii=False).encode() + b"\n")
            x = unit_rows(self.embedder.encode([d.text for d in docs]))
            if x.shape != (len(docs), dimension[0]):
                raise ValueError("inconsistent embedding batch")
            batch_labels = model.predict(x)
            np.add.at(sums, batch_labels, x)
            np.add.at(counts, batch_labels, 1)
            labels.extend(batch_labels)
        if first_digest.digest() != second_digest.digest():
            raise ValueError("stream changed between passes (IDs, text or order)")
        nonempty = np.flatnonzero(counts)
        partition = Partition(np.asarray(labels, dtype=int), {
            "algorithm": "StreamingMiniBatchKMeans", "passes": 2,
            "seed": self.algorithm.seed, "requested_k": model.n_clusters,
            "actual_k": len(nonempty), "embedding_passes": 2})
        index = ClusterIndex(tuple(str(i) for i in nonempty), sums[nonempty], counts[nonempty],
                             self.embedder.identity, {}, seen)
        return ColdStartResult(partition, index)
