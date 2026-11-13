"""Versioned sufficient statistics; safe JSON persistence, no pickle."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .vectors import unit_rows


@dataclass
class ClusterIndex:
    ids: tuple
    sums: np.ndarray
    counts: np.ndarray
    encoder_identity: str
    descriptions: dict
    seen_ids: set
    revision: int = 0

    def __post_init__(self):
        self.ids = tuple(self.ids)
        self.sums = np.array(self.sums, dtype=float, copy=True)
        raw_counts = np.asarray(self.counts)
        if (self.sums.ndim != 2 or self.sums.shape[1] < 1 or
                raw_counts.shape != (len(self.ids),) or len(self.sums) != len(self.ids)):
            raise ValueError("index shape mismatch")
        if not np.isfinite(self.sums).all() or not np.isfinite(raw_counts).all():
            raise ValueError("nonfinite index")
        if (raw_counts < 1).any() or (raw_counts != np.floor(raw_counts)).any():
            raise ValueError("counts must be positive integers")
        if len(set(self.ids)) != len(self.ids) or any(not isinstance(i, str) for i in self.ids):
            raise ValueError("cluster IDs must be unique strings")
        if not self.encoder_identity or self.revision < 0:
            raise ValueError("encoder identity and nonnegative revision required")
        self.counts = raw_counts.astype(np.int64)
        self.descriptions = dict(self.descriptions)
        if set(self.descriptions) - set(self.ids):
            raise ValueError("description refers to unknown cluster")
        self.seen_ids = set(self.seen_ids)
        # A cancellation centroid cannot be used for cosine retrieval.
        unit_rows(self.sums)

    @property
    def centers(self):
        return unit_rows(self.sums)

    def copy(self):
        return ClusterIndex(self.ids, self.sums, self.counts, self.encoder_identity,
                            self.descriptions, self.seen_ids, self.revision)

    def save(self, path):
        path = Path(path)
        payload = dict(schema_version=1, ids=self.ids, sums=self.sums.tolist(),
                       counts=self.counts.tolist(), encoder_identity=self.encoder_identity,
                       descriptions=self.descriptions, seen_ids=sorted(self.seen_ids),
                       revision=self.revision)
        # Atomic replacement prevents partial snapshots on interruption.
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False,
                                         encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
            temp = stream.name
        os.replace(temp, path)

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.pop("schema_version", None) != 1:
            raise ValueError("unsupported index schema")
        return cls(**payload)

    @classmethod
    def from_assignments(cls, document_ids, vectors, cluster_ids, encoder_identity, descriptions=None):
        """Import existing clustering. None means unassigned/noise; labels are strings."""
        x = unit_rows(vectors)
        if not len(document_ids) == len(x) == len(cluster_ids) or len(set(document_ids)) != len(document_ids):
            raise ValueError("unique aligned document IDs required")
        if any(c is not None and not isinstance(c, str) for c in cluster_ids):
            raise ValueError("cluster IDs must be strings or None for noise")
        ids = tuple(sorted(set(cluster_ids) - {None}))
        position = {c: i for i, c in enumerate(ids)}
        sums, counts = np.zeros((len(ids), x.shape[1])), np.zeros(len(ids), dtype=int)
        for vector, cluster in zip(x, cluster_ids):
            if cluster is not None:
                sums[position[cluster]] += vector
                counts[position[cluster]] += 1
        return cls(ids, sums, counts, encoder_identity, descriptions or {}, set(document_ids))
