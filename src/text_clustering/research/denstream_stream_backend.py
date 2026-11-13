"""Research-only causal window adapter for the audited faded-CF candidate.

No gold, K, parameter selection, network calls or factory registration. Input
normalization matches the existing DBSTREAM window protocol. A cached immutable
snapshot is reused only until the next successful learn. Snapshot-local labels
do not establish longitudinal cluster identity. Single-writer, not thread-safe.
"""
from dataclasses import asdict
import hashlib
import json
import math
import uuid

import numpy as np

from ..algorithms.river_stream_backend import (
    StreamPrediction, _numeric_array, _normalize_rows,
)
from .denstream_faded import DenStreamFaded, ALGORITHM_ID


BACKEND_PROTOCOL = "denstream-faded-cf-l2-snapshot-v1"


class DenStreamStreamBackend:
    """Transactional single-event learning and pure nearest-macrocenter queries.

    Constructor parameters are forwarded unchanged to the research core. A
    runner must explicitly record any departure from its defaults, especially
    n_samples_init. Fingerprints include *unaged* moments and buffered rows;
    snapshot hashes alone could conceal differences in the learned state.
    """

    def __init__(self, **parameters):
        self._model = DenStreamFaded(**parameters)
        self._model_id = uuid.uuid4().hex
        self._cached_revision = None
        self._cached_snapshot = None

    @property
    def state_revision(self):
        return self._model.n_samples_seen

    def state_fingerprint(self):
        # Deliberately coupled to the frozen candidate's immutable state schema.
        # Include all dataclass fields, rather than only prediction observables.
        state = self._model._state
        value = {key: val for key, val in vars(state).items()
                 if key not in ("p_micro_clusters", "o_micro_clusters")}
        for name in ("p_micro_clusters", "o_micro_clusters"):
            value[name] = [[int(key), asdict(stats)]
                           for key, stats in sorted(getattr(state, name).items())]
        value["parameters"] = asdict(self._model._parameters)
        value["backend_protocol"] = BACKEND_PROTOCOL
        return hashlib.sha256(json.dumps(value, sort_keys=True,
                              allow_nan=False).encode()).hexdigest()

    def _check_dimension(self, dimension):
        if dimension == 0:
            raise ValueError("Vectors must have at least one coordinate")
        if self._model.dimension is not None and dimension != self._model.dimension:
            raise ValueError("Vector dimension does not match learned state")

    def learn_one(self, vector):
        row = _numeric_array(vector)
        if row.ndim != 1:
            raise ValueError("learn_one requires one 1D vector")
        self._check_dimension(len(row))
        row = _normalize_rows(row.reshape(1, -1))[0]
        self._model.learn_one(tuple(float(v) for v in row))
        # Core commits atomically; a failed event must retain the valid cache.
        self._cached_revision = None
        self._cached_snapshot = None
        return self

    def _snapshot(self):
        if self._cached_revision != self.state_revision:
            snapshot = self._model.snapshot()
            self._cached_snapshot = snapshot
            self._cached_revision = self.state_revision
        return self._cached_snapshot

    def _metadata(self, snapshot):
        return {
            "backend": ALGORITHM_ID, "protocol": BACKEND_PROTOCOL,
            "parameters": asdict(self._model._parameters),
            "model_id": self._model_id, "state_revision": self.state_revision,
            "snapshot_id": f"{self._model_id}:{self.state_revision}",
            "label_scope": "snapshot; no persistent identity alignment",
            "normalization": "per-vector L2 float64; finite nonzero fixed dimension",
            "dimension": snapshot.dimension, "timestamp": snapshot.timestamp,
            "initialized": snapshot.initialized, "buffer_size": snapshot.buffer_size,
            "n_micro_clusters": len(snapshot.p_micro_clusters) + len(snapshot.o_micro_clusters),
            "n_p_micro_clusters": len(snapshot.p_micro_clusters),
            "n_o_micro_clusters": len(snapshot.o_micro_clusters),
            "n_macro_clusters": len(snapshot.macro_clusters),
            "empty_state_guard": ("uninitialized" if not snapshot.initialized else
                                  "empty_macro_centers" if not snapshot.macro_clusters else None),
            "unassigned_label": -1,
            "prediction_semantics": "nearest macrocenter; -1 only uninitialized/empty; no distance cutoff",
        }

    @property
    def metadata(self):
        return self._metadata(self._snapshot())

    def predict_window(self, vectors):
        rows = _numeric_array(vectors)
        if rows.ndim == 1 and rows.size == 0:
            rows = rows.reshape(0, self._model.dimension or 0)
        else:
            if rows.ndim != 2:
                raise ValueError("predict_window requires a 2D window")
            self._check_dimension(rows.shape[1])
        rows = _normalize_rows(rows)
        # Validate every query before even constructing a lazy macro snapshot.
        snapshot = self._snapshot()
        centers = [(label, stats.center()) for label, stats in
                   sorted(snapshot.macro_clusters.items())]
        labels = np.full(len(rows), -1, dtype=np.int64)
        if centers:
            for i, row in enumerate(rows):
                labels[i] = min(centers, key=lambda item: math.dist(row, item[1]))[0]
        labels.setflags(write=False)
        return StreamPrediction(labels, self._metadata(snapshot))
