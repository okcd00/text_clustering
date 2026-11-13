"""Guarded native River DBSTREAM, not a batch-fit clustering adapter.

One persistent upstream model learns each event exactly once. ``predict_window``
only queries its current state; it neither learns nor replays historical data.
Macro reclustering of microcluster summaries is native DBSTREAM, not a fit on
the query window. Labels are local to (model_id, state_revision), not stable
identities across learning steps. A window predicted after learning that same
window is retrospective evaluation, not prequential evaluation.

Audit: River 0.26.1 installed dbstream.py, SHA-256 below. Upstream defaults are
unchanged. L2 input normalization and the empty-state -1 guard are explicit
adapter semantics. Native predict_one returns 0 without centres, and native
_recluster can retain stale centres when its microcluster mapping is empty.
Check microstate BEFORE reading public centres; never repair upstream caches.
With nonempty centres native prediction is nearest macro-centroid assignment,
NOT an outlier detector. Predictions may materialize macro caches, but must not
change microclusters, shared density, timestamps, or the learning revision.

Importing this module needs no River and is Python 3.9 compatible. Construction
requires Python >=3.11 and separately installed river==0.26.1. No installation,
factory registration, upstream patching, hyperparameter tuning, gold labels, or
requested cluster count is performed here. The source pin covers dbstream.py,
not the entire dependency closure. River is BSD-3-Clause; dependency notices
remain applicable. This single-writer adapter is not thread-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import inspect
from pathlib import Path
import sys
from types import MappingProxyType
import uuid

import numpy as np


RIVER_VERSION = "0.26.1"
SOURCE_SHA256 = "dae803ea16562c02eda6c6fe7976cc4f89c88ce7ecf8ea3bbfc9c2bcc10962d9"
BACKEND_PROTOCOL = "river-dbstream-native-l2-snapshot-v1"
UPSTREAM_DEFAULTS = MappingProxyType({
    "clustering_threshold": 1.0,
    "fading_factor": 0.01,
    "cleanup_interval": 2,
    "intersection_factor": 0.3,
    "minimum_weight": 1.0,
})


def _load_dbstream():
    """Lazy, fail-closed dependency and source audit; no network access."""
    if sys.version_info < (3, 11):
        raise ImportError("River DBSTREAM requires Python >=3.11 and river==0.26.1")
    try:
        version = importlib.metadata.version("river")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError("Optional dependency missing: install river==0.26.1 separately") from exc
    if version != RIVER_VERSION:
        raise RuntimeError(f"Expected river=={RIVER_VERSION}; found {version}")
    cls = importlib.import_module("river.cluster.dbstream").DBSTREAM
    source = inspect.getsourcefile(cls)
    if source is None:
        raise RuntimeError("Cannot audit River DBSTREAM source file")
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise RuntimeError(f"River DBSTREAM source SHA-256 mismatch: {digest}")
    return cls


def _numeric_array(values):
    try:
        raw = np.asarray(values)
        if raw.dtype.kind not in "iuf":
            raise ValueError("Vectors must contain real numeric values")
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Vectors must be rectangular real numeric arrays") from exc
    if not np.isfinite(array).all():
        raise ValueError("Vectors must be finite")
    return array


def _normalize_rows(rows):
    """Scale first so finite huge/subnormal vectors normalize without overflow."""
    if not len(rows):
        return rows
    scale = np.max(np.abs(rows), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("Vectors must be nonzero")
    scaled = rows / scale
    return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)


@dataclass(frozen=True)
class StreamPrediction:
    """Detached window result; labels are read-only and metadata is JSON-safe.

    Mutating the returned metadata cannot change the backend or later results.
    The state revision counts successful learns, not macro-cache refreshes.
    """

    labels: np.ndarray
    metadata: dict

    @property
    def state_revision(self):
        return self.metadata["state_revision"]


class RiverStreamBackend:
    """Fixed-default incremental DBSTREAM with snapshot-scoped predictions.

    Only a successful ``learn_one`` establishes the input dimension and advances
    ``state_revision``. Invalid input is fully rejected before any model access
    or mutation. Upstream runtime failures are propagated, not transactionally
    rolled back. Metadata and prediction may incur native macro reclustering;
    include that cost in evaluation timing, not just the learn_one call.
    """

    def __init__(self):
        self._model = _load_dbstream()(**dict(UPSTREAM_DEFAULTS))
        self._dimension = None
        self._state_revision = 0
        self._model_id = uuid.uuid4().hex

    @property
    def state_revision(self):
        return self._state_revision

    def state_fingerprint(self):
        """Hash learned microstate, excluding UUID and lazy macro caches.

        Audited against the pinned upstream source. Reading this has no effect
        on the native state and cannot trigger macro reclustering.
        """
        import json
        m = self._model
        state = {"revision": self._state_revision, "dimension": self._dimension,
                 "time": m._time_stamp, "s": m.s, "s_t": m.s_t,
                 "micro": {k: {"center": v.center, "weight": v.weight,
                                "last_update": v.last_update}
                           for k, v in m.micro_clusters.items()}}
        return hashlib.sha256(json.dumps(state, sort_keys=True,
                                        allow_nan=False).encode()).hexdigest()

    def _check_dimension(self, dimension):
        if dimension == 0:
            raise ValueError("Vectors must have at least one coordinate")
        if self._dimension is not None and dimension != self._dimension:
            raise ValueError(f"Expected dimension {self._dimension}; got {dimension}")

    def learn_one(self, vector):
        """Normalize a fresh dense vector and perform exactly one native update."""
        row = _numeric_array(vector)
        if row.ndim != 1:
            raise ValueError("learn_one requires one 1D vector")
        self._check_dimension(len(row))
        row = _normalize_rows(row.reshape(1, -1))[0]
        # Upstream keeps the input mapping by reference when creating a microcluster.
        # Always allocate a new dict with all coordinates, including zeroes.
        self._model.learn_one({i: float(value) for i, value in enumerate(row)})
        self._dimension = len(row)
        self._state_revision += 1
        return self

    def _snapshot_metadata(self):
        micro_count = len(self._model.micro_clusters)
        if micro_count == 0:
            # Do not trust even public centers/n_clusters here: native empty-label
            # reclustering does not clear an earlier macro cache.
            macro_count = 0
            guard = "empty_microstate"
        else:
            centers = self._model.centers
            macro_count = len(centers)
            if self._model.n_clusters != macro_count:
                raise RuntimeError("River macro count and centres disagree")
            guard = "empty_macro_centers" if macro_count == 0 else None
        # s has no public property. Read this audited attribute only for counts;
        # zeroed entries are retained by upstream cleanup and still cost memory.
        stored_edges = sum(len(row) for row in self._model.s.values())
        positive_edges = sum(value > 0 for row in self._model.s.values() for value in row.values())
        return {
            "backend": "river.DBSTREAM",
            "protocol": BACKEND_PROTOCOL,
            "river_version": RIVER_VERSION,
            "source_sha256": SOURCE_SHA256,
            "license": "BSD-3-Clause (River; not a dependency-closure license)",
            "parameters": dict(UPSTREAM_DEFAULTS),
            "normalization": "per-vector L2 float64; finite nonzero fixed dimension",
            "model_id": self._model_id,
            "state_revision": self._state_revision,
            "snapshot_id": f"{self._model_id}:{self._state_revision}",
            "label_scope": "snapshot; no persistent identity alignment",
            "dimension": self._dimension,
            "n_micro_clusters": micro_count,
            "n_macro_clusters": macro_count,
            "n_stored_edges": stored_edges,
            "n_positive_edges": int(positive_edges),
            "edge_count_semantics": "stored s entries, including zeroes; positive is not time-decayed connectivity",
            "empty_state_guard": guard,
            "unassigned_label": -1,
            "prediction_semantics": "native nearest macro centroid; -1 only for empty state",
        }

    @property
    def metadata(self):
        """Detached current metadata; may materialize macro caches, never learns."""
        return self._snapshot_metadata()

    def predict_window(self, vectors):
        """Predict an entire validated window at one revision, with no learning.

        ``[]`` is a valid empty window. Other inputs must be a 2D numeric array;
        even an explicitly shaped empty array must have a compatible dimension.
        Before the first learn, queries do not establish a training dimension.
        """
        rows = _numeric_array(vectors)
        if rows.ndim == 1 and rows.size == 0:
            rows = rows.reshape(0, self._dimension or 0)
        else:
            if rows.ndim != 2:
                raise ValueError("predict_window requires a 2D window of vectors")
            self._check_dimension(rows.shape[1])
        rows = _normalize_rows(rows)
        # Validation of EVERY row precedes even lazy macro-cache materialization.
        metadata = self._snapshot_metadata()
        if metadata["empty_state_guard"] is not None:
            labels = np.full(len(rows), -1, dtype=np.int64)
        else:
            labels = np.array([
                self._model.predict_one({i: float(value) for i, value in enumerate(row)})
                for row in rows
            ], dtype=np.int64)
        labels.setflags(write=False)
        return StreamPrediction(labels=labels, metadata=metadata)
