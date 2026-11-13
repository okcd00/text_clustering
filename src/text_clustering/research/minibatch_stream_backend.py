"""Research-only, fixed-K sklearn MiniBatchKMeans with explicit chunk arrivals.

Audited against sklearn 1.6.1 and its official versioned API documentation:
https://scikit-learn.org/1.6/modules/generated/sklearn.cluster.MiniBatchKMeans.html
Only complete chunks reach partial_fit, once each, in arrival order. No fit,
replay, implicit flush, tuning, gold labels, factory registration or quality
admission. L2-normalized inputs do NOT make this spherical k-means: native
centres are not renormalized. After initialization there is no novelty rejection.

Single writer, not thread-safe. Prediction copies the entire estimator; that
cost belongs to prediction timing. The complete estimator (including RNG and
counts) and pending rows remain owned by this instance, and survive trusted
same-environment pickle round trips. Never load untrusted pickles. Fingerprints
are content digests, not portable checkpoints; they omit only the random model
identity. The source pin covers _kmeans.py, not compiled/dependency closure or
hardware/thread-level numerical reproducibility.
"""
from copy import deepcopy
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
import uuid

import numpy as np

from ..algorithms.river_stream_backend import StreamPrediction, _numeric_array, _normalize_rows

SKLEARN_VERSION = "1.6.1"
SOURCE_SHA256 = "d9c8727fa0a4e89689a1c77ca470a7f8d1fbc33fcdcd23ba36a838faef6f43d4"
BACKEND_PROTOCOL = "sklearn-minibatch-fixed-k-l2-snapshot-v1"
# Protocol fixes n_init=1; all other values below are pinned native defaults.
# No parameter is tuned or exposed as an override.
UPSTREAM_DEFAULTS = MappingProxyType({
    "init": "k-means++", "max_iter": 100, "verbose": 0,
    "compute_labels": True, "tol": 0.0, "max_no_improvement": 10,
    "init_size": None, "n_init": 1, "reassignment_ratio": 0.01,
})


def _load_native():
    """Fail closed before constructing a model; no installation or network I/O."""
    try:
        version = importlib.metadata.version("scikit-learn")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError("Research adapter requires scikit-learn==1.6.1") from exc
    if version != SKLEARN_VERSION:
        raise RuntimeError(f"Expected scikit-learn=={SKLEARN_VERSION}; found {version}")
    if importlib.import_module("sklearn").__version__ != SKLEARN_VERSION:
        raise RuntimeError("Loaded sklearn does not match scikit-learn==1.6.1")
    cls = importlib.import_module("sklearn.cluster").MiniBatchKMeans
    source = inspect.getsourcefile(cls)
    if source is None or hashlib.sha256(Path(source).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("MiniBatchKMeans source mismatch")
    return cls


def _state_json(value):
    """Encode every native attribute, with exact array bytes and full RNG state."""
    if isinstance(value, np.ndarray):
        return {"array_dtype": value.dtype.str, "shape": value.shape,
                "bytes": value.tobytes(order="C").hex()}
    if isinstance(value, np.random.RandomState):
        return {"RandomState": value.get_state()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported fingerprint state: {type(value).__name__}")


class MiniBatchStreamBackend:
    """Unsupervised chunk learner; caller must declare K's non-test provenance.

    k_source must be a nonblank string; its factual provenance cannot be checked
    automatically. state_revision counts successful arrivals, center_revision
    counts rows in successful complete chunks, and completed_chunks counts native
    updates. Validation errors are atomic. Runtime learning failures, including
    interrupts, poison the instance without rollback; discard it, never retry.
    Metadata and fingerprints remain available for diagnosis after failure.
    """

    def __init__(self, *, n_clusters, k_source, chunk_size=256, seed=0):
        for name, value in (("n_clusters", n_clusters), ("chunk_size", chunk_size)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(name + " must be a positive integer")
        if chunk_size < n_clusters:
            raise ValueError("chunk_size must be >= n_clusters")
        if (isinstance(seed, bool) or not isinstance(seed, Integral)
                or not 0 <= seed <= 2**32 - 1):
            raise ValueError("seed must be an integer in [0, 2**32 - 1]")
        if not isinstance(k_source, str) or not k_source.strip():
            raise ValueError("k_source must explicitly describe the source of K")
        self._parameters = dict(n_clusters=int(n_clusters), chunk_size=int(chunk_size),
                                seed=int(seed))
        self._native_parameters = dict(UPSTREAM_DEFAULTS, n_clusters=int(n_clusters),
                                       batch_size=int(chunk_size), random_state=int(seed))
        self._k_source = k_source
        self._model = _load_native()(**self._native_parameters)
        self._dimension = None
        self._pending = []
        self._state_revision = 0
        self._completed_chunks = 0
        self._failure = None
        self._model_id = uuid.uuid4().hex

    @property
    def state_revision(self):
        return self._state_revision

    def _require_usable(self):
        if self._failure is not None:
            raise RuntimeError("Backend is unusable after learning failure; discard this instance")

    def _check_dimension(self, dimension):
        if dimension < 1 or (self._dimension is not None and dimension != self._dimension):
            raise ValueError("Expected nonempty fixed-dimensional vectors")

    def learn_one(self, vector):
        self._require_usable()
        row = _numeric_array(vector)
        if row.ndim != 1:
            raise ValueError("learn_one requires one 1D vector")
        self._check_dimension(len(row))
        row = _normalize_rows(row[None, :])[0]
        # No mutation before all validation succeeds. Fail closed once mutation
        # starts: native partial_fit can modify centres/RNG before raising.
        try:
            self._pending.append(row)
            if len(self._pending) == self._parameters["chunk_size"]:
                self._model.partial_fit(np.stack(self._pending))
                self._pending.clear()
                self._completed_chunks += 1
            self._dimension = len(row)
            self._state_revision += 1
        except BaseException as exc:
            self._failure = type(exc).__name__
            raise
        return self

    @property
    def metadata(self):
        size = self._parameters["chunk_size"]
        guard = ("learning_failed" if self._failure is not None else
                 "no_completed_chunk" if not self._completed_chunks else None)
        return {
            "protocol": BACKEND_PROTOCOL, "backend": "sklearn.MiniBatchKMeans",
            "sklearn_version": SKLEARN_VERSION, "source_sha256": SOURCE_SHA256,
            "parameters": dict(self._parameters),
            "native_parameters": dict(self._native_parameters), "k_source": self._k_source,
            "model_id": self._model_id, "state_revision": self.state_revision,
            "snapshot_id": f"{self._model_id}:{self.state_revision}",
            "dimension": self._dimension, "completed_chunks": self._completed_chunks,
            "center_revision": self._completed_chunks * size,
            "pending_rows": len(self._pending), "stored_chunk_rows": len(self._pending),
            "usable": self._failure is None, "learning_failure": self._failure,
            "n_macro_clusters": self._parameters["n_clusters"] if guard is None else 0,
            "empty_state_guard": guard,
            "tail_policy": "retain pending rows; no implicit flush",
            "label_scope": "snapshot; no semantic ID alignment",
            "prediction_semantics": "native nearest centre; no novelty rejection after first chunk",
            "normalization": "per-vector L2 float64; finite nonzero fixed dimension",
            "unassigned_label": -1,
        }

    def state_fingerprint(self):
        # __getstate__ retains ALL native attributes, not a selected centre-only
        # projection. JSON encoding is canonical across trusted pickle restores.
        state = {key: value for key, value in self.__dict__.items()
                 if key not in ("_model", "_model_id")}
        state["model"] = self._model.__getstate__()
        state["protocol"] = BACKEND_PROTOCOL
        state["source_sha256"] = SOURCE_SHA256
        return hashlib.sha256(json.dumps(state, default=_state_json, sort_keys=True,
                                        separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def predict_many(self, vectors):
        """Detached, frozen prediction; reads never establish a training dimension.

        [] is an empty window; shaped empty matrices must still have a positive,
        compatible dimension. Every row is validated even before initialization.
        """
        self._require_usable()
        rows = _numeric_array(vectors)
        if rows.ndim == 1 and rows.size == 0:
            rows = rows.reshape(0, self._dimension or 0)
        else:
            if rows.ndim != 2:
                raise ValueError("predict_many requires a 2D window of vectors")
            self._check_dimension(rows.shape[1])
        rows = _normalize_rows(rows)
        metadata = self.metadata
        if not self._completed_chunks or not len(rows):
            labels = np.full(len(rows), -1, dtype=np.int64)
        else:
            snapshot = deepcopy(self._model)
            labels = np.array(snapshot.predict(rows), dtype=np.int64, copy=True)
        labels.setflags(write=False)
        return StreamPrediction(labels=labels, metadata=metadata)

    predict_window = predict_many
