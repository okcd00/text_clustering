"""Independent experimental BIRCH CF-leaf stream; not factory/admission wiring.

Audited sklearn 1.6.1, one normalized row per partial_fit, no global clustering.
Single writer only. Prediction copies only the audited native prediction
arrays, never the recursive CF tree. Fingerprints describe logical state, not portable checkpoints, memory
alias layouts, or dependency/compiled-code integrity. Backend copying and
serialization are rejected: native CF array views are not a safe continuation
checkpoint. Native learning failures poison the instance; no rollback or retry.
"""
import ctypes
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from numbers import Integral, Real
from pathlib import Path
import uuid
import time

import numpy as np

from .river_stream_backend import StreamPrediction, _numeric_array, _normalize_rows

SKLEARN_VERSION = "1.6.1"
SOURCE_SHA256 = "5ef16cc801c88042ce635956a8eba988bc96b6362d6d76838a4d211a6380809a"
BACKEND_PROTOCOL = "sklearn-birch-cf-leaf-l2-snapshot-v1"


def _load_native():
    try:
        version = importlib.metadata.version("scikit-learn")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError("Requires scikit-learn==1.6.1; no automatic installation") from exc
    if version != SKLEARN_VERSION:
        raise RuntimeError(f"Expected scikit-learn=={SKLEARN_VERSION}; found {version}")
    if importlib.import_module("sklearn").__version__ != SKLEARN_VERSION:
        raise RuntimeError("Loaded sklearn version mismatch")
    cls = importlib.import_module("sklearn.cluster").Birch
    source = inspect.getsourcefile(cls)
    if source is None or hashlib.sha256(Path(source).read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Birch source mismatch")
    return cls


def _graph_state(value):
    """Canonical traversal of all native attributes, including cyclic leaf links.

    Stable traversal IDs replace process addresses. Arrays retain dtype, shape
    and exact logical bytes, including allocated CF capacity. Unsupported state
    fails closed rather than silently falling back to repr or omitting fields.
    """
    seen, pending, nodes = {}, [], []

    def reference(obj):
        if obj is None or isinstance(obj, (str, bool, int)):
            return obj
        if isinstance(obj, float):
            return {"float": obj.hex()}
        if isinstance(obj, np.generic):
            return {"scalar": obj.dtype.str, "bytes": obj.tobytes().hex()}
        identity = id(obj)
        if identity not in seen:
            seen[identity] = len(pending)
            pending.append(obj)
        return {"ref": seen[identity]}

    root = reference(value)
    # Flat graph avoids Python/JSON recursion limits on long CF leaf chains.
    index = 0
    while index < len(pending):
        obj = pending[index]
        if isinstance(obj, np.ndarray):
            if obj.dtype.hasobject:
                raise TypeError("Object arrays are not fingerprintable")
            payload = {"dtype": obj.dtype.str, "shape": obj.shape,
                       "bytes": obj.tobytes(order="C").hex()}
        elif isinstance(obj, dict):
            if not all(isinstance(key, str) for key in obj):
                raise TypeError("Fingerprint dictionaries require string keys")
            payload = {key: reference(obj[key]) for key in sorted(obj)}
        elif isinstance(obj, (list, tuple)):
            payload = [reference(item) for item in obj]
        elif (type(obj).__module__ == "sklearn.cluster._birch"
              and type(obj).__name__ in ("Birch", "_CFNode", "_CFSubcluster")):
            payload = reference(vars(obj))
        else:
            raise TypeError(f"Unsupported fingerprint state: {type(obj).__name__}")
        nodes.append({"type": type(obj).__module__ + "." + type(obj).__name__,
                      "value": payload})
        index += 1
    return {"root": root, "nodes": nodes}


_FINGERPRINT_CHUNK_BYTES = 64 * 1024


def _array_byte_chunks(array):
    """Yield exact C-order bytes, preserving padding and arbitrary strides.

    Raw spans avoid whole-array copies and dtype conversions (which could lose
    structured padding or NaN bits). The array stays alive throughout this
    single-writer operation, including for negative and broadcast strides.
    """
    if not array.size or not array.itemsize:
        return
    address = array.__array_interface__["data"][0]

    def span(start, length):
        for offset in range(0, length, _FINGERPRINT_CHUNK_BYTES):
            yield ctypes.string_at(start + offset,
                                   min(_FINGERPRINT_CHUNK_BYTES, length - offset))

    if array.flags.c_contiguous:
        yield from span(address, array.nbytes)
    else:
        # Coalesce the longest contiguous suffix in logical C order.
        split, length = array.ndim, array.itemsize
        while split and (array.shape[split - 1] == 1
                         or array.strides[split - 1] == length):
            split -= 1
            length *= array.shape[split]
        for index in np.ndindex(array.shape[:split]):
            start = address + sum(i * stride for i, stride
                                  in zip(index, array.strides))
            yield from span(start, length)


def _graph_fingerprint(value):
    """Stream precisely the legacy sorted JSON graph into SHA-256.

    Storage is O(graph identities + largest dictionary key sort), plus bounded
    byte/hex chunks, rather than O(total CF array bytes). Reference assignment
    is still breadth-first although sorted JSON emits nodes before root.
    Pending retains objects, including attribute dictionaries, against ID reuse.
    """
    digest = hashlib.sha256()
    seen, pending = {}, []

    def emit(text):
        digest.update(text.encode())

    def json_value(obj):
        emit(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False))

    def reference(obj):
        if obj is None or isinstance(obj, (str, bool, int)):
            return obj
        if isinstance(obj, float):
            return {"float": obj.hex()}
        if isinstance(obj, np.generic):
            return {"scalar": obj.dtype.str, "bytes": obj.tobytes().hex()}
        identity = id(obj)
        if identity not in seen:
            seen[identity] = len(pending)
            pending.append(obj)
        return {"ref": seen[identity]}

    root = reference(value)
    emit('{"nodes":[')
    index = 0
    while index < len(pending):
        obj = pending[index]
        if index:
            emit(',')
        emit('{"type":')
        json_value(type(obj).__module__ + "." + type(obj).__name__)
        emit(',"value":')
        if isinstance(obj, np.ndarray):
            if obj.dtype.hasobject:
                raise TypeError("Object arrays are not fingerprintable")
            emit('{"bytes":"')
            for chunk in _array_byte_chunks(obj):
                emit(chunk.hex())
            emit('","dtype":')
            json_value(obj.dtype.str)
            emit(',"shape":')
            json_value(obj.shape)
            emit('}')
        elif isinstance(obj, dict):
            if not all(isinstance(key, str) for key in obj):
                raise TypeError("Fingerprint dictionaries require string keys")
            emit('{')
            for position, key in enumerate(sorted(obj)):
                if position:
                    emit(',')
                json_value(key)
                emit(':')
                json_value(reference(obj[key]))
            emit('}')
        elif isinstance(obj, (list, tuple)):
            emit('[')
            for position, item in enumerate(obj):
                if position:
                    emit(',')
                json_value(reference(item))
            emit(']')
        elif (type(obj).__module__ == "sklearn.cluster._birch"
              and type(obj).__name__ in ("Birch", "_CFNode", "_CFSubcluster")):
            json_value(reference(vars(obj)))
        else:
            raise TypeError(f"Unsupported fingerprint state: {type(obj).__name__}")
        emit('}')
        index += 1
    emit('],"root":')
    json_value(root)
    emit('}')
    return digest.hexdigest()


class BirchStreamBackend:
    """Unknown-K, snapshot-local CF leaf labels, with no novelty rejection.

    threshold and branching_factor default to native defaults, not tuned values.
    n_clusters must remain None. A fresh instance is required per protocol arm;
    continuation is allowed only within the same stream/dimension/configuration.
    Validation failures do not change state. Successful learn_one calls count
    towards state_revision, including arrivals in the final protocol window.
    """

    def __init__(self, *, threshold=0.5, branching_factor=50, n_clusters=None):
        if (isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real)
                or not np.isfinite(threshold) or threshold <= 0):
            raise ValueError("threshold must be finite and positive")
        if (isinstance(branching_factor, (bool, np.bool_))
                or not isinstance(branching_factor, Integral) or branching_factor < 2):
            raise ValueError("branching_factor must be an integer >= 2")
        if n_clusters is not None:
            raise ValueError("n_clusters must be None (unknown-K CF leaves)")
        self._model = _load_native()(threshold=float(threshold),
                                    branching_factor=int(branching_factor), n_clusters=None)
        self._parameters = self._model.get_params(deep=False)
        self._versions = {name: importlib.metadata.version(name)
                          for name in ("scikit-learn", "numpy", "scipy")}
        self._dimension = None
        self._state_revision = 0
        self._failure = None
        self._model_id = uuid.uuid4().hex
        # Observability only: never part of learned-state identity.
        self._timings = dict(native_fit_seconds=0.0, native_predict_seconds=0.0,
                             snapshot_seconds=0.0, fingerprint_seconds=0.0)

    @property
    def timings(self):
        return dict(self._timings)

    def __copy__(self):
        raise TypeError("Backend copying is unsupported; create a fresh instance")

    def __deepcopy__(self, memo):
        raise TypeError("Backend copying is unsupported; create a fresh instance")

    def __reduce_ex__(self, protocol):
        raise TypeError("Backend checkpoints are unsupported; create a fresh instance")

    @property
    def state_revision(self):
        return self._state_revision

    def _require_usable(self):
        if self._failure is not None:
            raise RuntimeError("Backend unusable after learning failure; discard instance")
        if self._model.get_params(deep=False) != self._parameters:
            raise RuntimeError("Native parameters changed; discard instance")

    def _check_dimension(self, dimension):
        if dimension < 1 or (self._dimension is not None and dimension != self._dimension):
            raise ValueError("Expected nonempty fixed-dimensional vectors")

    def learn_one(self, vector):
        self._require_usable()
        row = _numeric_array(vector)
        if row.ndim != 1:
            raise ValueError("learn_one requires one 1D vector")
        self._check_dimension(len(row))
        row = _normalize_rows(row[None, :])
        try:
            started = time.perf_counter()
            self._model.partial_fit(row)
            self._timings['native_fit_seconds'] += time.perf_counter() - started
            self._dimension = row.shape[1]
            self._state_revision += 1
        except BaseException as exc:
            self._failure = type(exc).__name__
            raise
        return self

    @property
    def metadata(self):
        count = len(getattr(self._model, "subcluster_centers_", ()))
        return {"backend": "sklearn.Birch", "protocol": BACKEND_PROTOCOL,
                "status": "experimental candidate; not admitted",
                "sklearn_version": SKLEARN_VERSION, "source_sha256": SOURCE_SHA256,
                "dependency_versions": dict(self._versions),
                "parameters": dict(self._parameters),
                "model_id": self._model_id, "state_revision": self.state_revision,
                "snapshot_id": f"{self._model_id}:{self.state_revision}",
                "dimension": self._dimension, "n_micro_clusters": count,
                "n_output_clusters": count, "global_clustering": False,
                "label_scope": "snapshot-local CF leaf subclusters; no identity alignment",
                "normalization": "per-vector L2 float64; finite nonzero fixed dimension",
                "prediction_semantics": "native nearest CF leaf centroid; no novelty rejection",
                "unassigned_label": -1,
                "empty_state_guard": "unlearned" if self.state_revision == 0 else None,
                "usable": self._failure is None and self._model.get_params(deep=False) == self._parameters,
                "learning_failure": self._failure}

    def state_fingerprint(self):
        started = time.perf_counter()
        state = {key: value for key, value in vars(self).items()
                 if key not in ("_model_id", "_timings")}
        state.update(protocol=BACKEND_PROTOCOL, source_sha256=SOURCE_SHA256)
        result = _graph_fingerprint(state)
        self._timings['fingerprint_seconds'] += time.perf_counter() - started
        return result

    def predict_window(self, vectors):
        """Validate all rows, then predict on a detached copy; never bind dimension.

        [] is an empty window. Explicit empty matrices still require a positive,
        compatible dimension. Empty learned state assigns -1 to valid queries.
        """
        self._require_usable()
        rows = _numeric_array(vectors)
        if rows.ndim == 1 and rows.size == 0:
            rows = rows.reshape(0, self._dimension or 0)
        else:
            if rows.ndim != 2:
                raise ValueError("predict_window requires a 2D window")
            self._check_dimension(rows.shape[1])
        rows = _normalize_rows(rows)
        metadata = self.metadata
        if not self.state_revision or not len(rows):
            labels = np.full(len(rows), -1, dtype=np.int64)
        else:
            # Audited Birch.predict -> _predict only reads these fitted fields.
            # A full deepcopy follows cyclic leaf links and can hit recursion
            # limits. This ephemeral estimator is ONLY a prediction snapshot;
            # never learn on it or expose it as a continuation checkpoint.
            started = time.perf_counter()
            snapshot = type(self._model)(**self._parameters)
            snapshot.n_features_in_ = self._model.n_features_in_
            for name in ("subcluster_centers_", "subcluster_labels_", "_subcluster_norms"):
                setattr(snapshot, name, getattr(self._model, name).copy())
            self._timings['snapshot_seconds'] += time.perf_counter() - started
            started = time.perf_counter()
            labels = np.array(snapshot.predict(rows), dtype=np.int64, copy=True)
            self._timings['native_predict_seconds'] += time.perf_counter() - started
        labels.setflags(write=False)
        return StreamPrediction(labels=labels, metadata=metadata)

    predict_many = predict_window
