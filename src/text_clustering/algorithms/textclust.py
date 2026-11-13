"""Pinned River 0.26.1 sparse-TF micro adapter (copy-native-v1).

Learning is native, including empty observations, cleanup and merge ordering.
Prediction uses one disposable deepcopy of the *unmaintained* boundary state
per query. Neither live nor either boundary snapshot is used for prediction.
No macro API, dense-vector conversion, gold input, or additional rejection rule.
This single-writer adapter is not thread-safe. Native failures propagate; a
failed learning call is not rolled back or retried. River is imported lazily.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
from numbers import Integral, Real
from pathlib import Path
import sys
import time
from types import MappingProxyType
import uuid

import numpy as np

RIVER_VERSION = "0.26.1"
SOURCE_SHA256 = "9df72ecd3f836af0e7793e98bd75bb7dc9997934fc93f958f38b3ebdcf043b13"
TOKENIZER_SOURCE_SHA256 = "3097987bebc1e97a5b720b0705824d09d1bd157bd9964e11b2228400ea0640f1"
BACKEND_PROTOCOL = "river_textclust_token_micro_snapshot_v1"
ADAPTER = "copy-native-v1"
TF_ENCODING = "ordered-typed-tf-json-v1"
FINGERPRINT_ENCODING = "ordered-typed-state-floathex-v1"
UPSTREAM_PARAMETERS = MappingProxyType({
    "radius": 0.3, "fading_factor": 0.0005, "tgap": 100,
    "term_fading": True, "real_time_fading": False,
    "micro_distance": "tfidf_cosine_distance",
    "macro_distance": "tfidf_cosine_distance", "num_macro": 3,
    "min_weight": 0, "auto_r": False, "auto_merge": True, "sigma": 1,
})
TOKENIZER_SPEC = MappingProxyType({
    "on": None, "strip_accents": True, "lowercase": True,
    "preprocessor": None, "stop_words": None, "tokenizer": None,
    "tokenizer_pattern": r"(?u)\b\w[\w\-]+\b", "ngram_range": (1, 2),
})


def _load_pinned(module, name, expected_hash):
    if sys.version_info < (3, 11):
        raise ImportError("TextClust requires Python >=3.11 and river==0.26.1")
    try:
        version = importlib.metadata.version("river")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError("Optional dependency river==0.26.1 is required") from exc
    if version != RIVER_VERSION:
        raise RuntimeError(f"Expected river=={RIVER_VERSION}; found {version}")
    cls = getattr(importlib.import_module(module), name)
    source = inspect.getsourcefile(cls)
    if source is None or hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError(f"Pinned {module} source SHA-256 mismatch")
    return cls


def build_tokenizer():
    """Construct the audited, fixed, stateless unigram/bigram BagOfWords."""
    return _load_pinned("river.feature_extraction.vectorize", "BagOfWords",
                        TOKENIZER_SOURCE_SHA256)(**dict(TOKENIZER_SPEC))


def validate_tf(tf):
    """Return a fresh ordered sparse mapping; omit zeroes, reject invalid data.

    Keys must be nonempty strings or pairs of nonempty strings. No stringifying,
    joining, sorting, vocabulary fitting, normalization, or dense input support.
    """
    if not isinstance(tf, Mapping):
        raise ValueError("TF must be a sparse term-frequency mapping, not dense vectors")
    result = {}
    for term, value in tf.items():
        if not ((isinstance(term, str) and term)
                or (isinstance(term, tuple) and len(term) == 2
                    and all(isinstance(t, str) and t for t in term))):
            raise ValueError("TF keys must be unigram strings or bigram string tuples")
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(value) or value < 0):
            raise ValueError("TF counts must be finite nonnegative real numbers")
        if value > 0:
            result[term] = int(value) if isinstance(value, Integral) else float(value)
    return result


def tokenize_tf(text):
    """Tokenize one raw string using the fixed audited spec, without fitting.

    The workflow builds one tokenizer per job to avoid repeated construction.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    return validate_tf(build_tokenizer().transform_one(text))


def encode_tf(tf):
    """JSON-safe, ordered typed-key evidence; string and tuple keys never collide."""
    return [{"kind": "unigram" if isinstance(k, str) else "bigram",
             "tokens": [k] if isinstance(k, str) else list(k), "count": v}
            for k, v in validate_tf(tf).items()]


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def tf_fingerprint(tf):
    """Hash compact UTF-8 JSON with sorted object fields, preserving term order."""
    return hashlib.sha256(_json_bytes(encode_tf(tf))).hexdigest()


def _state_value(value):
    """Inspect only stored fields, never estimator properties or methods.

    Dict order is semantic (native tie breaking); floats use exact hex. Inspect
    all instance fields, including caches, rather than selecting learned fields.
    Unknown types fail closed instead of hashing repr/memory addresses.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return ["int", str(int(value))]
    if isinstance(value, Real):
        return ["float", float(value).hex()]
    if isinstance(value, dict):
        return ["dict", [[_state_value(k), _state_value(v)] for k, v in value.items()]]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_state_value(v) for v in value]]
    if type(value).__module__ == "river.cluster.textclust":
        return [type(value).__qualname__, _state_value(vars(value))]
    raise TypeError(f"Unsupported TextClust state field type: {type(value).__name__}")


def fingerprint_textclust(model):
    """Pure native semantic fingerprint, including parameters and ordered state."""
    return hashlib.sha256(_json_bytes(_state_value(model))).hexdigest()


def _summary(model):
    clusters = model.micro_clusters
    return {
        "micro_ids": list(clusters), "n_micro_clusters": len(clusters),
        "n_micro_eligible": sum(c.weight > model.min_weight for c in clusters.values()),
        "sum_tf_entries": sum(len(c.tf) for c in clusters.values()),
        "vocabulary_size": len({term for c in clusters.values() for term in c.tf}),
        "merged_ids_entries": sum(len(c.merged_ids) for c in clusters.values()),
        "t": model.t, "n": model.n, "last_cleanup": model.last_cleanup,
        "next_cluster_id": model._cluster_id,
    }


@dataclass(frozen=True)
class TextClustPrediction:
    """Detached JSON-safe metadata and read-only int64 labels."""
    labels: np.ndarray
    metadata: dict

    @property
    def state_revision(self):
        return self.metadata["state_revision"]


class TextClustBackend:
    """Native learn-only live state; radius is the sole configurable parameter.

    ``metadata`` makes diagnostic copies and accounts for their boundary cost.
    ``live_summary`` and ``state_fingerprint`` are pure reads. Neither modifies
    native caches. Snapshot labels are not stable semantic topic identities.
    """
    def __init__(self, radius=0.3):
        if (isinstance(radius, bool) or not isinstance(radius, Real)
                or not math.isfinite(radius) or not 0 < radius <= 1):
            raise ValueError("radius must be finite and in (0, 1]")
        parameters = dict(UPSTREAM_PARAMETERS, radius=float(radius))
        self._model = _load_pinned("river.cluster.textclust", "TextClust",
                                  SOURCE_SHA256)(**parameters)
        self._state_revision = 0
        self._model_id = uuid.uuid4().hex

    @property
    def state_revision(self):
        return self._state_revision

    def state_fingerprint(self):
        # UUID deliberately excluded so independently warmed branches compare.
        return hashlib.sha256(_json_bytes([
            FINGERPRINT_ENCODING, self._state_revision,
            fingerprint_textclust(self._model),
        ])).hexdigest()

    def live_summary(self):
        return dict(_summary(self._model), state_revision=self.state_revision)

    def learn_one(self, tf):
        tf = validate_tf(tf)  # validation completes before native mutation
        self._model.learn_one(tf)
        self._state_revision += 1  # includes an empty TF observation
        return self

    def _make_window_snapshot(self):
        start = time.perf_counter()
        if self._model.micro_clusters and (
                not isinstance(self._model.t, Real) or not math.isfinite(self._model.t)):
            raise RuntimeError("Nonempty TextClust state has no valid learning clock")
        live_sha = self.state_fingerprint()
        stamp = time.perf_counter()
        base = deepcopy(self._model)
        base_copy_seconds = time.perf_counter() - stamp
        base_sha = fingerprint_textclust(base)
        stamp = time.perf_counter()
        effective = deepcopy(base)
        effective_copy_seconds = time.perf_counter() - stamp
        stamp = time.perf_counter()
        effective._updateweights()
        maintenance_seconds = time.perf_counter() - stamp
        effective_sha = fingerprint_textclust(effective)
        idf = effective._calculate_idf(effective.micro_clusters.values())
        live, view = _summary(base), _summary(effective)
        metadata = {
            "backend": "river.TextClust", "protocol": BACKEND_PROTOCOL,
            "adapter": ADAPTER, "river_version": RIVER_VERSION,
            "source_sha256": SOURCE_SHA256,
            "fingerprint_encoding": FINGERPRINT_ENCODING,
            "parameters": {k: getattr(base, k) for k in UPSTREAM_PARAMETERS},
            "representation": "sparse-term-frequency-unigram-bigram",
            "output_granularity": "micro", "unassigned_label": -1,
            "model_id": self._model_id, "state_revision": self.state_revision,
            "snapshot_id": f"{self._model_id}:{self.state_revision}",
            "label_scope": "snapshot; no persistent identity alignment",
            "prediction_semantics": "native nearest eligible micro; None only maps to -1",
            "tie_breaking": "first eligible micro in native dictionary insertion order",
            "live_state_sha256": live_sha, "base_state_sha256": base_sha,
            "effective_state_sha256": effective_sha,
            "live": dict(live, state_revision=self.state_revision),
            "effective": dict(view, state_revision=self.state_revision),
            "n_micro_live": live["n_micro_clusters"],
            "n_micro_effective": view["n_micro_clusters"],
            "n_micro_eligible": view["n_micro_eligible"],
            "snapshot_pruned_ids": [k for k in base.micro_clusters
                                    if k not in effective.micro_clusters],
            "merged_events": None, "expired_events": None,
            "event_counts_unavailable_reason": "native events are not instrumented; endpoint differences cannot attribute events",
        }
        if self.state_fingerprint() != live_sha or fingerprint_textclust(base) != base_sha:
            raise RuntimeError("Snapshot construction mutated protected state")
        metadata["costs"] = {
            "boundary_seconds": time.perf_counter() - start,
            "base_copy_seconds": base_copy_seconds,
            "effective_copy_seconds": effective_copy_seconds,
            "effective_maintenance_seconds": maintenance_seconds,
            "boundary_subcost_parent": "boundary_seconds",
        }
        return base, effective, idf, metadata

    @property
    def metadata(self):
        start = time.perf_counter()
        base, effective, idf, metadata = self._make_window_snapshot()
        del base, effective, idf
        metadata["costs"]["diagnostic_including_release_seconds"] = time.perf_counter() - start
        return metadata

    def predict_window(self, tfs):
        """Validate all TFs, then predict once per independent native deepcopy.

        Empty windows are valid. Query-copy, native prediction, fingerprint and
        release timings are children of per-query time, never additive stages.
        The effective view is diagnostic only; it is NOT the query-copy source.
        """
        if isinstance(tfs, (str, bytes, Mapping)):
            raise ValueError("predict_window requires an iterable of sparse TF mappings")
        rows = [validate_tf(tf) for tf in tfs]
        start = time.perf_counter()
        base, effective, idf, metadata = self._make_window_snapshot()
        labels, diagnostics, query_costs = [], [], []
        for tf in rows:
            query_start = time.perf_counter()
            stamp = time.perf_counter()
            query = deepcopy(base)
            copy_seconds = time.perf_counter() - stamp
            stamp = time.perf_counter()
            native_id = query.predict_one(tf, type="micro")
            native_seconds = time.perf_counter() - stamp
            stamp = time.perf_counter()
            if fingerprint_textclust(query) != metadata["effective_state_sha256"]:
                raise RuntimeError("Native query state differs from effective diagnostic view")
            fingerprint_seconds = time.perf_counter() - stamp
            stamp = time.perf_counter()
            del query
            release_seconds = time.perf_counter() - stamp
            empty = not tf
            no_effective = not effective.micro_clusters
            no_eligible = metadata["n_micro_eligible"] == 0
            reason = ("empty_tf" if empty else "no_effective_micro" if no_effective
                      else "no_eligible_micro" if no_eligible else "assigned")
            if native_id is None and reason == "assigned":
                raise RuntimeError("unexpected_none: native micro prediction has eligible state")
            labels.append(-1 if native_id is None else int(native_id))
            diagnostics.append({
                "empty_tf": empty, "no_live_micro": not base.micro_clusters,
                "no_effective_micro": no_effective, "no_eligible_micro": no_eligible,
                "all_oov": bool(tf) and all(k not in idf for k in tf), "reason": reason,
            })
            query_costs.append({
                "query_copy_seconds": copy_seconds, "native_predict_seconds": native_seconds,
                "query_fingerprint_seconds": fingerprint_seconds,
                "query_release_seconds": release_seconds,
                "query_seconds": time.perf_counter() - query_start,
            })
        stamp = time.perf_counter()
        if (self.state_fingerprint() != metadata["live_state_sha256"]
                or fingerprint_textclust(base) != metadata["base_state_sha256"]
                or fingerprint_textclust(effective) != metadata["effective_state_sha256"]):
            raise RuntimeError("Prediction mutated protected live/base/effective state")
        protected_check_seconds = time.perf_counter() - stamp
        del base, effective, idf, rows
        metadata["diagnostics"] = diagnostics
        metadata["n_predicted_clusters_window"] = len(set(labels) - {-1})
        metadata["costs"].update({
            "queries": query_costs, "query_subcost_parent": "query_seconds",
            "protected_check_seconds": protected_check_seconds,
            "prediction_including_boundary_seconds": time.perf_counter() - start,
        })
        array = np.array(labels, dtype=np.int64)
        array.setflags(write=False)
        return TextClustPrediction(array, metadata)
