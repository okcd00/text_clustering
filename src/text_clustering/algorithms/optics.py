"""Bounded, label-blind sklearn OPTICS Xi flat-partition baseline.

Authority: the installed scikit-learn 1.6.1 ``sklearn/cluster/_optics.py``:
``OPTICS.fit`` validates with dtype=float, computes the OPTICS ordering, and
calls ``cluster_optics_xi``; ``_validate_size`` requires both integer size
parameters <= n; ``_extract_xi_labels`` returns flat labels with -1 noise.
The reviewed file's SHA256 is
432524afae761976e6be738113ea392034656470826eddf2b744879349bff80f.
Upstream source: https://github.com/scikit-learn/scikit-learn/blob/1.6.1/sklearn/cluster/_optics.py

The adapter passes a raw float32 copy, without normalization or reduction.
In the reviewed release sklearn subsequently converts that copy to float64.
Xi intervals in ordering space are not a complete document hierarchy, and
this backend exports only Partition. No upstream sources are modified.
"""
from dataclasses import dataclass
import hashlib
from numbers import Integral
from types import MappingProxyType
from typing import Optional

import numpy as np
from scipy.sparse import issparse
import sklearn
from sklearn.cluster import OPTICS

from ..types import Partition


__all__ = ["OPTICSXiBackend"]

# Freeze the complete 1.6.1 constructor schema, including unused defaults.
_PARAMS = MappingProxyType({
    "min_samples": 5,
    "min_cluster_size": 15,
    "xi": 0.05,
    "predecessor_correction": True,
    "max_eps": float("inf"),
    "cluster_method": "xi",
    "metric": "euclidean",
    "algorithm": "brute",
    "n_jobs": 1,
    "p": 2,
    "metric_params": None,
    "eps": None,
    "leaf_size": 30,
    "memory": None,
})


@dataclass(frozen=True)
class OPTICSXiBackend:
    """Fixed UnknownK preset; accepts neither K, labels, nor parameter tuning.

    ``seed`` is optional audit metadata, not an RNG control. Results are
    deterministic for fixed input order and numerical environment, not a
    cross-platform or permutation-invariance guarantee.

    ``max_samples`` defaults to 5000; raising it is an explicit caller budget
    decision. It bounds admitted rows, not feature width, RSS or runtime:
    sklearn's OPTICS implementation has quadratic time complexity.
    """

    seed: Optional[int] = None
    max_samples: int = 5000

    def __post_init__(self):
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, Integral)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be None or an unsigned 32-bit integer (audit only)")
        if (isinstance(self.max_samples, bool)
                or not isinstance(self.max_samples, Integral) or self.max_samples < 1):
            raise ValueError("max_samples must be a positive integer")

    def fit(self, vectors: np.ndarray) -> Partition:
        if issparse(vectors):
            raise ValueError("expected a dense real feature matrix, not sparse input")
        try:
            raw = np.asarray(vectors)
        except (TypeError, ValueError) as exc:
            raise ValueError("expected a dense real matrix [n, d], d > 0") from exc
        if raw.ndim != 2 or raw.shape[1] == 0 or raw.dtype.kind not in "iuf":
            raise ValueError("expected a dense real matrix [n, d], d > 0")
        n, d = raw.shape
        if n > self.max_samples:
            raise ValueError(f"n={n} exceeds max_samples={self.max_samples}; review resource budget")
        minimum = max(_PARAMS["min_samples"], _PARAMS["min_cluster_size"])
        if n < minimum:
            raise ValueError(f"optics_xi requires at least {minimum} samples; got {n}")
        # A private contiguous buffer protects read-only views and caller data.
        # Reject overflow on conversion as well as nonfinite source values.
        with np.errstate(over="ignore", invalid="ignore"):
            x = np.array(raw, dtype=np.float32, order="C", copy=True)
        if not np.isfinite(x).all():
            raise ValueError("vectors must remain finite in float32; NaN/Inf are not noise")
        digest = hashlib.sha256(memoryview(x).cast("B")).hexdigest()
        model = OPTICS(**dict(_PARAMS))
        actual_params = model.get_params(deep=False)
        if actual_params != dict(_PARAMS):
            raise RuntimeError("sklearn OPTICS parameter schema changed; review fixed preset")
        labels = np.asarray(model.fit_predict(x))
        if labels.shape != (n,) or labels.dtype.kind not in "iu" or np.any(labels < -1):
            raise RuntimeError("unexpected sklearn labels; integer IDs and -1 noise required")
        if model.get_params(deep=False) != actual_params:
            raise RuntimeError("sklearn OPTICS changed fixed parameters during fit")
        # Strict JSON forbids Infinity. Keep the estimator's real inf above,
        # and encode only the metadata field as an explicitly documented string.
        json_params = dict(actual_params, max_eps="inf")
        return Partition(labels.copy(), {
            "algorithm": "OPTICS",
            "algorithm_id": "optics_xi",
            "implementation": "sklearn.cluster.OPTICS",
            "backend_protocol": "optics-xi-sklearn-v1",
            "k_regime": "UnknownK",
            "params": json_params,
            "parameter_encoding": {"max_eps": "string 'inf' means positive infinity"},
            "parameter_policy": "fixed preset; no K, labels or parameter selection",
            "versions": {"scikit-learn": sklearn.__version__, "numpy": np.__version__},
            "reviewed_source": {"version": "1.6.1", "file": "sklearn/cluster/_optics.py"},
            "seed": None if self.seed is None else int(self.seed),
            "seed_used": False,
            "determinism": "deterministic for fixed input order and numerical environment; seed is audit-only",
            "min_samples_includes_self": True,
            "actual_k": int(len(np.unique(labels[labels >= 0]))),
            "noise_label": -1,
            "noise_count": int(np.count_nonzero(labels == -1)),
            "noise_policy": "retain sklearn Xi labels; no reassignment or cluster-count correction",
            "input": {
                "shape": [int(n), int(d)],
                "source_dtype": raw.dtype.str,
                "fit_dtype": x.dtype.str,
                "fit_vectors_sha256": digest,
                "hash_scope": "C-order float32 adapter input before sklearn validation; shape/dtype recorded separately",
                "fit_vector_bytes": int(x.nbytes),
                "normalization": "none in adapter; caller owns preprocessing",
            },
            "resource_guard": {"max_samples": int(self.max_samples)},
            "capabilities": {"flat_partition": True, "complete_hierarchy": False},
            "hierarchy_policy": "Xi ordering intervals are not exported as a complete document hierarchy",
        })
