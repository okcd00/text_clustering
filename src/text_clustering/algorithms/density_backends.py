"""Fixed, label-blind sklearn density baselines (not paper reproductions).

Import directly from this module; the shared factory intentionally remains
unchanged. Callers own normalization, dataset identity, costs and evaluation.
"""
from dataclasses import dataclass
import hashlib
from numbers import Integral
import platform
from types import MappingProxyType

import numpy as np
import scipy
from scipy.sparse import issparse
import sklearn
from sklearn.cluster import HDBSCAN

from ..types import Partition


DEFAULT_PRESET = "eom-mcs5-ms5-v1"
# Freeze every sklearn constructor parameter, including execution parameters.
# New settings must get a new preset name rather than changing a recorded v1.
_COMMON_PARAMS = {
    "cluster_selection_epsilon": 0.0,
    "max_cluster_size": None,
    "metric": "euclidean",
    "metric_params": None,
    "alpha": 1.0,
    "algorithm": "kd_tree",
    "leaf_size": 40,
    "n_jobs": 1,
    "cluster_selection_method": "eom",
    "allow_single_cluster": False,
    "store_centers": None,
    "copy": True,
}
HDBSCAN_PRESETS = MappingProxyType({
    name: MappingProxyType(dict(_COMMON_PARAMS, min_cluster_size=size, min_samples=samples))
    for name, size, samples in (
        (DEFAULT_PRESET, 5, 5),
        ("eom-mcs15-ms5-v1", 15, 5),
        ("eom-mcs30-ms10-v1", 30, 10),
    )
})


@dataclass(frozen=True)
class HDBSCANBackend:
    """Full-data Euclidean HDBSCAN with raw integer noise label -1.

    ``preset`` is selected before gold evaluation, never by best gold score.
    A preregistered multi-preset batch reports every setting, not just a
    gold-selected winner. No gold, K, seed or arbitrary estimator overrides
    are accepted. This API cannot police downstream selection of results.

    ``max_samples`` is an execution guard, not a clustering hyperparameter or
    an RSS/time guarantee. Raising it requires a separately reviewed budget.
    Dense finite real feature matrices only; sparse/precomputed distances and
    NaN/Inf inputs are excluded. Input values are copied to float64 without
    normalization or dimensionality reduction; caller data is never mutated.
    """

    preset: str = DEFAULT_PRESET
    max_samples: int = 5000

    def __post_init__(self):
        if not isinstance(self.preset, str) or self.preset not in HDBSCAN_PRESETS:
            raise ValueError(f"unknown HDBSCAN preset; choose from {tuple(HDBSCAN_PRESETS)}")
        if (isinstance(self.max_samples, bool)
                or not isinstance(self.max_samples, Integral)
                or self.max_samples < 2):
            raise ValueError("max_samples must be an integer >= 2")

    def fit(self, vectors: np.ndarray) -> Partition:
        """Return assignments plus JSON-safe provenance; do not use gold here."""
        if issparse(vectors):
            raise ValueError("expected a dense real feature matrix, not sparse input")
        raw = np.asarray(vectors)
        if raw.ndim != 2 or raw.shape[1] == 0 or raw.dtype.kind not in "iuf":
            raise ValueError("expected a dense real matrix [n, d], d > 0")
        n, d = raw.shape
        if n > self.max_samples:
            raise ValueError(f"n={n} exceeds max_samples={self.max_samples}; review resource budget")
        params = dict(HDBSCAN_PRESETS[self.preset])
        minimum = max(params["min_cluster_size"], params["min_samples"])
        if n < minimum:
            raise ValueError(f"preset {self.preset} requires at least {minimum} samples; got {n}")
        # sklearn 1.6 uses float64 internally. A private, C-contiguous copy also
        # protects read-only arrays/memmaps and makes the fit-input hash explicit.
        with np.errstate(over="ignore", invalid="ignore"):
            x = np.array(raw, dtype=np.float64, order="C", copy=True)
        if not np.isfinite(x).all():
            # sklearn emits -2/-3 for Inf/NaN; partition-v2 only rejects -1.
            raise ValueError("vectors must remain finite in float64; NaN/Inf are not noise")
        vector_sha256 = hashlib.sha256(memoryview(x).cast("B")).hexdigest()
        model = HDBSCAN(**params)
        actual_params = model.get_params(deep=False)
        if actual_params != params:
            raise RuntimeError("sklearn HDBSCAN parameter schema changed; review and version presets")
        labels = model.fit_predict(x)
        if labels.shape != (n,) or labels.dtype.kind not in "iu" or np.any(labels < -1):
            raise RuntimeError("unexpected sklearn labels; only integer cluster IDs and -1 are supported")
        return Partition(labels, {
            "algorithm": "HDBSCAN",
            "implementation": "sklearn.cluster.HDBSCAN",
            "backend_protocol": "density-sklearn-hdbscan-v1",
            "preset": self.preset,
            "params": actual_params,
            "versions": {
                "scikit-learn": sklearn.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "python": platform.python_version(),
            },
            "min_samples_includes_self": True,
            "noise_label": -1,
            "noise_policy": "retain; no reassignment or cluster-count correction",
            "parameter_policy": "fixed preset; gold-based selection prohibited",
            "seed": None,
            "input": {
                "shape": [n, d],
                "source_dtype": raw.dtype.str,
                "fit_dtype": x.dtype.str,
                "fit_vectors_sha256": vector_sha256,
                "hash_scope": "C-order fit-input bytes before fit; shape/dtype recorded separately",
                "fit_vector_bytes": int(x.nbytes),
                "normalization": "none; caller owns preprocessing",
            },
            "resource_guard": {"max_samples": int(self.max_samples)},
        })
