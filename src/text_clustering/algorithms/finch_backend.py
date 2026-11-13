"""Optional finch-clust==0.2.3, exact cosine, fixed first-partition preset.

Reviewed release wheel SHA256:
153ff749c9bc53cbdb4a7d174db8dbc95c082a0d5014bb0650f99b4cdb0639bc.
In that release clust_rank uses exact pairwise distances for n <= ann_threshold.
The full upstream hierarchy is computed, including its normal early-exit rule;
we select column zero only after FINCH returns. No requested-K merge is run.
"""
from dataclasses import dataclass
import hashlib
import importlib.metadata
import inspect
from numbers import Integral
from pathlib import Path
import time

import numpy as np
from scipy.sparse import issparse

from ..types import Partition

__all__ = ["FINCHBackend"]
_VERSION = "0.2.3"
_MAX_SAMPLES = 5000
_WHEEL_SHA256 = "153ff749c9bc53cbdb4a7d174db8dbc95c082a0d5014bb0650f99b4cdb0639bc"


def _input(vectors, seed):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if issparse(vectors):
        raise ValueError("dense 2-D real vectors required; sparse input is not supported")
    try:
        raw = np.asarray(vectors)
    except (TypeError, ValueError) as exc:
        raise ValueError("dense 2-D real vectors required") from exc
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] == 0 or raw.dtype.kind not in "fiu":
        raise ValueError("dense 2-D real vectors with at least 2 rows and 1 feature required")
    if len(raw) > _MAX_SAMPLES:
        raise ValueError(f"n={len(raw)} exceeds hard max_samples={_MAX_SAMPLES}")
    if not np.isfinite(raw).all():
        raise ValueError("finite vectors required")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        x = np.array(raw, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(x).all() or not np.any(x != 0, axis=1).all():
        raise ValueError("each row must be finite and nonzero after float32 conversion")
    return x, dict(
        shape=list(x.shape), source_dtype=raw.dtype.str, fit_dtype=x.dtype.str,
        source_vectors_sha256=hashlib.sha256(raw.tobytes(order="C")).hexdigest(),
        fit_vectors_sha256=hashlib.sha256(x.tobytes(order="C")).hexdigest(),
        fit_vector_bytes=int(x.nbytes), identity_stage="original row order; before upstream FINCH",
    )


def _partition(output, n):
    if not isinstance(output, (tuple, list)) or len(output) != 3:
        raise RuntimeError("invalid FINCH output: expected hierarchy, counts, requested partition")
    hierarchy, counts, requested = output
    if requested is not None:
        raise RuntimeError("unexpected FINCH requested partition with req_clust=None")
    if issparse(hierarchy) or issparse(counts):
        raise RuntimeError("invalid FINCH hierarchy/counts: dense outputs required")
    try:
        hierarchy, counts = np.asarray(hierarchy), np.asarray(counts)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("invalid FINCH hierarchy/counts") from exc
    if (hierarchy.ndim != 2 or hierarchy.shape[0] != n or hierarchy.shape[1] == 0
            or hierarchy.dtype.kind not in "iu" or (hierarchy < 0).any()):
        raise RuntimeError("invalid FINCH hierarchy: nonnegative integer labels of shape (n, layers) required")
    if (counts.shape != (hierarchy.shape[1],) or counts.dtype.kind not in "iu"
            or (counts < 1).any() or (counts > n).any()):
        raise RuntimeError("invalid FINCH cluster counts")
    for layer, count in enumerate(counts):
        if not np.array_equal(np.unique(hierarchy[:, layer]), np.arange(int(count))):
            raise RuntimeError("FINCH cluster counts/contiguous label IDs disagree")
        if layer:
            if int(count) >= int(counts[layer - 1]):
                raise RuntimeError("FINCH hierarchy must reduce cluster count")
            # Every finer cluster must map to exactly one coarser cluster.
            pairs = np.unique(hierarchy[:, layer - 1:layer + 1], axis=0)
            if len(pairs) != int(counts[layer - 1]):
                raise RuntimeError("FINCH hierarchy is not nested")
    return hierarchy[:, 0].copy(), [int(count) for count in counts]


@dataclass(frozen=True)
class FINCHBackend:
    """Unknown-K preset with no gold, requested-K, or layer-selection API.

    The cap bounds sample count, not total workspace or feature width. Input is
    copied to raw float32 without external normalization or reduction. Seed is
    passed only as upstream random_state (unused on the reviewed exact path).
    """

    seed: int = 0

    def fit(self, vectors) -> Partition:
        x, identity = _input(vectors, self.seed)
        try:
            version = importlib.metadata.version("finch-clust")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ImportError("FINCH backend requires optional finch-clust==0.2.3") from exc
        if version != _VERSION:
            raise RuntimeError(f"FINCH backend requires finch-clust=={_VERSION}; found {version}")
        from finch import FINCH  # Optional dependency: imported only for a valid fit.

        params = dict(
            req_clust=None, initial_rank=None, distance="cosine", ensure_early_exit=True,
            ann_threshold=5000, faiss_threshold=5_000_000, faiss_use_gpu=False,
            ram_size_in_gb=None, faiss_kwargs=None, return_meta_labels=False,
            verbose=False, random_state=int(self.seed),
        )
        # Hash the actual implementation separately from the reviewed wheel identity.
        source_hash = hashlib.sha256(Path(inspect.getfile(FINCH)).read_bytes()).hexdigest()
        start = time.perf_counter()
        output = FINCH(x, **params)
        elapsed = time.perf_counter() - start
        labels, counts = _partition(output, len(x))
        return Partition(labels, dict(
            algorithm="FINCH", implementation="finch.FINCH",
            backend_protocol="finch-exact-cosine-first-partition-v1",
            classification="author package candidate", k_regime="UnknownK",
            versions={"finch-clust": version, "numpy": np.__version__,
                      "scipy": importlib.metadata.version("scipy"),
                      "scikit-learn": importlib.metadata.version("scikit-learn")},
            source={"reviewed_wheel_sha256": _WHEEL_SHA256,
                    "runtime_finch_source_sha256": source_hash,
                    "implementation_origin": "released package; version is not wheel authentication"},
            params=params, input=identity, max_samples=_MAX_SAMPLES,
            seed=int(self.seed), seed_used=False,
            determinism="random_state unused on exact path; ties, numerical libraries and row order may affect results",
            nearest_neighbor_backend="exact pairwise cosine at every level (n <= 5000)",
            hierarchy_layers=len(counts), hierarchy_cluster_counts=counts,
            selected_partition_index=0, selection_policy="fixed first partition; no gold or requested-K selection",
            clustering_seconds=elapsed,
            timing_scope="entire upstream FINCH call, including all computed hierarchy levels and early-exit work",
            preprocessing={"dtype": "float32", "copy": True, "normalization": "none in adapter",
                           "external_reduction": None},
            actual_k=counts[0], noise_count=0, noise_policy="FINCH returns a complete partition; no noise label",
        ))
