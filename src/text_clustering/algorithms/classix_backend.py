"""Unknown-K CLASSIX candidate using the released classixclustering==1.5.3.

Source/API checked 2026-11-04, without installing or executing the wheel:
https://pypi.org/project/classixclustering/1.5.3/
https://github.com/nla-group/classix/tree/70344c972d32f2bcaae6655d4750049b9794d94e

The inspected cp312 macOS arm64 release wheel has SHA256
05f3e68107bb7fd883fb5ef828d5ea63a015d44c0fc5ce010c213b12496a92b5.
Its classix/clustering.py has SHA256
76fa6ae9db7c10eec2de5adfe16700c9516433393d34fa3e2d05a0ce619d0ff0,
also verified against that file at the pinned commit. This is NOT a claim that
the entire release (including native binaries) equals the repository checkout.
Runtime distribution/module versions are checked and implementation files hashed;
version strings alone do not authenticate an installed wheel.

Upstream owns float64 conversion, mean centering, median centered-row-norm
scaling (zero scale becomes one), and PCA sorting. No external UMAP/PCA or
normalization is applied. CLASSIX has no random_state API: seed is audit-only.
PCA/SVD initialization, degenerate spectra, sorting ties, BLAS/SciPy versions,
and Python/Cython/platform dispatch preclude a cross-run/platform determinism
guarantee. We neither seed global RNGs nor patch upstream numerical routines.
"""
from dataclasses import dataclass
import hashlib
import importlib.metadata
import inspect
from numbers import Integral
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import issparse

from ..types import Partition


__all__ = ["CLASSIXBackend"]

_VERSION = "1.5.3"
_UPSTREAM_SHA = "70344c972d32f2bcaae6655d4750049b9794d94e"


def _input(vectors, seed, max_samples):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if (isinstance(max_samples, bool) or not isinstance(max_samples, Integral)
            or not 1 <= max_samples <= 5000):
        raise ValueError("max_samples must be an integer in [1, 5000]")
    if issparse(vectors):
        raise ValueError("nonempty dense 2-D real vectors required")
    try:
        raw = np.asarray(vectors)
    except (TypeError, ValueError) as exc:
        raise ValueError("nonempty dense 2-D real vectors required") from exc
    if raw.ndim != 2 or 0 in raw.shape or raw.dtype.kind not in "fiu":
        raise ValueError("nonempty dense 2-D real vectors required")
    if len(raw) > max_samples:
        raise ValueError(f"n={len(raw)} exceeds max_samples={max_samples}")
    # Check conversion safety, but hand the original dtype/values to CLASSIX.
    with np.errstate(over="ignore", invalid="ignore"):
        finite = np.isfinite(raw).all() and np.isfinite(raw.astype(np.float64)).all()
    if not finite:
        raise ValueError("finite float64-representable vectors required")
    x = np.array(raw, order="C", copy=True)
    digest = hashlib.sha256(x.tobytes()).hexdigest()
    return x, dict(shape=list(x.shape), source_dtype=raw.dtype.str,
                   fit_dtype=x.dtype.str, source_vectors_sha256=digest,
                   fit_vectors_sha256=digest, fit_vector_bytes=int(x.nbytes),
                   normalization="none in adapter; upstream preprocessing is part of CLASSIX",
                   identity_stage="original row order, before upstream conversion/scaling")


def _parameters(model):
    # 1.5.3 has no sklearn get_params(). These are its actual attribute names.
    return dict(sorting=model.sorting, metric=model.metric, radius=float(model.radius),
                minPts=int(model.minPts), group_merging=model.group_merging,
                mergeScale=float(model.mergeScale_), post_alloc=bool(model._CLASSIX__post_alloc),
                mergeTinyGroups=bool(model._CLASSIX__mergeTinyGroups),
                verbose=int(model._CLASSIX__verbose), short_log_form=bool(model._CLASSIX__truncate))


def _backend(function):
    module = sys.modules[function.__module__]
    path = Path(module.__file__)
    return dict(callable=f"{function.__module__}.{function.__name__}",
                module_file=str(path), module_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@dataclass(frozen=True)
class CLASSIXBackend:
    """Fixed radius preset; no K, gold, radius-search, or layer-selection API.

    The hard sample cap is an admission guard, not a memory/runtime guarantee:
    feature width and upstream workspace also contribute to resource use.
    Zero rows and constant matrices are valid for this Euclidean preset.
    """

    seed: int = 0
    max_samples: int = 5000

    def fit(self, vectors):
        x, identity = _input(vectors, self.seed, self.max_samples)
        try:
            version = importlib.metadata.version("classixclustering")
        except importlib.metadata.PackageNotFoundError as exc:
            raise ImportError("CLASSIX candidate requires classixclustering==1.5.3") from exc
        if version != _VERSION:
            raise RuntimeError(f"classixclustering requires version {_VERSION}, found {version}; "
                               "review preset before upgrading")
        import classix  # Optional dependency; never imported at module import time.

        if getattr(classix, "__version__", None) != _VERSION:
            raise RuntimeError("classix module version differs from required release 1.5.3")
        requested = dict(sorting="pca", metric="euclidean", radius=0.5, minPts=1,
                         group_merging="distance", mergeScale=1.5, post_alloc=True,
                         mergeTinyGroups=True, verbose=0, short_log_form=True)
        model = classix.CLASSIX(**requested)
        if _parameters(model) != requested:
            raise RuntimeError("CLASSIX parameters differ from the fixed release preset")
        start = time.perf_counter()
        model.fit(x)
        elapsed = time.perf_counter() - start
        params = _parameters(model)
        if params != requested:
            raise RuntimeError("CLASSIX changed fixed parameters during fit")
        labels = np.asarray(model.labels_)
        if labels.shape != (len(x),) or labels.dtype.kind not in "iu" or np.any(labels < -1):
            raise RuntimeError("invalid CLASSIX labels; integer IDs and -1 noise required")
        count = model.nrDistComp_
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise RuntimeError("invalid CLASSIX aggregation distance count")
        # Actual fitted internal data, not a second application of preprocessing.
        fitted = np.asarray(model._fit_data_)
        if fitted.shape != x.shape or not np.isfinite(fitted).all():
            raise RuntimeError("invalid CLASSIX preprocessed data")
        scale = float(model.dataScale_)
        if not np.isfinite(scale) or scale <= 0:
            raise RuntimeError("invalid CLASSIX preprocessing scale")
        # Release fit switches even Windows PCA dispatch to lm for all-zero data.
        aggregate = model._aggregate if np.any(fitted) else model._lm_aggregate
        metadata = dict(
            algorithm="CLASSIX", implementation="classix.CLASSIX",
            backend_protocol="classix-fixed-radius-v1", classification="author package candidate",
            k_regime="UnknownK", seed=int(self.seed), seed_used=False,
            determinism="not guaranteed; upstream PCA/SVD has no seed API; ties and backend may vary",
            versions={"classixclustering": version, "classix": classix.__version__,
                      "numpy": np.__version__, "scipy": importlib.metadata.version("scipy")},
            source=dict(repository="nla-group/classix", reviewed_commit=_UPSTREAM_SHA,
                        implementation_origin="released package, not a pinned-HEAD installation",
                        estimator_sha256=hashlib.sha256(
                            Path(inspect.getfile(classix.CLASSIX)).read_bytes()).hexdigest()),
            backend=dict(aggregation=_backend(aggregate), distance_merging=_backend(model._distance_merge)),
            requested_params=requested, params=params, input=identity,
            max_samples=int(self.max_samples), clustering_seconds=elapsed,
            preprocessing=dict(owner="upstream CLASSIX", dtype=fitted.dtype.str,
                               centering="column mean", scaling="median centered row L2 norm; zero -> 1",
                               data_scale=scale, external_reduction=None),
            distance_diagnostic=dict(nrDistComp_=int(count), scope="aggregation only",
                                     includes_merging=False, total_distance_computations=None),
            actual_k=int(len(np.unique(labels[labels >= 0]))), noise_label=-1,
            noise_count=int(np.count_nonzero(labels == -1)),
            noise_policy="retain upstream labels; no adapter reassignment",
            parameter_policy="fixed before evaluation; no K, gold, or radius selection",
        )
        return Partition(labels.copy(), metadata)
