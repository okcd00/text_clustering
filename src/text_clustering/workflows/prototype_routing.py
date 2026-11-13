"""Frozen known-support routing over precomputed vectors, without calibration.

This facade delegates to the evaluated PrototypeRouter. It does not encode text,
fit queries, tune gates, update online, or insert taxonomy nodes. Hierarchical
routing means parent-first gating only. Callers own support/query separation
and must supply vectors from the same representation; dimension checks alone
cannot verify encoder identity or that embeddings came from a real model.
"""
from numbers import Integral, Real

import numpy as np

from ..algorithms.prototype_routing import PrototypeRouter, RoutingScores


__all__ = ["FrozenPrototypeRouting"]


def _finite_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


def _gates(threshold, min_margin):
    # tune() uses endpoints outside the cosine/margin ranges for all-reject.
    threshold = _finite_real(threshold, "threshold")
    min_margin = _finite_real(min_margin, "min_margin")
    if min_margin < 0:
        raise ValueError("min_margin must be nonnegative")
    return threshold, min_margin


def _vectors(vectors, dimension=None, *, allow_empty=False, copy=True):
    try:
        raw = np.asarray(vectors)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected dense real matrix [n, d] with d > 0") from exc
    if raw.ndim != 2 or raw.shape[1] == 0 or raw.dtype.kind not in "fiu":
        raise ValueError("expected dense real matrix [n, d] with d > 0")
    if not allow_empty and raw.shape[0] == 0:
        raise ValueError("training support cannot be empty")
    if dimension is not None and raw.shape[1] != dimension:
        raise ValueError("vector dimensions do not match fitted support")
    if not copy:
        # Batched queries: backend checks finite/nonzero rows while converting
        # each block. Avoid allocating a full float64 copy before batching.
        return raw
    with np.errstate(over="ignore", invalid="ignore"):
        x = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(x).all():
        raise ValueError("vectors must be finite float64-representable values")
    # Keep values unchanged: the backend owns normalization and zero-row guards.
    return x


def _support_ids(values, count, name):
    try:
        result = np.array(values, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional ID sequence") from exc
    if result.ndim != 1 or len(result) != count:
        raise ValueError(f"{name} must align with training support")
    for value in result:
        if isinstance(value, (str, bytes)):
            if not value.strip():
                raise ValueError(f"{name} cannot contain empty IDs")
        elif isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
            _finite_real(value, name)
        else:
            raise ValueError(f"{name} must contain string or finite numeric IDs")
    # Object arrays must also have comparable scalar IDs for the backend's unique.
    try:
        np.unique(result)
    except TypeError as exc:
        raise ValueError(f"{name} must contain mutually comparable IDs") from exc
    return result


class FrozenPrototypeRouting:
    """Fit independent known-class support; score/predict never change the model.

    ``prototypes`` is 1 (centroid) or 3 (per-leaf KMeans). Each leaf must have
    one parent; at least two leaves are required, and three-prototype routing
    needs at least three support rows per leaf. These are backend constraints.
    ``threshold`` may be any finite real number; ``min_margin`` must be finite
    and nonnegative, including tune's 2.000001 all-reject endpoint.
    """

    def __init__(self, prototypes=1, seed=0, hierarchical=False,
                 threshold=0.55, min_margin=0.05):
        if (isinstance(prototypes, (bool, np.bool_))
                or not isinstance(prototypes, Integral) or prototypes not in (1, 3)):
            raise ValueError("supported prototype counts: 1, 3 (integers)")
        if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral)
                or not 0 <= seed < 2**32):
            raise ValueError("seed must be an unsigned 32-bit integer")
        if not isinstance(hierarchical, (bool, np.bool_)):
            raise ValueError("hierarchical must be boolean")
        self.prototypes, self.seed = int(prototypes), int(seed)
        self.hierarchical = bool(hierarchical)
        self.threshold, self.min_margin = _gates(threshold, min_margin)
        self._router = None

    def fit(self, vectors, leaf_ids, parent_ids):
        """Replace support atomically on success; no query gold or tuning inputs.

        Flat routing permits cancelling parent means while retaining valid leaf
        centers. Parent-first routing requires every parent center to be defined.
        """
        x = _vectors(vectors)
        leaves = _support_ids(leaf_ids, len(x), "leaf_ids")
        parents = _support_ids(parent_ids, len(x), "parent_ids")
        candidate = PrototypeRouter(self.prototypes, self.seed, self.hierarchical)
        # A failed fit (including a late KMeans/zero-center failure) leaves the
        # previously fitted router intact. No caller-owned arrays are retained.
        candidate.fit(x, leaves, parents)
        self._router = candidate
        return self

    def _fitted_router(self):
        if self._router is None:
            raise RuntimeError("fit known-class support before score or predict")
        return self._router

    def save(self, path, *, encoder_identity, support_identity):
        """Atomically save fitted centers/gates, not texts or support vectors.

        Provenance strings should identify the exact representation and ordered
        support set. Object/bytes label arrays cannot be stored. Normal models
        retain schema v1; undefined flat parent centers require schema v2.
        """
        from ..prototype_snapshot import save
        save(self, path, encoder_identity=encoder_identity, support_identity=support_identity)

    @classmethod
    def load(cls, path, *, expected_encoder_identity, expected_support_identity):
        """Restore without fitting or pickle; require explicit provenance matches."""
        from ..prototype_snapshot import load
        return load(path, expected_encoder_identity=expected_encoder_identity,
                    expected_support_identity=expected_support_identity)

    def score(self, vectors, *, batch_size=None) -> RoutingScores:
        """Score optionally in bounded row batches; preserve fitted dimension.

        Input arrays must not be mutated concurrently during this call.
        """
        router = self._fitted_router()
        values = _vectors(vectors, router.centers.shape[1], allow_empty=True,
                          copy=batch_size is None)
        return router.score(values, batch_size=batch_size)

    def predict(self, vectors, ids, *, batch_size=None):
        """Return backend decisions by unique ID; aligned empty batches yield {}.

        parent_id is an independent proposal among defined parent centers, not
        the assigned leaf's mapped parent. If all parent means cancel in flat
        mode, it is None even when a leaf is assigned.
        """
        self._fitted_router()
        threshold, min_margin = _gates(self.threshold, self.min_margin)
        if isinstance(ids, (str, bytes)):
            raise ValueError("ids must be a sequence of unique query IDs")
        try:
            ids = list(ids)
            for identity in ids:
                if identity is None or (isinstance(identity, (str, bytes))
                                        and not identity.strip()):
                    raise ValueError("query IDs cannot be missing or empty")
                if isinstance(identity, Real):
                    _finite_real(identity, "query ID")
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate query IDs")
        except TypeError as exc:
            raise ValueError("query IDs must be an iterable of hashable IDs") from exc
        scores = self.score(vectors, batch_size=batch_size)
        return PrototypeRouter.predict(scores, ids, threshold, min_margin)
