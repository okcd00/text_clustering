"""Given-K geometric engineering baselines; neither is a paper reproduction.

Both expose ``fit(vectors) -> Partition`` and never accept labels/gold or infer K.
Dense, finite, nonzero real embeddings only; caller data are never mutated.

Semantic references (checked 2026-11-03; locally tested sklearn 1.6.1):
- https://scikit-learn.org/1.6/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
- https://scikit-learn.org/1.6/modules/generated/sklearn.cluster.KMeans.html
- https://scikit-learn.org/1.6/modules/generated/sklearn.cluster.AgglomerativeClustering.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

sklearn KMeans minimizes squared Euclidean distances; normalizing only its input
is not spherical k-means. Average linkage averages *pairwise* distances across
clusters, not distances between their centroids.
"""
from numbers import Real

import numpy as np
import scipy
import sklearn
from sklearn.cluster import AgglomerativeClustering

from ..types import Partition


__all__ = ["SphericalKMeansBackend", "AgglomerativeCosineBackend"]


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _seed(value):
    value = _integer(value, "seed", 0)
    if value >= 2**32:
        raise ValueError("seed must be < 2**32")
    return value


def _vectors(vectors, n_clusters, *, require_unit=False, max_samples=None):
    try:
        raw = np.asarray(vectors)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected a dense real matrix [n, d]") from exc
    if (raw.ndim != 2 or 0 in raw.shape
            or not np.issubdtype(raw.dtype, np.number)
            or np.iscomplexobj(raw)):
        raise ValueError("expected a nonempty dense real matrix [n, d]")
    if n_clusters > len(raw):
        raise ValueError("require 1 <= K <= n")
    # Reject before making float64 copies or allocating pairwise distances.
    if max_samples is not None and len(raw) > max_samples:
        raise ValueError(
            f"n={len(raw)} exceeds max_samples={max_samples}; cosine average "
            "linkage needs quadratic memory. Raise the limit explicitly only "
            "after budgeting pairwise storage and working memory."
        )
    with np.errstate(over="ignore", invalid="ignore"):
        x = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(x).all():
        raise ValueError("vectors must be finite float64-representable values")
    scale = np.max(np.abs(x), axis=1)
    if np.any(scale == 0):
        raise ValueError("zero vectors have undefined cosine direction")
    # Scale first: both huge finite values and subnormal nonzero rows are safe.
    x /= scale[:, None]
    norms = np.linalg.norm(x, axis=1)
    if require_unit and not np.allclose(scale, 1.0 / norms, rtol=1e-5, atol=0):
        raise ValueError("AgglomerativeCosineBackend requires L2-normalized rows")
    x /= norms[:, None]
    return x


class SphericalKMeansBackend:
    """Lloyd-style cosine assignment and normalized sum updates, implemented here.

    Non-unit input is normalized. Each restart samples K distinct row indices
    without replacement (not k-means++), using a local seeded NumPy Generator.
    The lowest final sum(1 - cosine) wins; exact restart ties keep the first run.
    Assignment ties choose the lowest center index. Empty centers are reseeded
    from the worst represented rows, with stable row-order tie breaking. A
    nonempty cluster whose sum cancels to zero retains its previous unit center.
    Duplicate directions can still give actual_k < requested_k; no forced split.

    Final labels always correspond to cosine argmax of ``cluster_centers_``.
    At an iteration/tolerance stop those centers need not be normalized sums of
    the final labels; ``converged`` means assignment stability, not global
    optimality. ``tol`` bounds maximum Euclidean displacement of unit centers.

    With R=n_init, T=max_iter, B=min(batch_size,N): time is
    O(R*T*(N*K*d + N*log(N))) worst case (sorting only when centers are empty).
    Memory is O(N*d + B*K + K*d + N), excluding small restart/history records.
    This is full-data clustering, not a streaming fit or a large-N guarantee.
    Repeatability assumes the same row order, numeric stack and configuration.
    """

    def __init__(self, n_clusters, seed=0, n_init=10, max_iter=300,
                 tol=1e-6, batch_size=1024):
        self.n_clusters = _integer(n_clusters, "n_clusters")
        self.seed = _seed(seed)
        self.n_init = _integer(n_init, "n_init")
        self.max_iter = _integer(max_iter, "max_iter")
        self.batch_size = _integer(batch_size, "batch_size")
        if (isinstance(tol, (bool, np.bool_)) or not isinstance(tol, Real)
                or not np.isfinite(tol) or tol < 0):
            raise ValueError("tol must be finite and nonnegative")
        self.tol = float(tol)

    def _assign(self, x, centers):
        labels = np.empty(len(x), dtype=np.int64)
        similarities = np.empty(len(x), dtype=np.float64)
        for start in range(0, len(x), self.batch_size):
            stop = min(start + self.batch_size, len(x))
            scores = x[start:stop] @ centers.T
            chosen = np.argmax(scores, axis=1)
            labels[start:stop] = chosen
            similarities[start:stop] = scores[np.arange(stop - start), chosen]
        # Roundoff can put unit-vector dot products a few ulps outside [-1, 1].
        return labels, np.clip(similarities, -1.0, 1.0)

    def _run(self, x, rng):
        centers = x[rng.choice(len(x), self.n_clusters, replace=False)].copy()
        labels, scores = self._assign(x, centers)
        history = [float(np.sum(1.0 - scores))]
        reseeds = 0
        converged = False
        stop_reason = "max_iter"
        for iteration in range(1, self.max_iter + 1):
            sums = np.zeros_like(centers)
            np.add.at(sums, labels, x)
            counts = np.bincount(labels, minlength=self.n_clusters)
            updated = centers.copy()
            sum_scale = np.max(np.abs(sums), axis=1)
            nonzero = sum_scale > 0
            # Near-antipodal vectors can leave a tiny nonzero sum; scale before
            # squaring rather than mistaking norm underflow for cancellation.
            directions = sums[nonzero] / sum_scale[nonzero, None]
            updated[nonzero] = directions / np.linalg.norm(directions, axis=1)[:, None]
            empty = np.flatnonzero(counts == 0)
            if len(empty):
                worst = np.argsort(scores, kind="stable")[:len(empty)]
                updated[empty] = x[worst]
                reseeds += len(empty)
            shift = float(np.max(np.linalg.norm(updated - centers, axis=1)))
            new_labels, scores = self._assign(x, updated)
            history.append(float(np.sum(1.0 - scores)))
            converged = bool(np.array_equal(labels, new_labels))
            labels, centers = new_labels, updated
            if converged or shift <= self.tol:
                stop_reason = "labels_stable" if converged else "tolerance"
                break
        return labels, centers, {
            "cosine_loss": history[-1], "objective_history": history,
            "n_iter": iteration, "converged": converged,
            "stop_reason": stop_reason, "empty_center_reseeds": reseeds,
        }

    def fit(self, vectors):
        x = _vectors(vectors, self.n_clusters)
        rng = np.random.default_rng(self.seed)
        best = None
        best_run = 0
        losses = []
        for run in range(self.n_init):
            candidate = self._run(x, rng)
            losses.append(candidate[2]["cosine_loss"])
            if best is None or losses[-1] < best[2]["cosine_loss"]:
                best, best_run = candidate, run
        labels, self.cluster_centers_, diagnostics = best
        return Partition(labels, {
            "algorithm": "SphericalKMeans", "implementation": "engineering_baseline",
            "paper_reproduction": False, "implementation_version": 1,
            "seed": self.seed, "requested_k": self.n_clusters,
            "actual_k": int(len(np.unique(labels))), "metric": "cosine",
            "objective": "sum(1 - cosine_to_assigned_unit_center)",
            "input_normalization": "L2 rows", "numpy_version": np.__version__,
            "params": {"n_clusters": self.n_clusters, "seed": self.seed,
                       "n_init": self.n_init, "max_iter": self.max_iter,
                       "tol": self.tol, "batch_size": self.batch_size,
                       "init": "random_rows_without_replacement"},
            "best_run": best_run, "restart_losses": losses, **diagnostics,
        })


class AgglomerativeCosineBackend:
    """sklearn unstructured cosine average linkage for small/medium dense inputs.

    Require L2 row norms within relative tolerance 1e-5, then renormalize to
    remove input roundoff. No connectivity graph, sampling or approximate
    shortcut is used. The full hierarchy is built and cut at caller-supplied K;
    only a Partition is exposed, not the project's hierarchical task protocol.
    ``seed`` is recorded for a uniform interface, but unused: sklearn has no
    random_state for this algorithm. Ties depend on row order/library version.

    Pairwise cosine computation costs O(N^2*d); the unstructured SciPy average
    linkage path costs O(N^2) time and memory, plus O(N*d) embeddings. The default
    max_samples=5000 is a conservative guard, NOT a peak-memory budget: a single
    float64 condensed distance array alone is 8*N*(N-1)/2 bytes (~100 MB at 5000),
    and copies/workspace increase peak usage. Raising the cap is explicit opt-in.
    K=1 and K=N are exact trivial cuts, handled without building the tree;
    input validation and the sample-count guard still apply to these cases.
    """

    def __init__(self, n_clusters, seed=0, max_samples=5000):
        self.n_clusters = _integer(n_clusters, "n_clusters")
        self.seed = _seed(seed)
        self.max_samples = _integer(max_samples, "max_samples")

    def fit(self, vectors):
        x = _vectors(vectors, self.n_clusters, require_unit=True,
                     max_samples=self.max_samples)
        trivial_cut = self.n_clusters in (1, len(x))
        if self.n_clusters == 1:
            labels = np.zeros(len(x), dtype=np.int64)
        elif self.n_clusters == len(x):
            labels = np.arange(len(x), dtype=np.int64)
        else:
            model = AgglomerativeClustering(
                n_clusters=self.n_clusters, metric="cosine", linkage="average",
                connectivity=None, compute_full_tree=True,
                distance_threshold=None, compute_distances=False,
            )
            labels = model.fit_predict(x).astype(np.int64, copy=False)
        return Partition(labels, {
            "algorithm": "AgglomerativeCosineAverage",
            "implementation": "sklearn_backend", "paper_reproduction": False,
            "implementation_version": 1, "seed": self.seed, "seed_used": False,
            "requested_k": self.n_clusters, "actual_k": int(len(np.unique(labels))),
            "metric": "cosine", "linkage": "average", "trivial_cut": trivial_cut,
            "input_normalization": "require L2 rows (rtol=1e-5); renormalize",
            "params": {"n_clusters": self.n_clusters, "metric": "cosine",
                       "linkage": "average", "connectivity": None,
                       "compute_full_tree": True, "distance_threshold": None,
                       "compute_distances": False, "max_samples": self.max_samples},
            "condensed_distance_bytes": 0 if trivial_cut else 8 * len(x) * (len(x) - 1) // 2,
            "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
            "scipy_version": scipy.__version__,
        })
