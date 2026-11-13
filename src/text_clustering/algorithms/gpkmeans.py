"""GPKmeans assignment core: equations (5)--(12), Algorithm 2 of nie2026gpkmeans.

Not the paper's LLM acquisition or Active GPKmeans pipeline. Groups use row
indices, not document IDs. Contradictory/overlapping groups are soft evidence;
even with no groups this performs Hartigan-style moves after Lloyd initialization.
"""
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from sklearn.cluster import KMeans

from ..types import Partition


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _nonnegative(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _vectors(vectors):
    raw = np.asarray(vectors)
    if np.iscomplexobj(raw):
        raise ValueError("vectors must be real")
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 2 or 0 in x.shape or not np.isfinite(x).all():
        raise ValueError("vectors must be a nonempty finite 2D matrix")
    return x


@dataclass(frozen=True)
class Groups:
    """Immutable groups; repeated groups intentionally count as repeated evidence.

    No transitive closure or consistency filtering is performed. ML components
    and CL separation sets must be constructed upstream when using that protocol.
    """
    must_link: tuple = ()
    cannot_link: tuple = ()

    def __post_init__(self):
        for name in ("must_link", "cannot_link"):
            try:
                value = tuple(tuple(group) for group in getattr(self, name))
            except TypeError as exc:
                raise ValueError("groups must be collections of index sequences") from exc
            object.__setattr__(self, name, value)

    def validate(self, n):
        for group in self.must_link + self.cannot_link:
            # Check types before set/indexing: bool otherwise becomes a NumPy mask.
            if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
                   or not 0 <= i < n for i in group):
                raise ValueError("group index must be an integer inside dataset")
            if len(group) < 2 or len(set(group)) != len(group):
                raise ValueError("each group needs at least two distinct indices")


def penalty(counts, kind):
    """Normalized ML majority cost or CL within-group pair collision fraction."""
    counts = np.asarray(counts, dtype=np.float64)
    if (kind not in ("ml", "cl") or counts.ndim != 1 or not counts.size
            or not np.isfinite(counts).all() or (counts < 0).any()
            or (counts != np.floor(counts)).any() or counts.sum() < 2):
        raise ValueError("penalty requires kind ml/cl and integer counts totaling >= 2")
    m = counts.sum()
    return float((m - counts.max()) / (m - 1) if kind == "ml"
                 else np.sum(counts * (counts - 1)) / (m * (m - 1)))


def _labels(labels, n, k=None):
    z = np.asarray(labels)
    if (z.shape != (n,) or z.dtype.kind not in "iu" or (z < 0).any()
            or (k is not None and (z >= k).any())):
        raise ValueError("labels must be a length-N integer vector of nonnegative cluster IDs")
    if k is not None and len(np.unique(z)) != k:
        raise ValueError("initial_labels must populate every cluster in [0, K)")
    return z


def objective(x, labels, groups, lambda_ml=1.0, lambda_cl=1.0):
    """Slow independent Eq. (7) oracle; labels cannot contain rejection (-1).

    Noncontiguous nonnegative IDs are allowed for diagnostics and compacted
    before counting; this does not change the partition or group penalties.
    """
    x = _vectors(x)
    groups.validate(len(x))
    lambda_ml = _nonnegative(lambda_ml, "lambda_ml")
    lambda_cl = _nonnegative(lambda_cl, "lambda_cl")
    _, z = np.unique(_labels(labels, len(x)), return_inverse=True)
    sse = sum(float(((x[z == c] - x[z == c].mean(axis=0)) ** 2).sum())
              for c in np.unique(z))
    for collection, weight, kind in ((groups.must_link, lambda_ml, "ml"),
                                      (groups.cannot_link, lambda_cl, "cl")):
        if weight:
            for g in collection:
                sse += weight * penalty(np.bincount(z[list(g)]), kind)
    return float(sse)


def _group_move_deltas(occupancy, source, kind):
    """All K exact penalty deltas in O(K), including a zero stay-put entry.

    Remove the point once. For ML, each target can only increase the resulting
    maximum to its own count + 1; for CL use Eq. (10) directly. Input is not mutated.
    """
    m = int(occupancy.sum())
    if kind == "ml":
        remaining = occupancy.copy()
        remaining[source] -= 1
        new_max = np.maximum(remaining.max(), remaining + 1)
        deltas = (occupancy.max() - new_max) / (m - 1)
    elif kind == "cl":
        deltas = 2.0 * (occupancy - occupancy[source] + 1) / (m * (m - 1))
    else:
        raise ValueError("kind must be ml or cl")
    deltas[source] = 0.0
    return deltas


class GPKMeans:
    def __init__(self, n_clusters, groups=Groups(), lambda_ml=1.0, lambda_cl=1.0,
                 seed=0, max_sweeps=50, tolerance=1e-10, *, n_init=10):
        self.k = _integer(n_clusters, "n_clusters")
        if not isinstance(groups, Groups):
            raise ValueError("groups must be a Groups instance")
        self.groups = groups
        self.lambda_ml = _nonnegative(lambda_ml, "lambda_ml")
        self.lambda_cl = _nonnegative(lambda_cl, "lambda_cl")
        self.tolerance = _nonnegative(tolerance, "tolerance")
        self.max_sweeps = _integer(max_sweeps, "max_sweeps")
        self.n_init = _integer(n_init, "n_init")
        self.seed = seed

    def fit(self, vectors, *, initial_labels=None):
        """Return Partition (not self); do not mutate vectors or initial_labels.

        Optional initial_labels bypasses KMeans, enabling paired same-start
        experiments. It must populate exactly K clusters labeled 0..K-1.
        n_init is used only for the default k-means++/Lloyd initialization.
        """
        x = _vectors(vectors)
        if self.k > len(x):
            raise ValueError("n_clusters cannot exceed dataset size")
        self.groups.validate(len(x))
        # Translation leaves SSE unchanged and avoids sum-update cancellation
        # on embeddings with a large common offset. No L2 normalization is applied.
        x = x - x.mean(axis=0)
        if not np.isfinite(x).all():
            raise ValueError("vectors exceed safe float64 centering range")
        if initial_labels is None:
            z = KMeans(n_clusters=self.k, random_state=self.seed, n_init=self.n_init,
                       init="k-means++", algorithm="lloyd").fit_predict(x)
            if len(np.unique(z)) != self.k:
                raise ValueError("initialization has fewer nonempty clusters than K")
        else:
            z = _labels(initial_labels, len(x), self.k).astype(np.intp, copy=True)
        counts = np.bincount(z, minlength=self.k)
        sums = np.zeros((self.k, x.shape[1]))
        np.add.at(sums, z, x)
        centers = sums / counts[:, None]
        cache, incident = [], [[] for _ in x]
        for collection, weight, kind in ((self.groups.must_link, self.lambda_ml, "ml"),
                                         (self.groups.cannot_link, self.lambda_cl, "cl")):
            if not weight:
                continue
            for group in collection:
                index = len(cache)
                cache.append((np.bincount(z[list(group)], minlength=self.k), weight, kind))
                for i in group:
                    incident[i].append(index)
        history = [objective(x, z, self.groups, self.lambda_ml, self.lambda_cl)]
        moves_per_sweep = []
        converged = False
        for _ in range(self.max_sweeps):
            moved = 0
            for i, xi in enumerate(x):
                r = z[i]
                if counts[r] == 1:
                    continue  # preserve K nonempty clusters, as Algorithm 2
                # Reuse centroids; only accepted moves change two rows. Direct
                # differences avoid cancellation in ||x||²+||c||²-2<x,c>.
                difference = centers - xi
                distances = np.einsum("ij,ij->i", difference, difference)
                deltas = (counts / (counts + 1)) * distances
                deltas -= counts[r] / (counts[r] - 1) * distances[r]
                for index in incident[i]:
                    occupancy, weight, kind = cache[index]
                    deltas += weight * _group_move_deltas(occupancy, r, kind)
                deltas[r] = 0
                q = int(np.argmin(deltas))
                if deltas[q] < -self.tolerance:
                    counts[r] -= 1
                    counts[q] += 1
                    sums[r] -= xi
                    sums[q] += xi
                    centers[r] = sums[r] / counts[r]
                    centers[q] = sums[q] / counts[q]
                    z[i] = q
                    for index in incident[i]:
                        cache[index][0][r] -= 1
                        cache[index][0][q] += 1
                    moved += 1
            history.append(objective(x, z, self.groups, self.lambda_ml, self.lambda_cl))
            moves_per_sweep.append(moved)
            if moved == 0:
                converged = True
                break
        return Partition(z, {"algorithm": "GPKMeans-core", "seed": self.seed,
            "objective_history": history, "converged": converged,
            "implementation": "equations-5-12-algorithm-2; no LLM acquisition",
            "initialization": "supplied-labels" if initial_labels is not None else "k-means++/lloyd",
            "n_init": None if initial_labels is not None else self.n_init,
            "n_clusters": self.k, "lambda_ml": self.lambda_ml, "lambda_cl": self.lambda_cl,
            "tolerance": self.tolerance, "max_sweeps": self.max_sweeps,
            "n_sweeps": len(moves_per_sweep), "moves_per_sweep": moves_per_sweep,
            "stop_reason": "no-improving-move" if converged else "max_sweeps",
            "group_counts": {"must_link": len(self.groups.must_link),
                             "cannot_link": len(self.groups.cannot_link)},
            "constraint_source": "caller-supplied; not inferred by solver"})
