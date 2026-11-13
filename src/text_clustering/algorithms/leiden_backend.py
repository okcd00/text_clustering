"""Standalone, label-blind kNN + Leiden engineering baseline (not text SOTA).

Fixed preset: exact brute-force cosine k=15 excluding self, symmetric union,
max aggregation of duplicate edges, positive max(0, 1-distance) weights only.
Vertices retain input row order, including isolates. An empty positive-weight
graph is rejected: modularity has a zero total-weight denominator in that case.
Equal-distance boundary ties follow sklearn's selection, not a new tie-breaker.

The optional runtime pins are igraph==1.0.0 and leidenalg==0.12.0. They are
checked at fit time; importing this module does not import either dependency.
leidenalg's GPL-3.0-or-later dependency is recorded, not a relicensing of this
repository. No factory registration, installation, or thread control is done.

Primary sources checked 2026-11-04:
https://pypi.org/project/leidenalg/0.12.0/
https://pypi.org/project/igraph/1.0.0/
https://github.com/vtraag/leidenalg/blob/0.12.0/pyproject.toml
https://github.com/vtraag/leidenalg/blob/0.12.0/src/leidenalg/functions.py
https://python.igraph.org/en/1.0.0/api/igraph.Graph.html
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
The tagged find_partition API defines n_iterations=-1 as stopping at the first
iteration without improvement, not a guarantee of a global optimum.
"""

from dataclasses import dataclass
import hashlib
import importlib.metadata
from numbers import Integral
import os
import platform
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .discovery_backends import _input, _partition, _version
from ..types import Partition


IGRAPH_VERSION = "1.0.0"
LEIDENALG_VERSION = "0.12.0"
BACKEND_PROTOCOL = "knn15-cosine-union-positive-leiden-modularity-v1"
_THREAD_ENV = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
)


def _positive_union(distances, indices):
    """Canonical undirected edges; no summing reciprocal arcs or self loops.

    Graph identity v1 hashes a fixed header, little-endian int64 [n, m], then
    sorted (u < v) int64 endpoint pairs, then matching float64 weights. The
    vertex count binds isolated vertices too; this is not an isomorphism hash.
    """
    distances, indices = np.asarray(distances), np.asarray(indices)
    if (distances.ndim != 2 or distances.shape != indices.shape
            or distances.dtype.kind not in "fiu" or indices.dtype.kind not in "iu"
            or not np.isfinite(distances).all()
            or np.any(distances < 0) or np.any(distances > 2)
            or np.any(indices < 0) or np.any(indices >= len(indices))):
        raise RuntimeError("invalid cosine nearest-neighbor output")
    merged = {}
    positive_arcs = zero_arcs = self_arcs = 0
    for source, (neighbors, row_distances) in enumerate(zip(indices, distances)):
        for target, distance in zip(neighbors, row_distances):
            target = int(target)
            if source == target:
                self_arcs += 1
                continue
            weight = max(0.0, 1.0 - float(distance))
            if weight == 0.0:
                zero_arcs += 1
                continue
            positive_arcs += 1
            edge = (min(source, target), max(source, target))
            merged[edge] = max(merged.get(edge, 0.0), weight)
    edges = sorted(merged)
    weights = [merged[edge] for edge in edges]
    endpoints = np.asarray(edges, dtype="<i8").reshape(-1, 2)
    weight_array = np.asarray(weights, dtype="<f8")
    digest = hashlib.sha256(b"leiden-positive-union-graph-v1\0")
    digest.update(np.asarray([len(indices), len(edges)], dtype="<i8").tobytes())
    digest.update(endpoints.tobytes())
    digest.update(weight_array.tobytes())
    identity = {
        "graph_sha256": digest.hexdigest(),
        "identity_protocol": "leiden-positive-union-graph-v1",
        "vertex_order": "input row order",
        "vertex_count": len(indices),
        "edge_count": len(edges),
        "directed_neighbor_count": int(indices.size),
        "positive_neighbor_count": positive_arcs,
        "zero_weight_neighbors_dropped": zero_arcs,
        "self_neighbors_dropped": self_arcs,
        "duplicate_positive_edges_merged": positive_arcs - len(edges),
        "isolated_vertex_count": len(indices) - len(np.unique(endpoints)),
        "total_edge_weight": float(weight_array.sum()),
        "directed": False,
        "self_loops": False,
    }
    return edges, weights, identity


@dataclass(frozen=True)
class LeidenKNNBackend:
    """Fixed kNN15 + ModularityVertexPartition over 16..5000 dense rows.

    Only the seed and a resource guard (which may lower the 5000-row ceiling)
    are configurable. There is no cluster K, resolution, gold, or tuning API.
    Thread limits/joblib contexts belong to the caller; n_jobs=None preserves
    that control. Exact search can require quadratic distance work/storage.
    """

    seed: int = 0
    max_samples: int = 5000

    def fit(self, vectors) -> Partition:
        total_start = time.perf_counter()
        if isinstance(self.max_samples, Integral) and self.max_samples > 5000:
            raise ValueError("max_samples cannot exceed the preset ceiling of 5000")
        x, input_identity = _input(vectors, self.seed, self.max_samples)
        try:
            versions = {
                "igraph": _version("igraph", IGRAPH_VERSION),
                "leidenalg": _version("leidenalg", LEIDENALG_VERSION),
            }
        except importlib.metadata.PackageNotFoundError as exc:
            raise ImportError(
                "LeidenKNNBackend requires optional igraph==1.0.0 and "
                "leidenalg==0.12.0; no automatic installation is performed"
            ) from exc
        import igraph
        import leidenalg

        versions.update({name: importlib.metadata.version(name)
                         for name in ("numpy", "scipy", "scikit-learn")})
        versions["igraph_c"] = igraph.__igraph_version__
        versions["python"] = platform.python_version()
        neighbor_params = dict(n_neighbors=15, algorithm="brute", metric="cosine",
                               n_jobs=None)
        start = time.perf_counter()
        neighbors = NearestNeighbors(**neighbor_params).fit(x)
        # X=None excludes the query row by index, also with duplicate vectors.
        distances, indices = neighbors.kneighbors(X=None, return_distance=True)
        knn_seconds = time.perf_counter() - start
        if (np.shape(indices) != (len(x), 15)
                or np.any(np.asarray(indices) == np.arange(len(x))[:, None])):
            raise RuntimeError("expected exactly 15 non-self neighbors per row")

        start = time.perf_counter()
        edges, weights, graph_identity = _positive_union(distances, indices)
        if not edges:
            raise ValueError(
                "empty positive-weight kNN graph: modularity is undefined; "
                "preset rejects rather than inventing a partition"
            )
        graph = igraph.Graph(n=len(x), edges=edges, directed=False)
        graph_seconds = time.perf_counter() - start
        leiden_params = dict(initial_membership=None, n_iterations=-1,
                             max_comm_size=0, seed=int(self.seed))
        start = time.perf_counter()
        result = leidenalg.find_partition(
            graph, leidenalg.ModularityVertexPartition, weights=weights,
            **leiden_params,
        )
        clustering_seconds = time.perf_counter() - start
        partition = _partition(result.membership, len(x), {})
        if np.any(partition.labels < 0):
            raise RuntimeError("Leiden must assign every vertex; negative labels are invalid")
        partition.metadata = {
            "algorithm": "kNN + Leiden",
            "implementation": "sklearn.neighbors.NearestNeighbors + leidenalg.find_partition",
            "backend_protocol": BACKEND_PROTOCOL,
            "classification": "engineering baseline",
            "seed": int(self.seed),
            "max_samples": int(self.max_samples),
            "versions": versions,
            "license": "GPL-3.0-or-later",
            "license_scope": "leidenalg dependency; not a repository license declaration",
            "params": {
                "neighbors": neighbor_params,
                "graph": {"symmetrization": "union", "duplicate_edges": "max",
                          "weight": "max(0, 1-cosine_distance)",
                          "zero_weight_edges": "drop", "self_loops": False},
                "leiden": dict(leiden_params, partition_type="ModularityVertexPartition",
                               weights="positive graph edge weights"),
            },
            "input": input_identity,
            "graph": graph_identity,
            "knn_seconds": knn_seconds,
            "graph_seconds": graph_seconds,
            "clustering_seconds": clustering_seconds,
            "total_seconds": time.perf_counter() - total_start,
            "timing_scope": "wall clock; total includes validation and dependency loading",
            "stopping_policy": "first iteration without improvement; not global optimality",
            "empty_graph_policy": "reject with ValueError; modularity undefined",
            "neighbor_tie_policy": "sklearn selection; no cross-version tie guarantee",
            "parameter_policy": "fixed before evaluation; no gold, cluster K or resolution",
            "thread_policy": "externally controlled; no environment or threadpool mutation",
            "thread_environment": {name: os.environ.get(name) for name in _THREAD_ENV},
            "noise_label": -1,
            "noise_policy": "no rejection; retain upstream membership for all vertices",
        }
        return partition
