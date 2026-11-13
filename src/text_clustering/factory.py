"""Explicit allowlist: configurations select independent algorithm modules."""
from .algorithms import GPKMeans, Groups, HDBSCANBackend, KMeansBackend


def build_stream_algorithm(config):
    """Persistent-state protocol, separate from batch fit backends.

    Returned objects expose learn_one/predict_window, not fit. DBSTREAM accepts
    dense vectors from one encoder; TextClust accepts sparse token counts.
    sklearn_minibatch_stream requires fixed n_clusters and explicit k_source;
    it has no novelty rejection after its first complete learning chunk.
    sklearn_birch_stream requires n_clusters=None and emits snapshot-local CF
    leaf labels (unknown K), with no novelty rejection once learned.
    Use the matching dense/text workflow and record the representation identity.
    """
    config = dict(config)
    name = config.pop("name")
    if name in ("birch_capacity_freeze", "birch_rolling_rebuild"):
        from .algorithms.birch_maintenance import BirchMaintenance
        return BirchMaintenance(policy=name.removeprefix("birch_"), **config)
    if name == "sklearn_birch_stream":
        from .algorithms.birch_stream import BirchStreamBackend
        return BirchStreamBackend(**config)
    if name == "sklearn_minibatch_stream":
        from .research.minibatch_stream_backend import MiniBatchStreamBackend
        return MiniBatchStreamBackend(**config)
    if name == "clustream_data_init_v1":
        from .algorithms.clustream_data_init import CluStreamDataInit
        return CluStreamDataInit(**config)
    if name == "denstream_faded_cached_v1":
        from .research.denstream_cached_stream_backend import CachedDenStreamStreamBackend
        return CachedDenStreamStreamBackend(**config)
    if name in ("river_clustream", "river_denstream"):
        from .algorithms.river_micro_stream import RiverMicroStream
        return RiverMicroStream(method=name.removeprefix("river_"), **config)
    if name == "river_dbstream":
        from .algorithms.river_stream_backend import RiverStreamBackend
        return RiverStreamBackend(**config)
    if name == "river_textclust":
        from .algorithms.textclust import TextClustBackend
        return TextClustBackend(**config)
    if name == "river_textclust_snapshot":
        from .algorithms.textclust_snapshot import TextClustSnapshotBackend
        return TextClustSnapshotBackend(**config)
    raise ValueError(f"unknown stream algorithm: {name}")


def build_algorithm(config):
    config = dict(config)
    name = config.pop("name")
    if name == "finch_first":
        from .algorithms.finch_backend import FINCHBackend
        return FINCHBackend(**config)
    if name == "optics_xi":
        from .algorithms.optics import OPTICSXiBackend
        return OPTICSXiBackend(**config)
    if name == "classix_fixed":
        from .algorithms.classix_backend import CLASSIXBackend
        return CLASSIXBackend(**config)
    if name == "umap_fast_hdbscan":
        from .algorithms.fast_density_backend import UMAPFastHDBSCANBackend
        return UMAPFastHDBSCANBackend(**config)
    if name == "parc":
        from .algorithms.parc_backend import PARCBackend
        return PARCBackend(**config)
    if name == "leiden_knn":
        from .algorithms.leiden_backend import LeidenKNNBackend
        return LeidenKNNBackend(**config)
    if name == "minibatch_kmeans":
        return KMeansBackend(minibatch=True, **config)
    if name == "kmeans":
        return KMeansBackend(minibatch=False, **config)
    if name in ("spherical_kmeans", "agglomerative_cosine"):
        from .algorithms.geometric_backends import SphericalKMeansBackend, AgglomerativeCosineBackend
        return (SphericalKMeansBackend if name == "spherical_kmeans" else AgglomerativeCosineBackend)(**config)
    if name in ("bertopic_pipeline", "evoc"):
        from .algorithms.discovery_backends import BERTopicPipelineBackend, EVoCBackend
        return (BERTopicPipelineBackend if name == "bertopic_pipeline" else EVoCBackend)(**config)
    if name == "hdbscan_fixed":
        from .algorithms.density_backends import HDBSCANBackend as FixedHDBSCAN
        return FixedHDBSCAN(**config)
    if name == "hdbscan":
        return HDBSCANBackend(**config)
    if name == "gpkmeans":
        raw = config.pop("groups", {})
        groups = Groups(tuple(tuple(g) for g in raw.get("must_link", ())),
                        tuple(tuple(g) for g in raw.get("cannot_link", ())))
        return GPKMeans(groups=groups, **config)
    raise ValueError(f"unknown algorithm: {name}")


class _DenseEvolution:
    """Thin dense-vector facade; each run constructs a fresh stream backend.

    No scoring, encoding, identity alignment or experiment admission is added.
    Keep this adapter here so frozen backend/workflow evidence stays unchanged.
    """

    def __init__(self, *, algorithm, config=None):
        from collections.abc import Mapping
        from copy import deepcopy

        if not isinstance(algorithm, Mapping) or algorithm.get("name") not in (
                "sklearn_birch_stream", "sklearn_minibatch_stream", "river_dbstream",
                "river_clustream", "river_denstream",
                "clustream_data_init_v1", "denstream_faded_cached_v1",
                "birch_capacity_freeze", "birch_rolling_rebuild"):
            raise ValueError("evolution requires a dense stream algorithm mapping")
        if config is not None and not isinstance(config, Mapping):
            raise ValueError("evolution config must be a mapping")
        if set(config or {}) - {"warmup", "window_size", "update"}:
            raise ValueError("unknown evolution config option")
        self.algorithm = deepcopy(dict(algorithm))
        self.config = deepcopy(dict(config or {}))

    def run(self, vectors, order, *, on_prediction):
        from .workflows.evolution import run_window_protocol

        backend = build_stream_algorithm(self.algorithm)
        return run_window_protocol(backend, vectors, order,
                                   on_prediction=on_prediction, **self.config)


def build_tool(task, embedder=None, *, algorithm=None, config=None, judge=None,
               policy=None, generator=None):
    """Select a task facade; inject shared components without leaking gold.

    Clustering algorithms and navigation policies deliberately have different
    protocols. Frozen routing consumes precomputed known-support vectors, not
    an embedder or an online/navigation policy. A paper name is not evidence
    of a full pipeline implementation.
    """
    if task == "evolution":
        if any(value is not None for value in (embedder, judge, policy, generator)):
            raise ValueError("evolution accepts precomputed dense vectors, algorithm and config only")
        return _DenseEvolution(algorithm=algorithm, config=config)
    from .workflows.partitioning import ColdStart
    from .workflows.routing import OnePass, RoutingConfig
    from .workflows.navigation import Hierarchical, NavigationConfig
    from .algorithms.navigation import EvidenceNavigation
    if task == "text_evolution":
        from .workflows.text_snapshot_evolution import TextEvolution
        if any(value is not None for value in (embedder, judge, policy, generator)):
            raise ValueError("text_evolution accepts raw text, not embedding or navigation components")
        return TextEvolution(algorithm=algorithm, config=config)
    if task == "frozen_routing":
        from collections.abc import Mapping
        from .workflows.prototype_routing import FrozenPrototypeRouting
        if any(value is not None for value in (embedder, judge, policy, generator)):
            raise ValueError("frozen_routing accepts precomputed vectors, algorithm and config only")
        if not isinstance(algorithm, Mapping) or set(algorithm) != {"name"}:
            raise ValueError("frozen_routing requires algorithm with only a name")
        name = algorithm["name"]
        if not isinstance(name, str) or name not in ("centroid", "multi_prototype3", "knn"):
            raise ValueError(f"unknown frozen_routing algorithm: {name!r}")
        if config is not None and not isinstance(config, Mapping):
            raise ValueError("frozen_routing config must be a mapping")
        options = {} if config is None else dict(config)
        if name == "knn":
            from .algorithms.knn_routing import KNNRouting
            if set(options) - {"k", "threshold"}:
                raise ValueError("unknown knn config option")
            return KNNRouting(**options)
        if set(options) - {"seed", "hierarchical", "threshold", "min_margin"}:
            raise ValueError("unknown frozen_routing config option")
        return FrozenPrototypeRouting(prototypes=1 if name == "centroid" else 3, **options)
    config = {} if config is None else dict(config)
    if task in ("partitioning", "coldstart"):
        if algorithm is None or config or judge is not None or policy is not None or generator is not None:
            raise ValueError("coldstart requires algorithm only")
        return ColdStart(embedder, build_algorithm(algorithm) if isinstance(algorithm, dict) else algorithm)
    if task in ("routing", "onepass"):
        if algorithm is not None or policy is not None or generator is not None:
            raise ValueError("onepass accepts routing config and optional judge")
        return OnePass(embedder, RoutingConfig(**config), judge=judge)
    if task in ("navigation", "hier", "hierarchical"):
        if algorithm is not None or (policy is not None and judge is not None):
            raise ValueError("hierarchy accepts policy or default evidence policy with judge")
        return Hierarchical(policy if policy is not None else EvidenceNavigation(embedder, judge),
                            NavigationConfig(**config), generator=generator)
    raise ValueError(f"unknown task: {task}")
