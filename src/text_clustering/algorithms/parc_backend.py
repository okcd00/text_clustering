"""Optional PARC 0.40, with fail-closed guards around unsafe default paths.

This is NOT an unconditional stock-default baseline. Successful runs execute
the upstream pipeline unchanged (except cosine, one thread and Leiden seed).
Reject a run if its initial Leiden partition would need either default
postprocessing path: small_pop cleanup uses a wall clock; too_big refinement
hardcodes L2 and omits the thread setting. Never disable those operations and
then present the truncated result as PARC. Rejections must be reported, not
excluded from an aggregate over only successful datasets/seeds.

Audit source: official PyPI parc-0.40 wheel, downloaded 2026-11-04. Its source
hash is enforced before loading a per-run guard. No upstream files, module
globals, random generators or environment variables are modified. The guard
rebinds only run_subPARC's globals in a private subclass; its code object is
the original upstream code. A successful run cannot enter the unsafe paths.

Licensing is NOT just PARC's MIT: the runtime includes GPL igraph/leidenalg,
Apache-2.0 hnswlib, BSD numerical/UMAP packages and their transitive/bundled
components. Metadata records the installed dependency closure and shipped
license/notice hashes, including native components, rather than relicensing
that chain as MIT. Artifact-specific notices remain authoritative.
"""

from dataclasses import dataclass
import hashlib
import importlib.metadata
import inspect
from numbers import Integral
from pathlib import Path
import platform
import time
from types import FunctionType, MappingProxyType

import numpy as np

from .discovery_backends import _input, _partition, _version
from ..types import Partition


BACKEND_PROTOCOL = "parc-upstream-cosine-guarded-v1"
PARC_VERSION = "0.40"
HNSWLIB_VERSION = "0.8.0"
REQUIRED_VERSIONS = MappingProxyType({
    "parc": PARC_VERSION, "hnswlib": HNSWLIB_VERSION,
    "igraph": "1.0.0", "leidenalg": "0.12.0", "numpy": "1.26.4",
    "scipy": "1.13.1", "scikit-learn": "1.6.1", "umap-learn": "0.5.9.post2",
})
SOURCE_SHA256 = "721ca6c46b976447c82e78c8002ab28e84fe4e397de5e8e9701a9d070051bf5a"
INIT_SHA256 = "0741a4417462027880fe163e7ad9fef1606840c2035e0d42e2b554e4f3c9c0b2"
WHEEL_SHA256 = "7889f8d01485fabf37f7425685a7ec5e61d7e5e00eb16c5411b65ccb153d3f8a"
UPSTREAM_DEFAULTS = MappingProxyType({
    "true_label": None, "dist_std_local": 3, "jac_std_global": "median",
    "keep_all_local_dist": "auto", "too_big_factor": 0.4, "small_pop": 10,
    "jac_weighted_edges": True, "knn": 30, "n_iter_leiden": 5,
    "random_seed": 42, "num_threads": -1, "distance": "l2",
    "time_smallpop": 15, "partition_type": "ModularityVP",
    "resolution_parameter": 1.0, "knn_struct": None, "neighbor_graph": None,
    "hnsw_param_ef_construction": 150,
})
LICENSE_SOURCES = MappingProxyType({
    "parc": ("MIT", "https://github.com/ShobiStassen/PARC/blob/master/LICENSE"),
    "hnswlib": ("Apache-2.0", "https://github.com/nmslib/hnswlib/blob/v0.8.0/LICENSE"),
    "igraph": ("GPL-2.0-or-later", "https://github.com/igraph/python-igraph/blob/1.0.0/LICENSE"),
    "leidenalg": ("GPL-3.0-or-later", "https://github.com/vtraag/leidenalg/blob/0.12.0/LICENSE"),
})


class UnsafePARCError(RuntimeError):
    """No partition was produced; metadata preserves the rejected run's audit."""

    def __init__(self, reason, metadata):
        super().__init__(reason)
        self.metadata = metadata


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _preset(seed):
    params = dict(UPSTREAM_DEFAULTS)
    del params["true_label"]  # Not even a dummy gold argument is passed.
    params.update(distance="cosine", num_threads=1, random_seed=int(seed))
    return params


def _validated_input(vectors, seed, max_samples):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 4:
        raise ValueError("PARC seed must be an integer in 0..4")
    if (isinstance(max_samples, bool) or not isinstance(max_samples, Integral)
            or not 30 <= max_samples <= 5000):
        raise ValueError("PARC max_samples must be an integer in [30, 5000]")
    x, identity = _input(vectors, seed, max_samples)
    if len(x) < 30:
        raise ValueError("PARC requires at least 30 samples for fixed knn=30")
    norms = np.linalg.norm(x.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("PARC requires caller-owned L2-normalized rows; no normalization is performed")
    raw = np.asarray(vectors)
    identity.update(source_vectors_sha256=_sha256(raw.tobytes(order="C")),
                    normalization="caller L2-normalized; checked, not transformed",
                    normalization_rtol=1e-5, normalization_atol=1e-6)
    return x, identity


def _dependency_audit():
    """Include active transitive requirements and bundled notices, not extras.

    Distribution METADATA/RECORD and native artifact hashes identify local
    builds. The PARC wheel omits LICENSE; its declared license and upstream
    source are retained explicitly, not represented as an installed file.
    """
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    pending = list(REQUIRED_VERSIONS) + ["threadpoolctl", "packaging"]
    result = {}
    while pending:
        name = canonicalize_name(pending.pop())
        if name in result:
            continue
        dist = importlib.metadata.distribution(name)
        requirements = []
        for text in dist.requires or []:
            req = Requirement(text)
            if req.marker is None or req.marker.evaluate({"extra": ""}):
                actual = importlib.metadata.version(req.name)
                if not req.specifier.contains(actual, prereleases=True):
                    raise RuntimeError(f"unsatisfied PARC dependency: {name} requires {req}, found {actual}")
                requirements.append(str(req))
                pending.append(req.name)
        evidence = []
        for entry in dist.files or []:
            text = str(entry)
            basename = Path(text).name.lower()
            is_notice = any(word in basename for word in ("license", "licence", "copying", "notice"))
            is_native = text.endswith((".so", ".dylib", ".dll", ".pyd"))
            if is_notice or is_native or basename in ("metadata", "record", "direct_url.json"):
                path = Path(dist.locate_file(entry))
                if path.is_file():
                    evidence.append({"path": text, "sha256": _sha256(path.read_bytes()),
                                     "kind": "notice" if is_notice else "native" if is_native else "distribution"})
        license_text = dist.metadata.get("License", "")
        known = LICENSE_SOURCES.get(name)
        result[name] = {
            "version": dist.version, "requires": sorted(requirements),
            "source": f"https://pypi.org/project/{name}/{dist.version}/",
            "license_expression": dist.metadata.get("License-Expression"),
            "license_classifiers": [v for v in dist.metadata.get_all("Classifier", [])
                                    if v.startswith("License ::")],
            "license_metadata_sha256": _sha256(license_text.encode()) if license_text else None,
            "license_metadata_short": license_text if len(license_text) <= 200 else "see METADATA",
            "audited_project_license": known[0] if known else None,
            "license_source": known[1] if known else None,
            "files": sorted(evidence, key=lambda item: item["path"]),
        }
    return dict(sorted(result.items()))


def _load_upstream():
    try:
        versions = {name: _version(name, required) for name, required in REQUIRED_VERSIONS.items()}
        from parc import PARC
        import igraph
        from threadpoolctl import threadpool_info, threadpool_limits
        dependencies = _dependency_audit()
    except (importlib.metadata.PackageNotFoundError, ImportError) as exc:
        pins = ", ".join(f"{name}=={version}" for name, version in REQUIRED_VERSIONS.items())
        raise ImportError(f"PARCBackend requires optional {pins}, threadpoolctl, packaging and "
                          "PARC's transitive dependencies; no automatic installation") from exc
    source = inspect.getsourcefile(PARC)
    if source is None:
        raise RuntimeError("PARC source is unavailable; cannot audit this build")
    path = Path(source)
    actual = _sha256(path.read_bytes())
    init_hash = _sha256((path.parent / "__init__.py").read_bytes())
    if actual != SOURCE_SHA256 or init_hash != INIT_SHA256:
        raise RuntimeError("PARC source hash mismatch; review preset before using a modified build")
    signature = inspect.signature(PARC)
    expected = ["data", *UPSTREAM_DEFAULTS]
    if (list(signature.parameters) != expected or any(
            signature.parameters[key].default != value for key, value in UPSTREAM_DEFAULTS.items())):
        raise RuntimeError("PARC constructor API drift; review preset before upgrading")
    versions.update({name: entry["version"] for name, entry in dependencies.items()})
    versions.update(python=platform.python_version(), igraph_c=igraph.__igraph_version__)
    source_identity = {
        "path": str(path), "sha256": actual, "init_sha256": init_hash,
        "release": "https://pypi.org/project/parc/0.40/",
        "repository": "https://github.com/ShobiStassen/PARC",
        "reference_wheel": "parc-0.40-py3-none-any.whl",
        "reference_wheel_sha256": WHEEL_SHA256,
        "verification": "installed Python source matches audited wheel; wheel archive identity not inferred",
    }
    return PARC, versions, source_identity, dependencies, threadpool_limits, threadpool_info


def _guarded_class(upstream, n, metadata):
    """Intercept the original Leiden call, not its algorithm or membership.

    Use a private globals dictionary, avoiding a process-wide monkeypatch of
    parc or leidenalg (which could affect concurrent callers).
    """
    original = upstream.run_subPARC
    leiden = original.__globals__["leidenalg"]

    def reject(reason):
        metadata.update(status="rejected", rejection_reason=reason)
        raise UnsafePARCError(reason, metadata)

    class CheckedLeiden:
        def __getattr__(self, name):
            return getattr(leiden, name)

        def find_partition(self, graph, *args, **kwargs):
            if "initial_partition" in metadata:
                reject("unexpected repeated Leiden call; no alternative partition will be selected")
            result = leiden.find_partition(graph, *args, **kwargs)
            labels = _partition(result.membership, n, {}).labels
            if np.any(labels < 0):
                reject("PARC Leiden membership must assign every row; noise is not valid here")
            _, counts = np.unique(labels, return_counts=True)
            edges = np.asarray(graph.get_edgelist(), dtype="<i8").reshape(-1, 2)
            weights = np.asarray(graph.es["weight"], dtype="<f8")
            metadata["initial_partition"] = {
                "cluster_sizes": sorted(counts.tolist()),
                "labels_sha256": _sha256(labels.astype("<i8").tobytes()),
            }
            metadata["graph"] = {
                "vertices": graph.vcount(), "edges": graph.ecount(),
                "ordered_edges_sha256": _sha256(edges.tobytes()),
                "ordered_weights_sha256": _sha256(weights.tobytes()),
            }
            reasons = []
            if counts.min() < 10:
                reasons.append("small_pop<10 would activate wall-clock-dependent fragment cleanup (time_smallpop=15)")
            if counts.max() > 0.4 * n:
                reasons.append("too_big_factor=0.4 would activate hardcoded L2 refinement without num_threads=1")
            if reasons:
                reject("; ".join(reasons))
            return result

    namespace = dict(original.__globals__, leidenalg=CheckedLeiden())
    guarded = FunctionType(original.__code__, namespace, original.__name__,
                           original.__defaults__, original.__closure__)
    guarded.__kwdefaults__ = original.__kwdefaults__

    class GuardedPARC(upstream):
        run_subPARC = guarded

        def run_toobig_subPARC(self, *args, **kwargs):
            reject("unsafe large-cluster refinement reached despite initial-partition guard")

    return GuardedPARC


@dataclass(frozen=True)
class PARCBackend:
    """Fixed, label-blind PARC preset; only seed and a lower resource cap vary.

    Requires 30 <= N <= max_samples <= 5000. Seeds are restricted to 0..4.
    Rejections carry metadata in UnsafePARCError; they are not noise labels or
    successful partitions. Single-thread numerical limits are context-scoped
    but process-wide: run concurrent fits in separate processes, not threads.
    A parent process owns the job timeout; no wall-clock algorithm budget is
    introduced by this wrapper. Cross-platform bitwise reproducibility is not
    claimed; compare recorded versions, native artifacts and graph hashes.
    """

    seed: int = 0
    max_samples: int = 5000

    def fit(self, vectors) -> Partition:
        total_start = time.perf_counter()
        x, identity = _validated_input(vectors, self.seed, self.max_samples)
        upstream, versions, source, dependencies, limits, pool_info = _load_upstream()
        params = _preset(self.seed)
        metadata = {
            "algorithm": "PARC", "implementation": "parc.PARC.run_PARC",
            "backend_protocol": BACKEND_PROTOCOL,
            "classification": "upstream wrapper with fail-closed branch guards",
            "seed": int(self.seed), "max_samples": int(self.max_samples),
            "params": params, "versions": versions, "source": source,
            "input": identity, "dependencies": dependencies,
            "license": "mixed dependency chain; not MIT-only",
            "license_scope": "installed active dependency closure and shipped bundled notices; not a repository license declaration",
            "parameter_policy": "fixed before evaluation; no gold, trueK, resolution or graph overrides",
            "gold_parameter_passed": False,
            "upstream_dummy_truth": "run_PARC synthesizes all-ones truth only for unused reporting",
            "effective_hnsw": {
                "space": "cosine", "num_threads": 1, "random_seed": 100,
                "random_seed_scope": "HNSW default; wrapper seed controls Leiden only",
                "M": 48 if x.shape[1] > 30 else 24,
                "ef": min(len(x) - 10, 500),
                "ef_construction_requested": min(len(x) - 10, 500),
                "ef_construction": max(min(len(x) - 10, 500),
                                       48 if x.shape[1] > 30 else 24),
                "knn": 30, "self_neighbor_policy": "unchanged upstream query and pruning",
            },
            "effective_keep_all_local_dist": False,
            "rejection_policy": "reject initial Leiden clusters <10 or >0.4N; report every rejected run",
            "thread_policy": "HNSW num_threads=1; threadpoolctl limits=1; concurrent jobs require separate processes",
            "noise_label": -1, "noise_policy": "upstream must assign all rows; no rejection relabeling or reassignment",
            "timing_scope": "wall clock; total includes validation, imports and provenance hashing",
            "reproducibility_scope": "pinned implementation on recorded platform; unsafe postprocessing rejected",
            "platform": platform.platform(), "machine": platform.machine(),
        }
        guarded = _guarded_class(upstream, len(x), metadata)
        start = time.perf_counter()
        try:
            with limits(limits=1):
                # All upstream imports occur before entering the limiting context.
                metadata["threadpools_during_fit"] = pool_info()
                if any(pool.get("num_threads") != 1 for pool in metadata["threadpools_during_fit"]):
                    raise UnsafePARCError("numerical thread pool could not be limited to one thread", metadata)
                model = guarded(x, **params)
                model.run_PARC()
            if "initial_partition" not in metadata:
                raise RuntimeError("upstream bypassed the mandatory Leiden safety guard")
            result = _partition(model.labels, len(x), metadata)
            if np.any(result.labels < 0):
                raise RuntimeError("PARC unexpectedly returned noise; no silent repair is permitted")
            if _sha256(x.tobytes()) != identity["fit_vectors_sha256"]:
                raise RuntimeError("upstream mutated fit vectors; input identity no longer matches")
            metadata.update(status="ok", labels_sha256=_sha256(result.labels.astype("<i8").tobytes()))
            return result
        except UnsafePARCError as exc:
            metadata.update(status="rejected", rejection_reason=str(exc))
            raise
        finally:
            metadata.update(clustering_seconds=time.perf_counter() - start,
                            total_seconds=time.perf_counter() - total_start)
