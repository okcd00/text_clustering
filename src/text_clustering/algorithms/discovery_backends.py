"""Label-blind, fixed-preset discovery pipelines over caller-owned embeddings.

BERTopic clustering core only: no topic representation or outlier reassignment.
Optional dependencies are imported lazily. No K, gold, or layer override API.
"""
from dataclasses import dataclass
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
from numbers import Integral
import time

import numpy as np
from scipy.sparse import issparse
from ..types import Partition


def _input(vectors, seed, max_samples):
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    if isinstance(max_samples, bool) or not isinstance(max_samples, Integral) or max_samples < 16:
        raise ValueError('max_samples must be an integer >= 16')
    if issparse(vectors):
        raise ValueError('dense real embeddings required')
    raw = np.asarray(vectors)
    if raw.ndim != 2 or raw.dtype.kind not in 'fiu' or raw.shape[1] == 0:
        raise ValueError('dense 2-D real embeddings required')
    if not 16 <= len(raw) <= max_samples:
        raise ValueError('sample count outside [16, max_samples]')
    x = np.array(raw, dtype=np.float32, order='C', copy=True)
    if not np.isfinite(x).all() or np.any(np.linalg.norm(x, axis=1) == 0):
        raise ValueError('finite nonzero vectors required for cosine geometry')
    return x, dict(shape=list(x.shape), source_dtype=raw.dtype.str, fit_dtype=x.dtype.str,
                   fit_vectors_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                   fit_vector_bytes=x.nbytes, normalization='none; caller owns preprocessing')


def _version(name, required):
    actual = importlib.metadata.version(name)
    if actual != required:
        raise RuntimeError(f'{name} requires version {required}, found {actual}; review preset before upgrading')
    return actual


def _partition(labels, n, metadata):
    labels = np.asarray(labels)
    if labels.shape != (n,) or labels.dtype.kind not in 'iu' or np.any(labels < -1):
        raise RuntimeError('invalid cluster labels; only integer IDs and -1 noise allowed')
    return Partition(labels, metadata)


@dataclass(frozen=True)
class BERTopicPipelineBackend:
    """UMAP + external HDBSCAN, fixed matched-density preset v1.

    External min_samples=4 matches sklearn min_samples=5 (self included).
    This is a controlled clustering-core adaptation, NOT stock BERTopic defaults.
    """
    seed: int = 0
    max_samples: int = 5000

    def fit(self, vectors):
        x, identity = _input(vectors, self.seed, self.max_samples)
        versions = {p: _version(p, v) for p, v in [('umap-learn','0.5.9.post2'), ('hdbscan','0.8.40')]}
        from umap import UMAP
        from hdbscan import HDBSCAN
        reducer = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                       metric='cosine', random_state=int(self.seed), n_jobs=1, low_memory=True)
        clusterer = HDBSCAN(min_cluster_size=15, min_samples=4, metric='euclidean',
                            cluster_selection_method='eom', core_dist_n_jobs=1,
                            prediction_data=False, approx_min_span_tree=False)
        # Snapshot before UMAP mutates effective fit parameters internally.
        import copy
        requested_umap_params = copy.deepcopy(reducer.get_params())
        start = time.perf_counter()
        z = reducer.fit_transform(x)
        reduction_seconds = time.perf_counter()-start
        start = time.perf_counter()
        labels = clusterer.fit_predict(z)
        cluster_seconds = time.perf_counter()-start
        cluster_params = clusterer.get_params()
        cluster_params['memory'] = {'location': cluster_params['memory'].location}
        # JSON has no numeric Infinity; retain the upstream default explicitly.
        if np.isposinf(cluster_params['cluster_selection_epsilon_max']):
            cluster_params['cluster_selection_epsilon_max'] = 'Infinity'
        return _partition(labels, len(x), dict(algorithm='BERTopic clustering core',
            implementation='umap.UMAP + hdbscan.HDBSCAN', backend_protocol='bertopic-matched-density-v1',
            classification='core adaptation', seed=int(self.seed), versions=versions,
            params={'umap': reducer.get_params(), 'hdbscan': cluster_params},
            requested_umap_params=requested_umap_params,
            source_hashes={name: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
                           for name, cls in [('umap.UMAP', UMAP), ('hdbscan.HDBSCAN', HDBSCAN)]},
            input=identity, reduction_seconds=reduction_seconds, clustering_seconds=cluster_seconds,
            reduced_vectors_sha256=hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest(),
            reduced_shape=list(z.shape), reduced_dtype=z.dtype.str,
            noise_label=-1, noise_policy='retain; no reassignment',
            parameter_policy='fixed before evaluation; no gold or K',
            excluded_components=['embedding generation','c-TF-IDF','topic naming','topic reduction'],
            min_samples_includes_self=False))


@dataclass(frozen=True)
class EVoCBackend:
    """Pinned upstream default selection; gold and approximate K are never passed."""
    seed: int = 0
    max_samples: int = 5000

    def fit(self, vectors):
        x, identity = _input(vectors, self.seed, self.max_samples)
        version = _version('evoc', '0.3.2')
        from evoc import EVoC
        model = EVoC(random_state=int(self.seed), base_n_clusters=None, approx_n_clusters=None)
        labels = model.fit_predict(x)
        matches = [i for i, layer in enumerate(model.cluster_layers_) if np.array_equal(layer, labels)]
        if not matches:
            raise RuntimeError('upstream labels do not match any layer')
        return _partition(labels, len(x), dict(algorithm='EVoC', implementation='evoc.EVoC',
            backend_protocol='evoc-default-labelblind-v1', classification='upstream library pipeline',
            seed=int(self.seed), versions={'evoc':version}, params=model.get_params(), input=identity,
            implementation_sha256=hashlib.sha256(Path(inspect.getfile(EVoC)).read_bytes()).hexdigest(),
            selection_policy='upstream labels_; no gold-scored layer search; approx/base K unset',
            selected_layer_matches=matches,
            layer_cluster_counts=[len(set(l)-{-1}) for l in model.cluster_layers_],
            layer_coverages=[float(np.mean(l>=0)) for l in model.cluster_layers_],
            persistence_scores=[float(p) for p in model.persistence_scores_],
            noise_label=-1, noise_policy='retain; no reassignment'))
