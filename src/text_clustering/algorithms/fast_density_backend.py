"""Fixed, label-blind UMAP + fast-hdbscan engineering adaptation.

Only seed and the input-size safety limit are configurable. This is not stock
BERTopic and does not produce topic representations or reassign noise. The
min_samples=4 convention is based on the unweighted, low-dimensional Boruvka
path in fast-hdbscan 0.3.4 (query k=min_samples+1, including self).
"""
from dataclasses import dataclass
import hashlib
import inspect
import math
from pathlib import Path
import time

import numpy as np
from scipy.sparse import issparse

from .discovery_backends import _input, _partition, _version


# Full get_params schema audited against 0.3.4, including inactive defaults.
# A version string alone cannot detect a locally modified estimator's defaults.
_FAST_PARAMS = {
    'min_cluster_size': 15,
    'min_samples': 4,
    'cluster_selection_method': 'eom',
    'allow_single_cluster': False,
    'max_cluster_size': float('inf'),
    'cluster_selection_epsilon': 0.0,
    'cluster_selection_persistence': 0.0,
    'semi_supervised': False,
    'ss_algorithm': 'bc',
    'reproducible': True,
    'metric': 'euclidean',
    'algorithm': 'boruvka',
    'knn_k': None,
    'cannot_link': None,
    'validate_cannot_link': True,
    'cannot_link_groups': None,
    'metric_kwds': None,
}


def _json_safe(value):
    """Keep non-finite defaults explicit without emitting non-standard JSON."""
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError('metadata requires string dictionary keys')
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f'unsupported metadata value: {type(value).__name__}')


def _check_params(actual, expected, *, exact, name):
    """Fail before fitting if the audited schema or fixed settings drift."""
    if not isinstance(actual, dict):
        raise RuntimeError(f'{name} parameter schema drift: expected a dictionary')
    if exact and actual.keys() != expected.keys():
        raise RuntimeError(f'{name} parameter schema drift; review preset before upgrading')
    for key, value in expected.items():
        observed = actual.get(key)
        # Exact scalar types also prevent True from passing as 1, or an array
        # from triggering ambiguous equality. These presets use Python scalars.
        if key not in actual or type(observed) is not type(value) or observed != value:
            raise RuntimeError(f'{name} parameter drift at {key}; review preset before upgrading')


def _source_hashes(reducer_type, clusterer_type):
    """Hash actual loaded sources, never the separately downloaded audit copy."""
    cluster_source = Path(inspect.getfile(clusterer_type))
    sources = {
        'backend': Path(__file__),
        'umap.UMAP': Path(inspect.getfile(reducer_type)),
        'fast_hdbscan.HDBSCAN': cluster_source,
        'fast_hdbscan.boruvka': cluster_source.with_name('boruvka.py'),
    }
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sources.items()}


def _reduced(vectors, n):
    if issparse(vectors):
        raise RuntimeError('UMAP must return a finite dense (n_samples, 5) matrix')
    raw = np.asarray(vectors)
    if raw.shape != (n, 5) or raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all():
        raise RuntimeError('UMAP must return a finite dense (n_samples, 5) matrix')
    # Isolate the reducer's result from any in-place clusterer operations. No
    # extra normalization or dtype conversion is introduced in reduced space.
    z = np.array(raw, order='C', copy=True)
    return z, {
        'shape': list(z.shape),
        'source_dtype': raw.dtype.str,
        'fit_dtype': z.dtype.str,
        'fit_vectors_sha256': hashlib.sha256(z.tobytes()).hexdigest(),
        'fit_vector_bytes': z.nbytes,
        'normalization': 'none; UMAP output preserved',
    }


@dataclass(frozen=True)
class FastDensityBackend:
    """Fixed UMAP + fast-hdbscan 0.3.4; no K, gold, or tuning overrides."""

    seed: int = 0
    max_samples: int = 5000

    def fit(self, vectors):
        started = time.perf_counter()
        x, identity = _input(vectors, self.seed, self.max_samples)
        input_seconds = time.perf_counter() - started
        stage = time.perf_counter()
        versions = {p: _version(p, v) for p, v in (
            ('umap-learn', '0.5.9.post2'), ('fast-hdbscan', '0.3.4'))}
        from umap import UMAP
        from fast_hdbscan import HDBSCAN

        umap_preset = dict(n_neighbors=15, n_components=5, min_dist=0.0,
                           metric='cosine', random_state=int(self.seed),
                           n_jobs=1, low_memory=True)
        reducer = UMAP(**umap_preset)
        clusterer = HDBSCAN(min_cluster_size=15, min_samples=4,
                            cluster_selection_method='eom', metric='euclidean',
                            algorithm='boruvka', reproducible=True)
        umap_params = reducer.get_params(deep=False)
        fast_params = clusterer.get_params(deep=False)
        _check_params(umap_params, umap_preset, exact=False, name='UMAP')
        _check_params(fast_params, _FAST_PARAMS, exact=True, name='fast-hdbscan')
        params = _json_safe({'umap': umap_params, 'fast_hdbscan': fast_params})
        source_hashes = _source_hashes(UMAP, HDBSCAN)
        setup_seconds = time.perf_counter() - stage

        stage = time.perf_counter()
        reduced = reducer.fit_transform(x)
        reduction_seconds = time.perf_counter() - stage
        stage = time.perf_counter()
        z, reduced_identity = _reduced(reduced, len(x))
        reduced_validation_seconds = time.perf_counter() - stage
        stage = time.perf_counter()
        labels = clusterer.fit_predict(z)
        clustering_seconds = time.perf_counter() - stage
        result = _partition(labels, len(x), {})
        noise_count = int(np.count_nonzero(result.labels == -1))
        result.metadata = _json_safe(dict(
            algorithm='UMAP + fast HDBSCAN clustering core',
            implementation='umap.UMAP + fast_hdbscan.HDBSCAN',
            backend_protocol='umap-fast-hdbscan-fixed-density-v1',
            classification='engineering adaptation',
            seed=int(self.seed), versions=versions, params=params,
            input=identity, reduced=reduced_identity,
            implementation_sha256=source_hashes['fast_hdbscan.HDBSCAN'],
            source_hashes=source_hashes,
            reduction_seconds=reduction_seconds,
            clustering_seconds=clustering_seconds,
            cost_seconds=dict(input_validation=input_seconds, setup=setup_seconds,
                              reduction=reduction_seconds,
                              reduced_validation=reduced_validation_seconds,
                              clustering=clustering_seconds,
                              total=time.perf_counter() - started),
            cost_policy='wall clock; setup includes imports, version/params/source checks; '
                        'fit stages include any JIT compilation; matrix bytes are not peak memory',
            noise_label=-1, noise_policy='retain; no reassignment',
            noise_count=noise_count, noise_fraction=noise_count / len(x),
            cluster_count=len(set(result.labels.tolist()) - {-1}),
            parameter_policy='fixed before evaluation; no gold or K',
            min_samples_includes_self=False,
            min_samples_convention='unweighted 5-D euclidean Boruvka: query k=min_samples+1, '
                                   'including self; use fourth other neighbor',
            engineering_adaptations=[
                'BERTopicPipelineBackend UMAP preset with fast-hdbscan replacing hdbscan',
                'fixed min_cluster_size=15 and min_samples=4; not stock BERTopic defaults',
                'reproducible Boruvka on 5-D euclidean UMAP output',
            ],
            excluded_components=['embedding generation', 'c-TF-IDF', 'topic naming',
                                 'topic reduction', 'outlier reassignment'],
        ))
        return result


# Public integration name used by the factory; both names share the same
# immutable, seed/max_samples-only API and fixed protocol.
UMAPFastHDBSCANBackend = FastDensityBackend
