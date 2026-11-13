"""Pinned River microcluster adapters. Native algorithms, not paper reproductions.

Frozen prediction runs on a detached model. DenStream's empty potential state
is rejected (-1), never interpreted as cluster zero. Native event-count decay
is retained; calendar time only defines evaluation windows.
"""
from copy import deepcopy
import hashlib
import importlib.metadata
import pickle
import time

import numpy as np
from .river_stream_backend import StreamPrediction, _numeric_array, _normalize_rows


class RiverMicroStream:
    def __init__(self, *, method, **parameters):
        if importlib.metadata.version('river') != '0.26.1':
            raise RuntimeError('requires river==0.26.1')
        from river import cluster
        if method not in ('clustream', 'denstream'):
            raise ValueError('unknown method')
        self.method = method
        self.model = (cluster.CluStream if method == 'clustream' else cluster.DenStream)(**parameters)
        self.parameters = parameters
        self.state_revision = 0
        self.dimension = None
        self.timings = dict(learn_seconds=0., predict_seconds=0., fingerprint_seconds=0.)

    def _rows(self, values):
        x = _numeric_array(values)
        if x.ndim != 2 or not x.shape[1] or not np.isfinite(x).all() or np.any(np.linalg.norm(x, axis=1) == 0):
            raise ValueError('finite nonzero matrix required')
        if self.dimension is not None and x.shape[1] != self.dimension:
            raise ValueError('dimension mismatch')
        return _normalize_rows(x)

    def learn_one(self, vector):
        start = time.perf_counter()
        x = self._rows(np.asarray(vector)[None, :])[0]
        self.model.learn_one(dict(enumerate(x)))
        self.dimension = len(x)
        self.state_revision += 1
        self.timings['learn_seconds'] += time.perf_counter()-start
        return self

    @property
    def metadata(self):
        m = self.model
        counts = ({'micro_clusters': len(m.micro_clusters)} if self.method == 'clustream' else
                  {'potential_micro_clusters': len(m.p_micro_clusters),
                   'outlier_micro_clusters': len(m.o_micro_clusters)})
        return dict(backend='river.'+self.method, version='0.26.1', parameters=self.parameters,
                    state_revision=self.state_revision, dimension=self.dimension,
                    state_counts=counts, serialized_state_bytes=len(pickle.dumps(m, protocol=5)),
                    label_scope='snapshot-local', clock='native event count, not elapsed calendar time')

    def state_fingerprint(self):
        start = time.perf_counter()
        value = hashlib.sha256(pickle.dumps((self.model, self.dimension, self.state_revision), protocol=5)).hexdigest()
        self.timings['fingerprint_seconds'] += time.perf_counter()-start
        return value

    def predict_window(self, vectors):
        start = time.perf_counter()
        x = self._rows(vectors)
        frozen = deepcopy(self.model)
        usable = (frozen._kmeans_mc is not None if self.method == 'clustream'
                  else frozen.initialized and bool(frozen.p_micro_clusters))
        labels = [frozen.predict_one(dict(enumerate(row))) if usable else -1 for row in x]
        labels = np.array([-1 if v is None else v for v in labels], dtype=np.int64)
        self.timings['predict_seconds'] += time.perf_counter()-start
        return StreamPrediction(labels=labels, metadata=self.metadata)
