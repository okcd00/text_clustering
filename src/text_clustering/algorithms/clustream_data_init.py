"""Explicit CluStream macro-readout adaptation; native microcluster updates retained.

The published River adapter remains unchanged. At its existing macro refresh
clock, replace Gaussian online KMeans initialization with data-seeded batch
KMeans on the same (unweighted) microcenters. No labels or future vectors enter.
This changes both initialization and macro optimization, not a paper reproduction.
"""
import time
import numpy as np
from .river_micro_stream import RiverMicroStream


class CluStreamDataInit(RiverMicroStream):
    def __init__(self, *, n_macro_clusters=5, seed=0, **parameters):
        super().__init__(method='clustream', n_macro_clusters=n_macro_clusters,
                         seed=seed, **parameters)
        self.timings['macro_replacement_seconds'] = 0.

    def learn_one(self, vector):
        super().learn_one(vector)
        m = self.model
        if m._kmeans_mc is not None and m._timestamp % m.time_gap == m.time_gap - 1:
            from sklearn.cluster import KMeans
            start = time.perf_counter()
            centers = np.array([[center.get(j, 0.) for j in range(self.dimension)]
                                for center in m._mc_centers.values()])
            fitted = KMeans(n_clusters=m.n_macro_clusters, init='k-means++',
                            n_init=10, random_state=m.seed).fit(centers)
            # River predict_one uses these centers; micro learning never reads them.
            m._kmeans_mc.centers = {i: dict(enumerate(row))
                                   for i, row in enumerate(fitted.cluster_centers_)}
            m.centers = m._kmeans_mc.centers
            elapsed = time.perf_counter() - start
            self.timings['macro_replacement_seconds'] += elapsed
            self.timings['learn_seconds'] += elapsed
        return self

    @property
    def metadata(self):
        value = super().metadata
        value.update(backend='clustream_data_init_v1',
                     macro_policy='unweighted microcenters; k-means++; n_init=10; native refresh clock',
                     evidence_level='engineering adaptation; not upstream reproduction')
        return value
