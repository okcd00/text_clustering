"""Research-only derived-state cache; the frozen v1 core remains unchanged.

Cache entries are successful ``at(tick)`` results for the exact immutable
statistics object, not an approximation based on invariant centers. Only
training nearest-neighbor scans and pruning use this cache. Snapshot/predict
and the public FadedStatistics fresh-object contract are unchanged.
"""

import math

from .denstream_faded import DenStreamFaded, _distance


ALGORITHM_ID = "denstream-river-faded-cf-cached-v1"


class CachedDenStreamFaded(DenStreamFaded):
    """Single-writer experimental alternative with O(MD) auxiliary storage.

    Entries retain their source object to prevent Python id reuse. They are
    dropped on tick changes and after every event if the source is no longer
    live. Failed events preserve learned state; only derived caches/counters
    may change. Neither readout nor prediction populates these caches.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._derived_tick = None
        self._derived = {}
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def cache_stats(self):
        return {"hits": self._cache_hits, "misses": self._cache_misses,
                "entries": len(self._derived)}

    def _aged_entry(self, source, time):
        if time != self._derived_tick:
            self._derived.clear()
            self._derived_tick = time
        key = id(source)
        entry = self._derived.get(key)
        if entry is not None and entry[0] is source:
            self._cache_hits += 1
            return entry
        # Never cache an exception or bypass moment validation, even when
        # pruning only needs the weight. Age directly from original moments.
        aged = source.at(time)
        entry = [source, aged, None]
        self._derived[key] = entry
        self._cache_misses += 1
        return entry

    def _nearest(self, point, clusters, time):
        best, best_distance = None, math.inf
        for micro_id in sorted(clusters):
            entry = self._aged_entry(clusters[micro_id], time)
            if entry[1].weight == 0:
                continue
            if entry[2] is None:
                entry[2] = entry[1].center()
            distance = _distance(point, entry[2])
            if distance < best_distance:
                best, best_distance = micro_id, distance
        return best

    def _prune(self, potential, outlier, time):
        if time <= 0 or time % self._time_period != 0:
            return
        for micro_id, mc in tuple(potential.items()):
            if self._aged_entry(mc, time)[1].weight < self.beta * self.mu:
                del potential[micro_id]
        rate = math.log(2.0) * self.decaying_factor
        denominator = -math.expm1(-rate * self._time_period)
        for micro_id, mc in tuple(outlier.items()):
            age = time - mc.creation_time + self._time_period
            xi = -math.expm1(-rate * age) / denominator
            if self._aged_entry(mc, time)[1].weight < xi:
                del outlier[micro_id]

    def learn_one(self, vector):
        try:
            return super().learn_one(vector)
        finally:
            # Also retire transient candidate entries after an exception.
            # No moment hashing: object identity lookup is O(1), not O(D).
            live = {id(mc) for pool in (self.p_micro_clusters,
                                       self.o_micro_clusters)
                    for mc in pool.values()}
            self._derived = {key: value for key, value in self._derived.items()
                             if key in live}
