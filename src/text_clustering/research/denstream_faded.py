"""Research-only ``denstream-river-faded-cf-v1``; not native River.

Control-flow adaptation of river/cluster/denstream.py at River commit
4768de0579f977315e65cd23ee658534c1d62c26, source SHA-256
e242200e76f1e02965e5bf9481b3357bb4fefcc98f45f9cb9fa1ab06fbd3bd39.
Uses the independently written immutable FadedStatistics kernel. See
denstream_faded_readme.md for retained policies, repairs and numerical limits.
No River runtime dependency, factory registration, labels or formal results.

Upstream notice, retained here so Python source distributions and wheels carry
the complete notice with the adapted implementation:

BSD 3-Clause License

Copyright (c) 2020, the river developers
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from collections.abc import Mapping, Set
from dataclasses import dataclass
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Iterable, Optional, Tuple

from .faded_statistics import FadedStatistics


ALGORITHM_ID = "denstream-river-faded-cf-v1"
UPSTREAM_COMMIT = "4768de0579f977315e65cd23ee658534c1d62c26"
UPSTREAM_SOURCE_SHA256 = "e242200e76f1e02965e5bf9481b3357bb4fefcc98f45f9cb9fa1ab06fbd3bd39"


class MicroClusterLimitError(RuntimeError):
    """An event would exceed the live p+o cap; the event was not committed."""


def _real(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, not bool")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be representable as a float") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, not bool")
    return int(value)


def _vector(vector: Iterable[Real], dimension: Optional[int]) -> Tuple[float, ...]:
    if isinstance(vector, (Mapping, Set, str, bytes, bytearray)):
        raise ValueError("vector must be a dense ordered vector")
    try:
        point = tuple(_real(value, "coordinate") for value in vector)
    except TypeError as exc:
        raise ValueError("vector must be a dense ordered vector") from exc
    if not point or (dimension is not None and len(point) != dimension):
        raise ValueError("vector must have the model's positive dimension")
    return point


def _distance(left: Tuple[float, ...], right: Tuple[float, ...]) -> float:
    return _real(math.dist(left, right), "Euclidean distance")


@dataclass(frozen=True)
class _Parameters:
    decaying_factor: float
    beta: float
    mu: float
    epsilon: float
    n_samples_init: int
    stream_speed: int
    max_micro_clusters: int
    time_period: int


@dataclass(frozen=True)
class _State:
    timestamp: int
    n_samples_seen: int
    initialized: bool
    dimension: Optional[int]
    buffer: Tuple[Tuple[float, ...], ...]
    next_micro_cluster_id: int
    p_micro_clusters: Mapping
    o_micro_clusters: Mapping


@dataclass(frozen=True)
class DenStreamSnapshot:
    """Immutable readout. All statistics are at ``timestamp``.

    Maps are read-only, vectors/components are tuples, and FadedStatistics is
    frozen. Macro labels are contiguous, snapshot-local integers starting at 0.
    Persistent microcluster IDs are never reused or changed on promotion.
    """

    timestamp: int
    n_samples_seen: int
    initialized: bool
    dimension: Optional[int]
    buffer_size: int
    next_micro_cluster_id: int
    p_micro_clusters: Mapping[int, FadedStatistics]
    o_micro_clusters: Mapping[int, FadedStatistics]
    components: Tuple[Tuple[int, ...], ...]
    micro_to_macro: Mapping[int, int]
    macro_clusters: Mapping[int, FadedStatistics]

    @property
    def micro_clusters(self) -> Mapping[int, FadedStatistics]:
        """Both pools by globally unique ID (including zero-mass states)."""
        return MappingProxyType(dict(sorted({**self.p_micro_clusters,
                                            **self.o_micro_clusters}.items())))


class DenStreamFaded:
    """Dense streaming research candidate, using upstream natural event time.

    ``learn_one`` is transactional: every validation, radius calculation and
    pruning operation succeeds before a single state-pointer commit. There is
    no label/weight input, normalization, wall clock, cache or model training.
    ``predict_one`` assigns the nearest macrocenter without a distance cutoff;
    only uninitialized/empty readout returns -1. See the companion README.
    """

    def __init__(self, decaying_factor=.25, beta=.75, mu=2, epsilon=.02,
                 n_samples_init=1000, stream_speed=100, max_micro_clusters=10000):
        decaying_factor = _real(decaying_factor, "decaying_factor")
        beta = _real(beta, "beta")
        mu = _real(mu, "mu")
        epsilon = _real(epsilon, "epsilon")
        if decaying_factor <= 0 or not 0 < beta <= 1 or epsilon <= 0:
            raise ValueError("require decaying_factor>0, 0<beta<=1 and epsilon>0")
        threshold = _real(beta * mu, "beta * mu")
        if threshold <= 1:
            raise ValueError("beta * mu must be greater than 1")
        n_samples_init = _positive_int(n_samples_init, "n_samples_init")
        stream_speed = _positive_int(stream_speed, "stream_speed")
        max_micro_clusters = _positive_int(max_micro_clusters, "max_micro_clusters")
        # Natural logarithm, as upstream, NOT log2. log1p avoids cancellation
        # when threshold is large. Bound periods to exactly representable ticks.
        period = math.log1p(1.0 / (threshold - 1.0)) / decaying_factor
        if not math.isfinite(period) or not 0 < period <= 2**53:
            raise ValueError("pruning period is outside the supported numeric range")
        self._parameters = _Parameters(decaying_factor, beta, mu, epsilon,
                                       n_samples_init, stream_speed,
                                       max_micro_clusters, math.ceil(period))
        self._state = _State(-1, 0, False, None, (), 0,
                             MappingProxyType({}), MappingProxyType({}))

    @property
    def decaying_factor(self):
        return self._parameters.decaying_factor

    @property
    def beta(self):
        return self._parameters.beta

    @property
    def mu(self):
        return self._parameters.mu

    @property
    def epsilon(self):
        return self._parameters.epsilon

    @property
    def n_samples_init(self):
        return self._parameters.n_samples_init

    @property
    def stream_speed(self):
        return self._parameters.stream_speed

    @property
    def max_micro_clusters(self):
        return self._parameters.max_micro_clusters

    @property
    def _time_period(self):
        return self._parameters.time_period

    @property
    def timestamp(self):
        return self._state.timestamp

    @property
    def initialized(self):
        return self._state.initialized

    @property
    def n_samples_seen(self):
        return self._state.n_samples_seen

    @property
    def _n_samples_seen(self):
        return self._state.n_samples_seen

    @property
    def dimension(self):
        return self._state.dimension

    @property
    def next_micro_cluster_id(self):
        return self._state.next_micro_cluster_id

    @property
    def p_micro_clusters(self) -> Mapping[int, FadedStatistics]:
        """Persistent moments, at their own last update/reference times."""
        return self._state.p_micro_clusters

    @property
    def o_micro_clusters(self) -> Mapping[int, FadedStatistics]:
        return self._state.o_micro_clusters

    def _check_capacity(self, potential, outlier):
        # Applies before allocation, even if this event would subsequently prune.
        if len(potential) + len(outlier) >= self.max_micro_clusters:
            raise MicroClusterLimitError(
                f"creating a microcluster would exceed max_micro_clusters="
                f"{self.max_micro_clusters}; learning event not committed")

    def _initialize(self, buffer, time, next_id):
        potential = {}
        covered = set()
        for i, point in enumerate(buffer):
            if i in covered:
                continue
            # Upstream excludes the seed before counting uncovered neighbors.
            neighbors = [j for j, other in enumerate(buffer)
                         if j != i and j not in covered
                         and _distance(point, other) < self.epsilon]
            if len(neighbors) > self.mu:
                self._check_capacity(potential, {})
                mc = FadedStatistics.empty(len(point), self.decaying_factor, time)
                mc = mc.insert(point, time)
                for j in neighbors:
                    mc = mc.insert(buffer[j], time)
                potential[next_id] = mc
                next_id += 1
                covered.update((i, *neighbors))
        # Deliberately no recursive DBSCAN expansion, no outlier retention, and
        # no retroactive aging from the individual buffer arrival times.
        return potential, next_id

    @staticmethod
    def _nearest(point, clusters, time):
        best = None
        best_distance = math.inf
        for micro_id in sorted(clusters):
            mc = clusters[micro_id].at(time)
            if mc.weight == 0:
                continue  # Complete underflow has no defined center.
            distance = _distance(point, mc.center())
            if distance < best_distance:
                best, best_distance = micro_id, distance
        return best

    def _merge(self, point, singleton, potential, outlier, time, next_id):
        # Only the nearest member of each pool is considered, p before o.
        for pool in (potential, outlier):
            micro_id = self._nearest(point, pool, time)
            if micro_id is None:
                continue
            candidate = pool[micro_id].candidate(point, time)
            if candidate.radius() <= self.epsilon:
                # The accepted object IS the tested candidate: no stale-time
                # second insert and no double insertion during promotion.
                if pool is outlier and candidate.weight > self.beta * self.mu:
                    del outlier[micro_id]
                    potential[micro_id] = candidate
                else:
                    pool[micro_id] = candidate
                return next_id
        self._check_capacity(potential, outlier)
        outlier[next_id] = singleton
        return next_id + 1

    def _prune(self, potential, outlier, time):
        if time <= 0 or time % self._time_period != 0:
            return
        for micro_id, mc in tuple(potential.items()):
            if mc.at(time).weight < self.beta * self.mu:
                del potential[micro_id]
        # expm1 evaluates (1 - 2**(-lambda * age)) without cancellation.
        rate = math.log(2.0) * self.decaying_factor
        denominator = -math.expm1(-rate * self._time_period)
        for micro_id, mc in tuple(outlier.items()):
            age = time - mc.creation_time + self._time_period
            xi = -math.expm1(-rate * age) / denominator
            if mc.at(time).weight < xi:
                del outlier[micro_id]

    def learn_one(self, vector: Iterable[Real]) -> "DenStreamFaded":
        """Consume one unlabeled vector; failed events leave all state intact."""
        old = self._state
        point = _vector(vector, old.dimension)
        count = old.n_samples_seen + 1
        time = count // self.stream_speed - 1
        if time > 2**53:
            raise ValueError("event timestamp is not an exactly representable tick")
        # Validate squares/moments even for an initialization point that might
        # later be discarded. Never defer malformed input until initialization.
        singleton = FadedStatistics.empty(len(point), self.decaying_factor, time)
        singleton = singleton.insert(point, time)
        potential = dict(old.p_micro_clusters)
        outlier = dict(old.o_micro_clusters)
        next_id = old.next_micro_cluster_id
        initialized = old.initialized
        buffer = old.buffer
        if not initialized:
            buffer = (*buffer, point)
            if len(buffer) == self.n_samples_init:
                potential, next_id = self._initialize(buffer, time, next_id)
                buffer, initialized = (), True
            # Upstream returns before periodic pruning on the init event.
        else:
            next_id = self._merge(point, singleton, potential, outlier, time, next_id)
            # Upstream runs this on EVERY event at a qualifying tick, not merely
            # on the first event advancing the clock to that tick.
            self._prune(potential, outlier, time)
        self._state = _State(time, count, initialized, len(point), buffer, next_id,
                             MappingProxyType(potential), MappingProxyType(outlier))
        return self

    def snapshot(self) -> DenStreamSnapshot:
        """Pure current-time graph readout; no pruning, clock ticks or caches."""
        state = self._state
        potential = {i: mc.at(state.timestamp)
                     for i, mc in sorted(state.p_micro_clusters.items())}
        outlier = {i: mc.at(state.timestamp)
                   for i, mc in sorted(state.o_micro_clusters.items())}
        centers = {i: mc.center() for i, mc in potential.items() if mc.weight > 0}
        radii = {i: potential[i].radius() for i in centers}
        # Every positive-mass p node belongs to a component. Weight<=mu does
        # not remove a node: upstream makes it an isolated macrocluster.
        remaining = set(centers)
        components = []
        micro_to_macro = {}
        macro_clusters = {}
        while remaining:
            seed = min(remaining)
            remaining.remove(seed)
            stack, members = [seed], []
            while stack:
                current = stack.pop()
                members.append(current)
                if potential[current].weight <= self.mu:
                    continue
                neighbors = []
                for other in sorted(remaining):
                    if potential[other].weight <= self.mu:
                        continue
                    distance = _distance(centers[current], centers[other])
                    if (distance < 2 * self.epsilon
                            and distance <= radii[current] + radii[other]):
                        neighbors.append(other)
                remaining.difference_update(neighbors)
                stack.extend(reversed(neighbors))
            component = tuple(sorted(members))
            label = len(components)
            components.append(component)
            aggregate = potential[component[0]]
            for micro_id in component[1:]:
                aggregate = aggregate.merge(potential[micro_id], state.timestamp)
            macro_clusters[label] = aggregate
            micro_to_macro.update((i, label) for i in component)
        return DenStreamSnapshot(
            state.timestamp, state.n_samples_seen, state.initialized,
            state.dimension, len(state.buffer), state.next_micro_cluster_id,
            MappingProxyType(potential), MappingProxyType(outlier),
            tuple(components), MappingProxyType(dict(sorted(micro_to_macro.items()))),
            MappingProxyType(macro_clusters))

    def predict_one(self, vector: Iterable[Real]) -> int:
        """Nearest macrocenter, lower label on ties; -1 only for empty readout.

        Queries are validated even when empty, but never establish dimension,
        learn, normalize, geometrically reject, or advance the event clock.
        """
        point = _vector(vector, self.dimension)
        if not self.initialized:
            return -1
        snapshot = self.snapshot()
        best, best_distance = -1, math.inf
        for label, mc in snapshot.macro_clusters.items():
            distance = _distance(point, mc.center())
            if distance < best_distance:
                best, best_distance = label, distance
        return best
