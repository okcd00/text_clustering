"""Unlabelled, single-writer maintenance wrappers for the audited BIRCH backend.

capacity_freeze bounds leaf CF count, not bytes/internal nodes, and permanently
stops learning at the limit (even inputs that could merge are ignored).
rolling_rebuild learns between rebuilds, so old samples persist until the next
rebuild. It is NOT a hard CF cap or an exact sliding-window model between ticks.
A rebuild temporarily holds both trees; reported CF counts are not memory bytes.
No continuation checkpoint/copy support; native failures require discarding us.
"""
from collections import deque
from numbers import Integral
import uuid
import time

from .birch_stream import BirchStreamBackend, _graph_fingerprint
from .river_stream_backend import StreamPrediction, _numeric_array, _normalize_rows


def _positive(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class BirchMaintenance:
    """Same learn_one/predict_window interface; state_revision counts arrivals.

    model_revision counts arrivals that changed the active model, not replayed
    rows. Snapshot IDs use reception revisions, including while frozen. Native
    model IDs/revisions are exposed separately; labels remain snapshot-local.
    """

    def __init__(self, *, policy, max_cf=None, window_size=None,
                 rebuild_every=None, threshold=0.5, branching_factor=50):
        if policy not in ("capacity_freeze", "rolling_rebuild"):
            raise ValueError("Unknown maintenance policy")
        if policy == "capacity_freeze":
            max_cf = _positive("max_cf", max_cf)
            if window_size is not None or rebuild_every is not None:
                raise ValueError("Window options require rolling_rebuild")
        else:
            window_size = _positive("window_size", window_size)
            rebuild_every = _positive("rebuild_every", rebuild_every)
            if max_cf is not None:
                raise ValueError("rolling_rebuild does not provide a hard CF cap")
        self._config = dict(policy=policy, max_cf=max_cf, window_size=window_size,
                            rebuild_every=rebuild_every, threshold=threshold,
                            branching_factor=branching_factor)
        self._backend = self._fresh()
        self._window = deque(maxlen=window_size) if window_size else None
        self._revision = self._model_revision = self._rebuilds = 0
        self._peak_cf = self._peak_live_cf = 0
        self._frozen = False
        self._failure = None
        self._id = uuid.uuid4().hex
        self._retired_timings = dict.fromkeys(self._backend.timings, 0.0)
        self._maintenance_seconds = self._fingerprint_seconds = 0.0

    @property
    def timings(self):
        """Cumulative seconds, detached from state and fingerprints.

        maintenance_seconds is rebuild wall time INCLUDING native_fit_seconds;
        these overlap. fingerprint_seconds is wrapper wall time INCLUDING the
        nested native fingerprint, counted once rather than summed twice.
        """
        active = self._backend.timings
        total = {key: value + active[key]
                 for key, value in self._retired_timings.items()}
        total["fingerprint_seconds"] = self._fingerprint_seconds
        total["maintenance_seconds"] = self._maintenance_seconds
        return total

    def _retire_timings(self, backend):
        for key, value in backend.timings.items():
            self._retired_timings[key] += value

    def _rebuild(self):
        started = time.perf_counter()
        fresh = None
        try:
            fresh = self._fresh()
            old_cf = self._backend.metadata["n_micro_clusters"]
            for sample in self._window:
                fresh.learn_one(sample)
                self._observe(fresh, old_cf)
            self._retire_timings(self._backend)
            self._backend = fresh
            self._rebuilds += 1
        finally:
            if fresh is not None and fresh is not self._backend:
                self._retire_timings(fresh)
            self._maintenance_seconds += time.perf_counter() - started

    def _fresh(self):
        return BirchStreamBackend(threshold=self._config["threshold"],
                                  branching_factor=self._config["branching_factor"])

    def __copy__(self):
        raise TypeError("Maintenance copying is unsupported")

    def __deepcopy__(self, memo):
        raise TypeError("Maintenance copying is unsupported")

    def __reduce_ex__(self, protocol):
        raise TypeError("Maintenance checkpoints are unsupported")

    @property
    def state_revision(self):
        return self._revision

    def _require_usable(self):
        if self._failure is not None:
            raise RuntimeError("Maintenance unusable after learning failure; discard instance")
        self._backend._require_usable()

    def _observe(self, backend, old_cf=0):
        count = backend.metadata["n_micro_clusters"]
        self._peak_cf = max(self._peak_cf, count)
        self._peak_live_cf = max(self._peak_live_cf, old_cf + count)
        return count

    def learn_one(self, vector):
        self._require_usable()
        row = _numeric_array(vector)
        if row.ndim != 1:
            raise ValueError("learn_one requires one 1D vector")
        self._backend._check_dimension(len(row))
        _normalize_rows(row[None, :])  # Validate zero vectors even while frozen.
        if self._frozen:
            self._revision += 1
            return self
        try:
            if self._window is not None:
                self._window.append(row)
            if (self._window is not None and
                    (self._revision + 1) % self._config["rebuild_every"] == 0):
                self._rebuild()
            else:
                self._backend.learn_one(row)
            count = self._observe(self._backend)
            # Audited rowwise BIRCH adds at most one leaf CF per arrival.
            if self._config["policy"] == "capacity_freeze":
                if count > self._config["max_cf"]:
                    raise RuntimeError("Native BIRCH violated rowwise CF bound")
                self._frozen = count == self._config["max_cf"]
            self._revision += 1
            self._model_revision += 1
        except BaseException as exc:
            self._failure = type(exc).__name__
            raise
        return self

    @property
    def metadata(self):
        native = self._backend.metadata
        return {**native, "maintenance_policy": dict(self._config),
                "native_model_id": native["model_id"],
                "native_state_revision": native["state_revision"],
                "model_id": self._id, "state_revision": self._revision,
                "snapshot_id": f"{self._id}:{self._revision}",
                "model_revision": self._model_revision,
                "revision_semantics": "validated arrivals including frozen no-op learning",
                "frozen": self._frozen, "rebuild_count": self._rebuilds,
                "window_count": len(self._window) if self._window is not None else 0,
                "peak_cf_count": self._peak_cf,
                "peak_live_cf_count": self._peak_live_cf,
                "peak_semantics": "observed after each row; live includes both rebuild trees",
                "hard_cf_limit": self._config["max_cf"],
                "usable": native["usable"] and self._failure is None,
                "learning_failure": self._failure or native["learning_failure"]}

    def state_fingerprint(self):
        # Explicit logical payload: no UUIDs, wall-clock values or timing metrics.
        started = time.perf_counter()
        try:
            return self._logical_fingerprint()
        finally:
            self._fingerprint_seconds += time.perf_counter() - started

    def _logical_fingerprint(self):
        return _graph_fingerprint(dict(config=self._config,
            backend=self._backend.state_fingerprint(), revision=self._revision,
            model_revision=self._model_revision, rebuilds=self._rebuilds,
            window=list(self._window) if self._window is not None else None,
            frozen=self._frozen, failure=self._failure,
            peak_cf=self._peak_cf, peak_live_cf=self._peak_live_cf))

    def predict_window(self, vectors):
        self._require_usable()
        result = self._backend.predict_window(vectors)
        return StreamPrediction(labels=result.labels, metadata=self.metadata)

    predict_many = predict_window


def capacity_freeze(max_cf, **backend_options):
    """Construct a permanent leaf-CF-cap freeze policy."""
    return BirchMaintenance(policy="capacity_freeze", max_cf=max_cf, **backend_options)


def rolling_rebuild(window_size, rebuild_every, **backend_options):
    """Rebuild on every rebuild_every-th valid arrival using its latest window."""
    return BirchMaintenance(policy="rolling_rebuild", window_size=window_size,
                            rebuild_every=rebuild_every, **backend_options)
