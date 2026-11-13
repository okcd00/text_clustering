"""Research-only cached DenStream adapter with a distinct protocol identity.

Validation, normalized learning, snapshot lifecycle and queries are inherited
from DenStreamStreamBackend. Only core construction and protocol identity differ;
no factory registration or change to the frozen original adapter is required.
"""

from dataclasses import asdict
import hashlib
import json
import uuid

from .denstream_cached import ALGORITHM_ID, CachedDenStreamFaded
from .denstream_stream_backend import DenStreamStreamBackend


BACKEND_PROTOCOL = "denstream-faded-cf-cached-l2-snapshot-v1"


class CachedDenStreamStreamBackend(DenStreamStreamBackend):
    """Single-writer cached research core under the existing window contract.

    The parent constructor cannot inject a core, and its identity methods use
    module constants rather than class attributes. Override those three seams
    explicitly; inherit all numerical and window behavior without monkeypatching.
    """

    def __init__(self, **parameters):
        # Mirror the parent's adapter fields without constructing an unused core.
        self._model = CachedDenStreamFaded(**parameters)
        self._model_id = uuid.uuid4().hex
        self._cached_revision = None
        self._cached_snapshot = None

    def _metadata(self, snapshot):
        metadata = super()._metadata(snapshot)
        metadata["backend"] = ALGORITHM_ID
        metadata["protocol"] = BACKEND_PROTOCOL
        return metadata

    def state_fingerprint(self):
        # Repeat only the frozen state serialization: the parent's method binds
        # its original module protocol. Derived caches/counters are not learned
        # state, and must not change the fingerprint after a failed event/read.
        state = self._model._state
        value = {key: val for key, val in vars(state).items()
                 if key not in ("p_micro_clusters", "o_micro_clusters")}
        for name in ("p_micro_clusters", "o_micro_clusters"):
            value[name] = [[int(key), asdict(stats)]
                           for key, stats in sorted(getattr(state, name).items())]
        value["parameters"] = asdict(self._model._parameters)
        value["backend_protocol"] = BACKEND_PROTOCOL
        return hashlib.sha256(json.dumps(value, sort_keys=True,
                              allow_nan=False).encode()).hexdigest()
