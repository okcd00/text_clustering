"""Read-only effective-window prediction on pinned River 0.26.1.

Learning is inherited unchanged. Prediction maintains two disposable boundary
copies once and searches the effective copy with native distance arithmetic.
This is a separate protocol, not a claim of equivalence to copy-native-v1.
Invariance is checked at window boundaries, including query exceptions, not
per query. Mutations are detected, not rolled back. Single-writer only; River
is imported lazily by the inherited pinned constructor.
"""
from collections.abc import Mapping
import time

import numpy as np

from text_clustering.algorithms.textclust import (
    TextClustBackend, TextClustPrediction, fingerprint_textclust, validate_tf,
)

ADAPTER = "effective-snapshot-native-distance-v1"
BACKEND_PROTOCOL = "river_textclust_effective_snapshot_v1"


def _predict_window(backend: TextClustBackend, tfs) -> TextClustPrediction:
    """Predict a window without learning or changing backend state.

    Accept an existing pinned TextClustBackend. The caller owns learning and
    synchronization; concurrent learning during this call is unsupported.
    """
    if isinstance(tfs, (str, bytes, Mapping)):
        raise ValueError("predict_window requires an iterable of sparse TF mappings")
    rows = [validate_tf(tf) for tf in tfs]
    start = time.perf_counter()
    base, effective, idf, metadata = backend._make_window_snapshot()
    labels, diagnostics, query_costs = [], [], []
    try:
        for tf in rows:
            stamp = time.perf_counter()
            closest, best = None, float("inf")
            if tf:
                # Exactly River.predict_one's ordered TF wrapping and
                # get_assignment's temporary microcluster construction.
                query = effective.MicroCluster(
                    {k: {"tf": v} for k, v in tf.items()},
                    1, 1, effective.realtime, None)
                for key, micro in effective.micro_clusters.items():
                    if micro.weight > effective.min_weight:
                        distance = effective._micro_distance.dist(query, micro, idf)
                        if distance < best:
                            best, closest = distance, key
            distance_seconds = time.perf_counter() - stamp
            empty = not tf
            no_effective = not effective.micro_clusters
            no_eligible = metadata["n_micro_eligible"] == 0
            reason = ("empty_tf" if empty else "no_effective_micro" if no_effective
                      else "no_eligible_micro" if no_eligible else "assigned")
            if closest is None and reason == "assigned":
                raise RuntimeError("unexpected_none: native micro prediction has eligible state")
            labels.append(-1 if closest is None else int(closest))
            diagnostics.append(dict(
                empty_tf=empty, no_live_micro=not base.micro_clusters,
                no_effective_micro=no_effective, no_eligible_micro=no_eligible,
                all_oov=bool(tf) and all(k not in idf for k in tf), reason=reason))
            query_costs.append(dict(native_distance_search_seconds=distance_seconds,
                                    query_seconds=time.perf_counter() - stamp))
    finally:
        stamp = time.perf_counter()
        # Evaluate all three even if the first one fails. No per-query
        # state traversal; mutations are detected, not rolled back.
        actual = (backend.state_fingerprint(), fingerprint_textclust(base),
                  fingerprint_textclust(effective))
        expected = tuple(metadata[k] for k in (
            "live_state_sha256", "base_state_sha256", "effective_state_sha256"))
        if actual != expected:
            raise RuntimeError("Prediction mutated protected live/base/effective state")
        protected_seconds = time.perf_counter() - stamp
    metadata["diagnostics"] = diagnostics
    metadata["n_predicted_clusters_window"] = len(set(labels) - {-1})
    metadata["costs"].update(
        queries=query_costs, query_subcost_parent="query_seconds",
        protected_check_seconds=protected_seconds,
        prediction_including_boundary_seconds=time.perf_counter() - start,
        cost_scope="local wall seconds; boundary/query children are not additive totals")
    array = np.array(labels, dtype=np.int64)
    array.setflags(write=False)
    return TextClustPrediction(array, metadata)


class TextClustSnapshotBackend(TextClustBackend):
    """Native learning with read-only effective-snapshot window prediction.

    The inherited diagnostic ``metadata`` property dispatches through this
    snapshot override too, so every metadata entry point uses the new protocol.
    Labels have snapshot scope, not persistent semantic topic identities.
    """

    def _make_window_snapshot(self):
        base, effective, idf, metadata = super()._make_window_snapshot()
        metadata.update(
            adapter=ADAPTER,
            protocol=BACKEND_PROTOCOL,
            research_only=False,
            invariance_scope="window boundaries, including query exceptions",
            prediction_semantics=(
                "native nearest eligible micro on once-maintained effective snapshot; "
                "None only maps to -1"),
        )
        metadata["costs"]["cost_scope"] = (
            "local wall seconds; boundary/query children are not additive totals")
        return base, effective, idf, metadata

    def predict_window(self, tfs):
        """Validate the full window, then search without per-query state copies."""
        return _predict_window(self, tfs)
