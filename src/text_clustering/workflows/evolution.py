"""Persistent vector-stream orchestration, deliberately without gold labels.

This is window-predict-then-learn, not per-item prequential evaluation. Cluster
IDs belong to a snapshot; callers must not concatenate them for global ACC.
"""
from copy import deepcopy
from numbers import Integral
import time

import numpy as np


def run_window_protocol(backend, vectors, order, *, warmup=512, window_size=256,
                        update=True, on_prediction):
    """Persist each window's predictions before exposing its rows to learning.

    ``on_prediction(record)`` must durably save the detached record or raise.
    If it raises, that window is not learned. There is no retry/rollback of
    successful earlier updates. Supply a fresh backend per arm/seed. The final
    window is also learned in the updating arm, with its cost retained although
    it cannot improve a later scored prediction. Returns costs and state only;
    scoring belongs in a separate process after prediction artifacts are closed.
    """
    for name, value in (("warmup", warmup), ("window_size", window_size)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(update, bool) or not callable(on_prediction):
        raise ValueError("update must be bool and on_prediction must be callable")
    x = np.asarray(vectors)
    ids = np.asarray(order)
    if x.ndim != 2 or x.dtype.kind not in "fiu" or x.shape[1] == 0:
        raise ValueError("vectors must be a dense 2D real array")
    if not np.isfinite(x).all() or np.any(np.all(x == 0, axis=1)):
        raise ValueError("vectors must be finite and nonzero")
    if not warmup < len(x):
        raise ValueError("warmup must leave at least one prediction row")
    if (ids.shape != (len(x),) or ids.dtype.kind not in "iu"
            or not np.array_equal(np.sort(ids), np.arange(len(x)))):
        raise ValueError("order must be an exact permutation of row indices")
    if backend.state_revision != 0:
        raise ValueError("protocol requires a fresh unlearned backend")
    costs = []
    start = time.perf_counter()
    for row_id in ids[:warmup]:
        backend.learn_one(x[row_id])
    warmup_seconds = time.perf_counter() - start
    fingerprint = getattr(backend, "state_fingerprint", lambda: None)
    initial_state_sha256 = fingerprint()
    for window_index, offset in enumerate(range(warmup, len(x), window_size)):
        row_ids = ids[offset:offset + window_size]
        revision = backend.state_revision
        start = time.perf_counter()
        state_before = fingerprint()
        prediction = backend.predict_window(x[row_ids])
        if fingerprint() != state_before:
            raise RuntimeError("prediction mutated learned microstate")
        predict_seconds = time.perf_counter() - start
        if backend.state_revision != revision:
            raise RuntimeError("prediction unexpectedly learned from the window")
        labels = np.asarray(prediction.labels)
        if labels.shape != (len(row_ids),) or labels.dtype.kind not in "iu" or np.any(labels < -1):
            raise RuntimeError("invalid prediction labels")
        record = {"window_index": window_index, "offset": offset,
                  "row_ids": row_ids.tolist(), "labels": labels.tolist(),
                  "snapshot": deepcopy(prediction.metadata),
                  "state_revision_before": revision,
                  "microstate_sha256": state_before,
                  "prediction_including_boundary_seconds": predict_seconds}
        start = time.perf_counter()
        on_prediction(deepcopy(record))
        persist_seconds = time.perf_counter() - start
        start = time.perf_counter()
        if update:
            for row_id in row_ids:
                backend.learn_one(x[row_id])
        update_seconds = time.perf_counter() - start
        expected = revision + (len(row_ids) if update else 0)
        if backend.state_revision != expected:
            raise RuntimeError("unexpected number of learning updates")
        costs.append({"window_index": window_index, "n": len(row_ids),
                      "prediction_including_boundary_seconds": predict_seconds,
                      "persist_seconds": persist_seconds, "update_seconds": update_seconds,
                      "state_revision_after": backend.state_revision})
    start = time.perf_counter()
    terminal = deepcopy(backend.metadata)
    terminal_snapshot_seconds = time.perf_counter() - start
    return {"protocol": "stream-window-v1", "update": update,
            "n": len(x), "n_scored": len(x) - warmup,
            "warmup": int(warmup), "window_size": int(window_size),
            "initial_state_sha256": initial_state_sha256,
            "warmup_seconds": warmup_seconds, "windows": costs,
            "terminal_snapshot": terminal,
            "terminal_snapshot_seconds": terminal_snapshot_seconds}
