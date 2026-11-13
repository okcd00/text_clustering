"""Explicit raw-text facade and separately identified snapshot-window protocol.

Reuse the frozen causal scheduler, not its experiment identity. The public
callback receives the new identity before persistence; historical source and
formal runner remain unchanged. No supervised scoring or resource caps here.
"""
from collections.abc import Mapping
from copy import deepcopy
import time

from ..algorithms.textclust_snapshot import (
    ADAPTER, BACKEND_PROTOCOL, TextClustSnapshotBackend,
)
from .text_evolution import run_text_window_protocol
from . import text_evolution as legacy
from ..algorithms.textclust import TextClustBackend

PROTOCOL = "textclust-effective-snapshot-window-v1"


def _identity(record, snapshot):
    if (record.get('protocol'), record.get('method'), record.get('adapter')) != (
            legacy.PROTOCOL, legacy.BACKEND_PROTOCOL, legacy.ADAPTER):
        raise RuntimeError("scheduler identity mismatch")
    if snapshot.get("adapter") != ADAPTER or snapshot.get("protocol") != BACKEND_PROTOCOL:
        raise RuntimeError("snapshot identity mismatch")
    record.update(protocol=PROTOCOL, method=BACKEND_PROTOCOL, adapter=ADAPTER)


def _check_window(record):
    snapshot = record['snapshot']
    if snapshot.get('adapter') != ADAPTER or snapshot.get('protocol') != BACKEND_PROTOCOL:
        raise RuntimeError('window snapshot identity mismatch')
    if (snapshot['state_revision'] != record['state_revision_before'] or
            snapshot['live_state_sha256'] != record.get(
                'microstate_sha256', record.get('microstate_sha256_before'))):
        raise RuntimeError('window snapshot state mismatch')


def run_snapshot_window_protocol(backend, texts, order, *, warmup=512,
                                 window_size=256, update=True, on_prediction):
    """Same causal 512/256 schedule, separately identified prediction adapter.

    The callback must persist/flush/fsync or raise, as in the old scheduler.
    Single-writer only. This interface is not formal evaluation admission.
    """
    started = time.perf_counter()
    if type(backend) is not TextClustSnapshotBackend:
        raise TypeError("snapshot workflow requires TextClustSnapshotBackend")
    if not callable(on_prediction):
        raise ValueError("on_prediction must be callable")

    def persist(record):
        _check_window(record)
        _identity(record, record["snapshot"])
        on_prediction(record)

    result = run_text_window_protocol(
        backend, texts, order, warmup=warmup, window_size=window_size,
        update=update, on_prediction=persist)
    for window in result['windows']:
        _check_window(window)
    _identity(result, result["terminal_snapshot"])
    # This adapter has no per-query deepcopy. The inherited timer covers native
    # search plus local diagnostics, not just the distance arithmetic.
    result["latency_seconds"]["query_including_diagnostics"] = result["latency_seconds"].pop(
        "query_including_copy")
    result['adapter_workflow_seconds'] = time.perf_counter() - started
    result['adapter_cost_scope'] = (
        'adapter_workflow_seconds includes entry/return validation; workflow_seconds '
        'is the nested scheduler; persistence includes identity bridge checks')
    return result


class TextEvolution:
    """Select a raw-text stream algorithm; every run gets a fresh backend.

    Run parameters and algorithm parameters are separate. Callers own durable
    prediction storage and input identity; returned data includes token evidence.
    """
    def __init__(self, *, algorithm, config=None):
        if not isinstance(algorithm, Mapping) or algorithm.get("name") not in (
                "river_textclust", "river_textclust_snapshot"):
            raise ValueError("text_evolution requires a TextClust algorithm mapping")
        if set(algorithm) - {"name", "radius"}:
            raise ValueError("unknown text_evolution algorithm option")
        if config is not None and not isinstance(config, Mapping):
            raise ValueError("text_evolution config must be a mapping")
        if set(config or {}) - {"warmup", "window_size", "update"}:
            raise ValueError("unknown text_evolution config option")
        self.algorithm = deepcopy(dict(algorithm))
        self.config = deepcopy(dict(config or {}))

    def run(self, texts, order, *, on_prediction):
        from ..factory import build_stream_algorithm
        backend = build_stream_algorithm(self.algorithm)
        expected = (TextClustSnapshotBackend if self.algorithm['name'] ==
                    'river_textclust_snapshot' else TextClustBackend)
        if type(backend) is not expected:
            raise TypeError('algorithm and workflow backend identities differ')
        runner = (run_snapshot_window_protocol if self.algorithm["name"] ==
                  "river_textclust_snapshot" else run_text_window_protocol)
        return runner(backend, texts, order, on_prediction=on_prediction, **self.config)
