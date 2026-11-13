"""Causal raw-text windows, intentionally separate from the dense workflow.

The callback must durably persist (serialize, write, flush, fsync) or raise.
A returned callback receipt is not assumed to establish durability: that is the
caller's contract. No retries, gold, scoring, resource supervisor or data loader
live here. The outer runner owns run/config/corpus/seed/document IDs, input
identity validation, immutable files, caps, process wall/CPU/RSS and failures.
It must bind that context to each prediction record before persisting it.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from numbers import Integral
import time

import numpy as np

from ..algorithms.textclust import (
    ADAPTER, BACKEND_PROTOCOL, TF_ENCODING, TOKENIZER_SOURCE_SHA256, TOKENIZER_SPEC,
    build_tokenizer, encode_tf, validate_tf,
)

PROTOCOL = "textclust-token-window-v1"


def _bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def _latency(values):
    """NumPy linear quantiles, not distributions of window-normalized queries."""
    return {"n": len(values), "p50": float(np.quantile(values, .5, method="linear"))
            if values else None,
            "p95": float(np.quantile(values, .95, method="linear")) if values else None,
            "max": max(values) if values else None, "quantile_method": "linear"}


def run_text_window_protocol(backend, texts, order, *, warmup=512, window_size=256,
                             update=True, on_prediction):
    """Tokenize warmup/current-window strings, persist predictions, then learn.

    Mirrors evolution.run_window_protocol's arguments, record keys and return
    keys, except ``texts`` replaces ``vectors``. Requires a fresh backend. Empty
    texts/TFs stay in the stream and advance revision when learned. The last
    short window is retained and updated. Defaults implement the 512/256 plan;
    smaller explicit values support semantic tests, not formal protocol claims.

    Callback records are detached, JSON-safe and describe only pre-update
    predictions. Persistence/update timings and live endpoint transitions become
    available afterwards in returned ``windows``. Never rewrite old labels.
    Exceptions propagate, leaving earlier successful windows intact; there is
    no successful/incomplete return masquerading as a completed job.

    All ``*_seconds`` are measured wall seconds, not process CPU. Top-level
    phases and each window's ``phase_seconds`` are disjoint. Snapshot/query
    timers and percentiles are nested diagnostics, NOT additional stages to
    sum. ``workflow_seconds`` includes setup, copies, fingerprints, tokenization,
    callback serialization/fsync, release and terminal diagnostics; excludes
    caller startup/input-file reads, final-result serialization and process exit.
    """
    workflow_start = time.perf_counter()
    start = time.perf_counter()
    for name, value in (("warmup", warmup), ("window_size", window_size)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(update, bool) or not callable(on_prediction):
        raise ValueError("update must be bool and on_prediction must be callable")
    if isinstance(texts, (str, bytes, Mapping)):
        raise ValueError("texts must be a sequence of raw strings, not a labeled bundle")
    texts = list(texts)  # reading strings is not tokenization or future fitting
    if not all(isinstance(text, str) for text in texts):
        raise ValueError("texts must contain only raw strings, not dense floats or labeled rows")
    if not warmup < len(texts):
        raise ValueError("warmup must leave at least one prediction row")
    if isinstance(order, (str, bytes, Mapping)):
        raise ValueError("order must be an exact permutation of row indices")
    ordered = list(order)
    if (len(ordered) != len(texts)
            or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in ordered)
            or sorted(ordered) != list(range(len(texts)))):
        raise ValueError("order must be an exact permutation of row indices")
    ids = [int(i) for i in ordered]
    if backend.state_revision != 0:
        raise ValueError("protocol requires a fresh unlearned backend")
    input_validation_seconds = time.perf_counter() - start
    start = time.perf_counter()
    tokenizer = build_tokenizer()
    tokenizer_spec = dict(TOKENIZER_SPEC, ngram_range=list(TOKENIZER_SPEC["ngram_range"]))
    tokenizer_sha = hashlib.sha256(_bytes(tokenizer_spec)).hexdigest()
    order_sha = hashlib.sha256(_bytes(ids)).hexdigest()
    token_stream = hashlib.sha256()
    tokenizer_setup_seconds = time.perf_counter() - start

    def tokenize(row_ids):
        rows, evidence = [], []
        for row_id in row_ids:
            text = texts[row_id]
            tf = validate_tf(tokenizer.transform_one(text))
            terms = encode_tf(tf)
            encoded = _bytes(terms)
            # JSONL records make stream framing unambiguous and retain row order.
            token_stream.update(_bytes([row_id, terms]) + b"\n")
            rows.append(tf)
            evidence.append({
                "row_id": row_id,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "raw_empty_text": text == "", "blank_text": not text.strip(),
                "empty_tf": not tf, "terms": terms,
                "tf_sha256": hashlib.sha256(encoded).hexdigest(),
            })
        return rows, evidence

    start = time.perf_counter()
    warmup_rows, warmup_evidence = tokenize(ids[:warmup])
    warmup_tokenization_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for tf in warmup_rows:
        backend.learn_one(tf)
    warmup_seconds = time.perf_counter() - start
    start = time.perf_counter()
    if backend.state_revision != warmup:
        raise RuntimeError("unexpected warmup learning count")
    initial_state_sha256 = backend.state_fingerprint()
    initial_live = backend.live_summary()
    warmup_token_sha = token_stream.hexdigest()
    # Keep only evidence, not warmup TF mappings; current-window TF cache is bounded.
    del warmup_rows, tf
    warmup_diagnostics_seconds = time.perf_counter() - start
    windows = []
    query_latencies = []
    branch = "updating" if update else "frozen"
    for window_index, offset in enumerate(range(warmup, len(texts), window_size)):
        row_ids = ids[offset:offset + window_size]
        revision = backend.state_revision
        start = time.perf_counter()
        rows, evidence = tokenize(row_ids)
        tokenization_seconds = time.perf_counter() - start
        start = time.perf_counter()
        state_before = backend.state_fingerprint()
        live_before = backend.live_summary()
        prediction = backend.predict_window(rows)
        if backend.state_fingerprint() != state_before or backend.state_revision != revision:
            raise RuntimeError("prediction mutated learned TextClust state")
        predict_seconds = time.perf_counter() - start
        start = time.perf_counter()
        labels = np.asarray(prediction.labels)
        if (labels.shape != (len(row_ids),) or labels.dtype.kind not in "iu"
                or np.any(labels < -1)):
            raise RuntimeError("invalid prediction labels")
        diagnostics = prediction.metadata["diagnostics"]
        if len(diagnostics) != len(rows):
            raise RuntimeError("missing per-query diagnostics")
        empty = sum(d["empty_tf"] for d in diagnostics)
        all_oov = sum(d["all_oov"] for d in diagnostics)
        assigned = int(np.count_nonzero(labels != -1))
        counts = {
            "n": len(rows), "assigned": assigned,
            "empty_tf_count": empty, "empty_tf_denominator": len(rows),
            "raw_empty_text_count": sum(e["raw_empty_text"] for e in evidence),
            "all_oov_count_nonempty": all_oov, "nonempty_tf_count": len(rows) - empty,
            "n_predicted_clusters_window": len(set(labels.tolist()) - {-1}),
            "empty_tf_rate": _rate(empty, len(rows)),
            "raw_empty_text_rate": _rate(sum(e["raw_empty_text"] for e in evidence), len(rows)),
            "blank_text_rate": _rate(sum(e["blank_text"] for e in evidence), len(rows)),
            "all_oov_rate_nonempty": _rate(all_oov, len(rows) - empty),
            "coverage": _rate(assigned, len(rows)),
        }
        record = {
            "window_index": window_index, "offset": offset,
            "row_ids": row_ids, "labels": labels.tolist(),
            "snapshot": deepcopy(prediction.metadata),
            "state_revision_before": revision, "microstate_sha256": state_before,
            "prediction_including_boundary_seconds": predict_seconds,
            "protocol": PROTOCOL, "method": BACKEND_PROTOCOL, "adapter": ADAPTER,
            "branch": branch, "tokenizer_spec_sha256": tokenizer_sha,
            "tf_encoding": TF_ENCODING, "token_evidence": evidence,
            "token_prefix_sha256": token_stream.hexdigest(),
            "counts": counts, "tokenization_seconds": tokenization_seconds,
        }
        record_preparation_seconds = time.perf_counter() - start
        start = time.perf_counter()
        on_prediction(deepcopy(record))  # MUST return only after durable persistence
        persist_seconds = time.perf_counter() - start
        start = time.perf_counter()
        # A callback cannot mutate this record/cache via the detached copy.
        if backend.state_revision != revision or backend.state_fingerprint() != state_before:
            raise RuntimeError("persistence callback mutated backend state")
        callback_state_check_seconds = time.perf_counter() - start
        start = time.perf_counter()
        if update:
            for tf in rows:
                backend.learn_one(tf)
            del tf
        update_seconds = time.perf_counter() - start
        start = time.perf_counter()
        expected = revision + (len(rows) if update else 0)
        if backend.state_revision != expected:
            raise RuntimeError("unexpected number of learning updates")
        live_after = backend.live_summary()
        after_sha = backend.state_fingerprint()
        if not update and after_sha != initial_state_sha256:
            raise RuntimeError("frozen live state changed")
        old_ids, new_ids = set(live_before["micro_ids"]), set(live_after["micro_ids"])
        first_id, next_id = live_before["next_cluster_id"], live_after["next_cluster_id"]
        transitions = {
            "born_ids_count": next_id - first_id,
            "born_id_interval": [first_id, next_id],  # half-open, includes transient births
            "boundary_disappeared_ids": sorted(old_ids - new_ids),
            "surviving_ids": sorted(old_ids & new_ids),
            "merged_events": None, "expired_events": None,
            "event_counts_unavailable_reason": "native events are not instrumented; endpoint differences cannot attribute events",
        }
        query_latencies.extend(c["query_seconds"] for c in prediction.metadata["costs"]["queries"])
        window = {
            "window_index": window_index, "offset": offset, "row_ids": list(row_ids),
            "n": len(rows), "state_revision_before": revision,
            "state_revision_after": backend.state_revision,
            "microstate_sha256_before": state_before, "microstate_sha256_after": after_sha,
            "live_before": live_before, "live_after": live_after,
            "snapshot": prediction.metadata, "counts": counts, "transitions": transitions,
            "tokenization_seconds": tokenization_seconds,
            "prediction_including_boundary_seconds": predict_seconds,
            "persist_seconds": persist_seconds, "update_seconds": update_seconds,
            "is_last_window": offset + len(rows) == len(texts),
        }
        post_update_diagnostics_seconds = time.perf_counter() - start
        start = time.perf_counter()
        del rows, evidence, record, prediction, labels, diagnostics
        release_seconds = time.perf_counter() - start
        window["phase_seconds"] = {
            "tokenization": tokenization_seconds, "prediction_including_boundary": predict_seconds,
            "record_preparation": record_preparation_seconds, "persistence": persist_seconds,
            "callback_state_check": callback_state_check_seconds, "update": update_seconds,
            "post_update_diagnostics": post_update_diagnostics_seconds, "release": release_seconds,
        }
        windows.append(window)
    start = time.perf_counter()
    terminal_before = backend.state_fingerprint()
    terminal = backend.metadata
    if backend.state_fingerprint() != terminal_before:
        raise RuntimeError("terminal diagnostic mutated live state")
    terminal_snapshot_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = {
        "protocol": PROTOCOL, "method": BACKEND_PROTOCOL, "adapter": ADAPTER,
        "status": "complete", "update": update, "branch": branch,
        "n": len(texts), "n_scored": len(texts) - warmup,
        "warmup": int(warmup), "window_size": int(window_size),
        "order": ids, "order_sha256": order_sha,
        "order_encoding": "compact-json-integer-array-v1",
        "warmup_row_ids": ids[:warmup], "warmup_token_evidence": warmup_evidence,
        "tokenizer_spec": tokenizer_spec, "tokenizer_spec_sha256": tokenizer_sha,
        "tokenizer_source_sha256": TOKENIZER_SOURCE_SHA256, "tf_encoding": TF_ENCODING,
        "token_stream_encoding": "compact-jsonl-[row_id,ordered-typed-tf]-v1",
        "warmup_token_prefix_sha256": warmup_token_sha,
        "token_stream_sha256": token_stream.hexdigest(),
        "initial_state_sha256": initial_state_sha256, "initial_live": initial_live,
        "warmup_seconds": warmup_seconds,
        "warmup_tokenization_seconds": warmup_tokenization_seconds,
        "windows": windows, "terminal_snapshot": terminal,
        "terminal_snapshot_seconds": terminal_snapshot_seconds,
        "latency_seconds": {
            "boundary": _latency([w["snapshot"]["costs"]["boundary_seconds"] for w in windows]),
            "query_including_copy": _latency(query_latencies),
            "window_update": _latency([w["update_seconds"] for w in windows]),
        },
        "phase_seconds": {
            "input_validation": input_validation_seconds,
            "tokenizer_setup": tokenizer_setup_seconds,
            "warmup_tokenization": warmup_tokenization_seconds, "warmup_learning": warmup_seconds,
            "warmup_diagnostics": warmup_diagnostics_seconds,
            "terminal_snapshot": terminal_snapshot_seconds,
        },
        "cost_scope": "workflow wall; includes callback, excludes process startup/exit and result serialization",
        "cost_nesting": "sum phase_seconds plus windows[*].phase_seconds only; other timers are nested",
    }
    result["phase_seconds"]["result_preparation"] = time.perf_counter() - start
    result["workflow_seconds"] = time.perf_counter() - workflow_start
    return result
