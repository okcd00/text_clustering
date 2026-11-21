#!/usr/bin/env python3
"""Offline ESC diagnostic; writes aggregates only. No external model or API."""
import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

PIN = "46edefee5e82e0917b823abe0a18bf8c7770f15c"
HASHES = {"dev": "3ed972fad712d2160a4eafe5252cf0183e88f36b8234467b8785e8c70a3086c1",
          "test": "8991adc0ce58602f11172f2e19fa3b1c8237ec4852d3fd3f3c69a070ce46141b"}
COUNTS = {"dev": (65, 61, 6), "test": (189, 184, 16)}
GRID = (.15, .25, .35, .45, .55, .65)
ARMS = ("frozen_prototype", "daily_centroid")


def vector(tokens):
    # Unigrams and adjacent bigrams, sublinear term frequency, no fitted corpus state.
    words = [w.casefold() for t in tokens for w in re.findall(r"[^\W\d_]+", t)]
    counts = Counter([(w,) for w in words] + list(zip(words, words[1:])))
    weighted = {k: 1 + math.log(v) for k, v in counts.items()}
    norm = math.sqrt(math.fsum(v * v for v in weighted.values()))
    return {k: v / norm for k, v in weighted.items()} if norm else {}


def dct(root):
    days = set()
    for node in root.iter():
        value = node.get("value", "")
        if node.get("DCT") == "TRUE" and re.match(r"^\d{4}-\d\d-\d\d(?:T|$)", value):
            try:
                days.add(date.fromisoformat(value[:10]).isoformat())
            except ValueError:
                pass
    return next(iter(days)) if len(days) == 1 else None


def gold_chain_ids(ecb, path, tokens):
    # Separate ECB+ file: confirm identical ordered token IDs/text before using its annotations.
    xml = ET.parse(ecb / path.parent.name / path.name.removesuffix(".xml")).getroot()
    if [(n.get("t_id"), n.text) for n in xml.findall("token")] != tokens:
        return None
    actions = {n.get("m_id") for n in xml.findall("Markables/*")
               if n.tag.startswith(("ACTION_", "NEG_ACTION_")) and n.findall("token_anchor")}
    return frozenset(n.get("note") for n in xml.findall("Relations/CROSS_DOC_COREF")
                     if n.get("note", "").startswith("ACT")
                     and any(s.get("m_id") in actions for s in n.findall("source")))


def load(root, split):
    esc = root / "annotated_data/v1.0"
    ecb = root / "ECB+_LREC2014/ECB+"
    files = sorted((esc / split).rglob("*.xml"))
    manifest = hashlib.sha256()
    rows, labels, event_gold, fingerprints = [], {}, {}, []
    unaligned = 0
    for f in files:
        raw = f.read_bytes()
        manifest.update(f"{f.relative_to(esc).as_posix()} {hashlib.sha256(raw).hexdigest()}\n".encode())
        doc = ET.fromstring(raw)
        day = dct(doc)
        if day is None:
            continue
        token_nodes = doc.findall("token")
        raw_tokens = [(n.get("t_id"), n.text) for n in token_nodes]
        # Sentence zero is the origin URL field; never use it, or XML labels/attributes.
        words = [n.text or "" for n in token_nodes if n.get("sentence") != "0"]
        features = vector(words)
        if not features:
            raise ValueError("Empty text after the frozen preprocessing")
        key = hashlib.sha256(f.relative_to(esc).as_posix().encode()).hexdigest()
        fingerprints.append(hashlib.sha256(" ".join(" ".join(words).casefold().split()).encode()).hexdigest())
        rows.append((day, key, features))
        labels[key] = f.parent.name
        chains = gold_chain_ids(ecb, f, raw_tokens)
        if chains is None:
            unaligned += 1
        else:
            event_gold[key] = chains
    sha = manifest.hexdigest()
    if sha != HASHES[split] or (len(files), len(rows), len(set(labels.values()))) != COUNTS[split]:
        raise ValueError(f"Unexpected ESC {split} release: {sha}")
    meta = {"documents": len(files), "dated": len(rows), "topics": len(set(labels.values())),
            "days": len({d for d, _, _ in rows}), "manifest_sha256": sha,
            "duplicate_dated_text_rows": len(fingerprints) - len(set(fingerprints)),
            "event_gold_unaligned_rows": unaligned, "event_gold_with_chains_rows": sum(map(bool, event_gold.values()))}
    return sorted(rows), labels, event_gold, meta, set(fingerprints)


def replay(rows, threshold, adaptive):
    days = defaultdict(list)
    for day, doc, features in rows:
        days[day].append((doc, features))
    sums, norms, output = {}, {}, {}
    for day, batch in sorted(days.items()):
        pending = []
        for doc, features in sorted(batch):
            best, best_score = None, -math.inf
            for cluster, terms in sums.items():
                score = math.fsum(v * terms.get(k, 0) for k, v in features.items()) / norms[cluster]
                if score > best_score:  # insertion order is deterministic on ties
                    best, best_score = cluster, score
            assigned = best if best_score >= threshold else doc
            output[doc] = assigned
            pending.append((doc, assigned, features))
        # No same-date neighbor was visible at prediction time.
        updated = set()
        for doc, assigned, features in pending:
            if assigned == doc:
                sums[assigned] = Counter(features)
            elif adaptive:
                sums[assigned].update(features)
            else:
                continue
            updated.add(assigned)
        for assigned in updated:
            norms[assigned] = math.sqrt(math.fsum(v * v for v in sums[assigned].values()))
    return output


def metrics(rows, topics, event_gold, assigned):
    tp = fp = fn = event_pos = event_captured = 0
    for (d1, a, _), (d2, b, _) in combinations(rows, 2):
        if d1 == d2:
            continue
        pred = assigned[a] == assigned[b]
        same = topics[a] == topics[b]
        tp += pred and same
        fp += pred and not same
        fn += not pred and same
        if event_gold.get(a, frozenset()) & event_gold.get(b, frozenset()):
            event_pos += 1
            event_captured += pred
    occupied = Counter(assigned.values())
    return {"cross_day_topic_pairs": {"tp": tp, "fp": fp, "fn": fn,
               "precision": tp / (tp + fp) if tp + fp else None,
               "recall": tp / (tp + fn) if tp + fn else None,
               "f1": 2*tp / (2*tp + fp + fn) if tp + fp + fn else None},
            "confirmed_cross_day_event_link_pairs": {"captured": event_captured, "total": event_pos},
            "clusters": len(occupied), "largest_cluster_rows": max(occupied.values(), default=0)}


def run(root):
    dev, labels, event_gold, dm, dfp = load(root, "dev")
    rows = []
    for threshold in GRID:
        scores = {arm: metrics(dev, labels, event_gold, replay(dev, threshold, i == 1))
                  ["cross_day_topic_pairs"]["f1"] for i, arm in enumerate(ARMS)}
        rows.append({"threshold": threshold, "dev_f1": scores})
    best = max(rows, key=lambda row: (sum(row["dev_f1"].values()), row["threshold"]))["threshold"]
    # Read test only after selecting the shared threshold.
    test, labels, event_gold, tm, tfp = load(root, "test")
    scores = {arm: metrics(test, labels, event_gold, replay(test, best, i == 1))
              for i, arm in enumerate(ARMS)}
    return {"source_commit": PIN, "dev": dm, "test": tm,
            "exact_normalized_text_overlap": len(dfp & tfp), "cross_day_topic_pair_threshold": best,
            "dev_grid": rows, "test_results": scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("esc_checkout", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.esc_checkout), ensure_ascii=False, indent=2, sort_keys=True))
