"""Finite, nonzero normalized vectors and bounded-memory exact retrieval."""
import numpy as np


def unit_rows(values):
    x = np.array(values, dtype=np.float64, copy=True)
    if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("expected finite matrix [n, d], d > 0")
    # Scale before squaring to avoid overflow for finite, large components.
    scale = np.max(np.abs(x), axis=1)
    if np.any(scale <= 1e-12):
        raise ValueError("zero or numerically tiny vectors cannot be compared by cosine")
    scaled = x / scale[:, None]
    return scaled / np.linalg.norm(scaled, axis=1)[:, None]


def topk_cosine(query, vectors, k=5, block_size=4096):
    if k < 1 or block_size < 1:
        raise ValueError("k and block_size must be positive")
    q = unit_rows(np.asarray(query).reshape(1, -1))[0]
    if vectors.ndim != 2 or vectors.shape[1] != len(q):
        raise ValueError("vector dimensions do not match")
    best = []
    for start in range(0, len(vectors), block_size):
        scores = vectors[start:start + block_size] @ q
        best.extend((start + j, float(s)) for j, s in enumerate(scores))
        best = sorted(best, key=lambda p: (-p[1], p[0]))[:k]
    return best
