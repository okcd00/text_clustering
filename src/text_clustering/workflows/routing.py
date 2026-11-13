"""Retrieve, gate, optionally rerank with NLI; pure prediction, explicit update."""
from dataclasses import dataclass
import numpy as np
from ..types import Assignment, Candidate
from ..vectors import topk_cosine, unit_rows


@dataclass(frozen=True)
class RoutingConfig:
    min_cosine: float = 0.55
    min_margin: float = 0.05
    top_k: int = 5
    min_entailment: float = 0.7
    max_contradiction: float = 0.2
    nli_margin: float = 0.1

    def __post_init__(self):
        if not np.isfinite(self.min_cosine) or not -1 <= self.min_cosine <= 1:
            raise ValueError("min_cosine must be in [-1, 1]")
        for name in ("min_margin", "min_entailment", "max_contradiction", "nli_margin"):
            if not np.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"invalid {name}")
        if self.top_k < 2:
            raise ValueError("top_k >= 2 is required to test ambiguity")


class OnePass:
    def __init__(self, embedder, config=RoutingConfig(), judge=None):
        self.embedder, self.config, self.judge = embedder, config, judge

    def _vector(self, document, index):
        if index.encoder_identity != self.embedder.identity:
            raise ValueError("encoder mismatch: rebuild the index")
        x = unit_rows(self.embedder.encode([document.text]))
        if x.shape != (1, index.sums.shape[1]):
            raise ValueError("encoder must return one vector of index dimension")
        return x[0]

    def _predict(self, document, index, vector):
        retrieved = topk_cosine(vector, index.centers, self.config.top_k)
        candidates = [Candidate(index.ids[i], s) for i, s in retrieved]
        calls = 0
        def decision(status, cluster, reason):
            return Assignment(status, cluster, tuple(candidates), reason, index.revision, calls)
        if not candidates or candidates[0].cosine < self.config.min_cosine:
            return decision("novel", None, "no embedding candidate passes threshold; novelty candidate only")
        if self.judge is not None:
            # Missing descriptions are a configuration error, never silent NLI fallback.
            if any(not index.descriptions.get(c.id, "").strip() for c in candidates):
                raise ValueError("NLI routing requires explicit category hypotheses")
            pairs = [(document.text, index.descriptions[c.id]) for c in candidates]
            evidence = self.judge.score(pairs)
            if len(evidence) != len(candidates):
                raise ValueError("NLI result count mismatch")
            calls = len(pairs)
            candidates = [Candidate(c.id, c.cosine, nli) for c, nli in zip(candidates, evidence)]
            eligible = sorted((c for c in candidates if c.cosine >= self.config.min_cosine
                               and c.nli.contradiction <= self.config.max_contradiction),
                              key=lambda c: (-c.nli.entailment, c.id))
            if not eligible or eligible[0].nli.entailment < self.config.min_entailment:
                return decision("rejected", None, "NLI did not verify membership; not proof of novelty")
            if len(eligible) > 1 and eligible[0].nli.entailment - eligible[1].nli.entailment < self.config.nli_margin:
                return decision("ambiguous", None, "NLI margin below threshold")
            return decision("assigned", eligible[0].id, "embedding and directional NLI accepted")
        if len(candidates) > 1 and candidates[0].cosine - candidates[1].cosine < self.config.min_margin:
            return decision("ambiguous", None, "cosine margin below threshold")
        return decision("assigned", candidates[0].id, "cosine threshold and margin accepted")

    def predict(self, document, index):
        return self._predict(document, index, self._vector(document, index))

    def update(self, document, index, expected_revision):
        """Return a new snapshot; caller owns atomic publication/concurrency control."""
        if expected_revision != index.revision:
            raise ValueError("stale index revision")
        if document.id in index.seen_ids:
            raise ValueError("document ID already processed")
        vector = self._vector(document, index)
        decision = self._predict(document, index, vector)
        updated = index.copy()
        updated.seen_ids.add(document.id)
        if decision.status == "assigned":
            i = updated.ids.index(decision.cluster_id)
            proposed = updated.sums[i] + vector
            unit_rows(proposed.reshape(1, -1))
            updated.sums[i] = proposed
            updated.counts[i] += 1
        updated.revision += 1
        return decision, updated
