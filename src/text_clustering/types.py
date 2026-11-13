"""Shared contracts. Scores are not calibrated probabilities unless stated."""
from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class Document:
    id: str
    text: str


class Embedder(Protocol):
    identity: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class NLIResult:
    entailment: float
    neutral: float
    contradiction: float

    def __post_init__(self):
        p = np.array([self.entailment, self.neutral, self.contradiction])
        if not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
            raise ValueError("NLI result must contain three probabilities summing to one")


class EntailmentJudge(Protocol):
    def score(self, pairs: Sequence[Tuple[str, str]]) -> Sequence[NLIResult]: ...


@dataclass(frozen=True)
class Candidate:
    id: str
    cosine: float
    nli: Optional[NLIResult] = None


@dataclass(frozen=True)
class Assignment:
    status: str  # assigned | ambiguous | novel | rejected
    cluster_id: Optional[str]
    candidates: Tuple[Candidate, ...]
    reason: str
    index_revision: int
    nli_pairs: int = 0


@dataclass
class Partition:
    labels: np.ndarray  # -1 is noise, never a cluster
    metadata: Mapping[str, object] = field(default_factory=dict)


class ClusterAlgorithm(Protocol):
    def fit(self, vectors: np.ndarray) -> Partition: ...
