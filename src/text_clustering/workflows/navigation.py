"""Budgeted multi-path navigation with independent ENTER / EMIT / ADD.

Adapted design, not a copy or trained reproduction of kb_agent_guide ProbNav.
No prediction mutates the shared taxonomy. ADD creates reviewable proposals.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple
import hashlib
import json
import math
from types import MappingProxyType


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    parent_id: Optional[str]
    description: str = ""


class Taxonomy:
    def __init__(self, nodes, revision=0):
        nodes = tuple(nodes)
        self.nodes = MappingProxyType({n.id: n for n in nodes})
        if not nodes or len(self.nodes) != len(nodes) or revision < 0:
            raise ValueError("nonempty unique nodes and nonnegative revision required")
        if any(not n.id or not n.label.strip() for n in nodes):
            raise ValueError("node ID/label cannot be empty")
        roots = [n.id for n in nodes if n.parent_id is None]
        if len(roots) != 1:
            raise ValueError("taxonomy must have exactly one root")
        self.root_id, self.revision = roots[0], revision
        children = {n.id: [] for n in nodes}
        for node in nodes:
            visited, current = set(), node
            while current.parent_id is not None:
                if current.id in visited or current.parent_id not in self.nodes:
                    raise ValueError("cycle or missing parent")
                visited.add(current.id)
                current = self.nodes[current.parent_id]
            if node.parent_id is not None:
                children[node.parent_id].append(node.id)
        self.children = MappingProxyType({k: tuple(sorted(v)) for k, v in children.items()})
        canonical = [(n.id, n.label, n.parent_id, n.description) for n in sorted(nodes, key=lambda n: n.id)]
        self.fingerprint = hashlib.sha256(json.dumps(canonical, ensure_ascii=False).encode()).hexdigest()

    def path(self, node_id):
        result = []
        while node_id is not None:
            result.append(node_id)
            node_id = self.nodes[node_id].parent_id
        return tuple(reversed(result))


@dataclass(frozen=True)
class NavigationState:
    document: object
    node: Node
    children: Tuple[Node, ...]
    path: Tuple[str, ...]
    remaining_judgments: int


@dataclass(frozen=True)
class NavigationEvidence:
    enter: Mapping[str, float]
    emit: float
    add: float
    judgments: int = 0
    exhausted: bool = False
    # Scores may be cosine heuristics. Never interpret them as calibrated probabilities.


@dataclass(frozen=True)
class NavigationConfig:
    enter_threshold: float = 0.6
    emit_threshold: float = 0.65
    add_threshold: float = 0.8
    beam_width: int = 2
    max_expansions: int = 64
    max_judgments: int = 128
    max_proposals: int = 4

    def __post_init__(self):
        for value in (self.enter_threshold, self.emit_threshold, self.add_threshold):
            if not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError("invalid navigation threshold")
        if self.beam_width < 1 or self.max_expansions < 1 or min(self.max_judgments, self.max_proposals) < 0:
            raise ValueError("invalid navigation budget")


@dataclass(frozen=True)
class NodeProposal:
    parent_id: str
    label: str
    description: str
    document_id: str
    taxonomy_revision: int
    taxonomy_fingerprint: str


@dataclass(frozen=True)
class NavigationResult:
    status: str  # complete | budget_exhausted
    assignments: Tuple[str, ...]
    proposals: Tuple[NodeProposal, ...]
    visited: Tuple[str, ...]
    pending: Tuple[str, ...]
    judgments: int
    events: Tuple[str, ...] = ()


class Hierarchical:
    def __init__(self, policy, config=NavigationConfig(), generator=None):
        self.policy, self.config, self.generator = policy, config, generator

    def predict(self, document, taxonomy):
        frontier = deque([taxonomy.root_id])
        visited, emitted, proposals, events = [], [], [], []
        judgments, exhausted = 0, False
        while frontier and len(visited) < self.config.max_expansions:
            node_id = frontier.popleft()
            state = NavigationState(document, taxonomy.nodes[node_id],
                tuple(taxonomy.nodes[i] for i in taxonomy.children[node_id]),
                taxonomy.path(node_id), self.config.max_judgments - judgments)
            scores = self.policy.score(state)
            if set(scores.enter) != set(taxonomy.children[node_id]):
                raise ValueError("policy must score every exposed child")
            if any(not math.isfinite(v) or not -1 <= v <= 1
                   for v in [scores.emit, scores.add, *scores.enter.values()]):
                raise ValueError("invalid policy scores")
            if not isinstance(scores.judgments, int) or not 0 <= scores.judgments <= state.remaining_judgments:
                raise ValueError("policy exceeded judgment budget")
            judgments += scores.judgments
            if scores.exhausted:
                frontier.appendleft(node_id)
                exhausted = True
                break
            visited.append(node_id)
            if scores.emit >= self.config.emit_threshold and node_id != taxonomy.root_id:
                emitted.append(node_id)
            selected = sorted((i for i, s in scores.enter.items() if s >= self.config.enter_threshold),
                              key=lambda i: (-scores.enter[i], i))
            if len(selected) > self.config.beam_width:
                events.append(f"beam_pruned:{node_id}:{len(selected) - self.config.beam_width}")
            frontier.extend(selected[:self.config.beam_width])
            if scores.add >= self.config.add_threshold:
                if self.generator is None:
                    events.append(f"add_unhandled:{node_id}")
                elif len(proposals) >= self.config.max_proposals:
                    events.append(f"proposal_budget:{node_id}")
                    exhausted = True
                else:
                    generated = self.generator(state)
                    if generated is not None:
                        label, description = generated
                        if not label.strip():
                            raise ValueError("generator returned empty label")
                        existing = {n.label.strip().casefold() for n in state.children}
                        if label.strip().casefold() in existing:
                            events.append(f"duplicate_proposal:{node_id}")
                        else:
                            proposals.append(NodeProposal(node_id, label.strip(), description,
                                document.id, taxonomy.revision, taxonomy.fingerprint))
        if frontier:
            exhausted = True
        return NavigationResult("budget_exhausted" if exhausted else "complete",
            tuple(emitted), tuple(proposals), tuple(visited), tuple(frontier), judgments, tuple(events))


def apply_proposal(taxonomy, proposal, support_document_ids, min_support=3):
    """Explicit application with freshness/support/exact-duplicate checks.

    Support IDs are supplied by the caller after semantic review. Distinct IDs alone
    do not establish conceptual novelty; use held-out human/model validation.
    """
    if (proposal.taxonomy_revision != taxonomy.revision or
            proposal.taxonomy_fingerprint != taxonomy.fingerprint):
        raise ValueError("stale or foreign taxonomy proposal")
    support = set(support_document_ids)
    if min_support < 2 or len(support) < min_support or proposal.document_id not in support:
        raise ValueError("insufficient distinct supporting documents")
    if proposal.parent_id not in taxonomy.nodes or not proposal.label.strip():
        raise ValueError("invalid proposal")
    if any(taxonomy.nodes[i].label.strip().casefold() == proposal.label.strip().casefold()
           for i in taxonomy.children[proposal.parent_id]):
        raise ValueError("duplicate sibling label")
    key = hashlib.sha256((proposal.parent_id + "\0" + proposal.label.casefold()).encode()).hexdigest()[:24]
    node = Node("gen-" + key, proposal.label, proposal.parent_id, proposal.description)
    return Taxonomy([*taxonomy.nodes.values(), node], taxonomy.revision + 1)
