"""Embedding and optional directional-NLI navigation policies.

A conservative engineering baseline, NOT kb_agent_guide's trained probability head.
ADD requires parent membership plus low child evidence; semantic review still needed.
"""
from ..hierarchical import NavigationEvidence
from ..vectors import unit_rows


class EvidenceNavigation:
    def __init__(self, embedder, judge=None, parent_threshold=0.7,
                 child_reject_threshold=0.35, max_contradiction=0.2):
        for v in (parent_threshold, child_reject_threshold, max_contradiction):
            if not 0 <= v <= 1:
                raise ValueError("invalid evidence threshold")
        self.embedder, self.judge = embedder, judge
        self.parent_threshold, self.child_reject_threshold = parent_threshold, child_reject_threshold
        self.max_contradiction = max_contradiction

    def score(self, state):
        is_root = len(state.path) == 1
        nodes = list(state.children) + ([] if is_root else [state.node])
        if not nodes:
            return NavigationEvidence({}, 0, 0)
        if self.judge is not None:
            if len(nodes) > state.remaining_judgments:
                return NavigationEvidence({n.id: 0 for n in state.children}, 0, 0, exhausted=True)
            if any(not n.description.strip() for n in nodes):
                raise ValueError("NLI needs explicit node hypotheses, not opaque labels")
            evidence = self.judge.score([(state.document.text, n.description) for n in nodes])
            if len(evidence) != len(nodes):
                raise ValueError("NLI result count mismatch")
            values = [e.entailment if e.contradiction <= self.max_contradiction else 0.0 for e in evidence]
            judgments = len(nodes)
        else:
            x = unit_rows(self.embedder.encode([state.document.text] + [n.description or n.label for n in nodes]))
            if len(x) != len(nodes) + 1:
                raise ValueError("encoder count mismatch")
            values = (x[1:] @ x[0]).clip(-1, 1).tolist()
            judgments = 0
        enter = {n.id: v for n, v in zip(state.children, values)}
        emit = 0 if is_root else values[-1]
        # An empty child list is not evidence that a known leaf needs expansion.
        # Leaf/root expansion requires a separate explicit granularity policy.
        add = float(not is_root and bool(state.children) and emit >= self.parent_threshold and
                    max(enter.values(), default=-1) < self.child_reject_threshold)
        return NavigationEvidence(enter, emit, add, judgments)
