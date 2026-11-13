"""Compatibility imports; implementation lives in workflows.navigation."""
from .workflows.navigation import (
    Node, Taxonomy, NavigationState, NavigationEvidence, NavigationConfig, NodeProposal, NavigationResult, Hierarchical, apply_proposal
)

__all__ = ['Node', 'Taxonomy', 'NavigationState', 'NavigationEvidence', 'NavigationConfig', 'NodeProposal', 'NavigationResult', 'Hierarchical', 'apply_proposal']
