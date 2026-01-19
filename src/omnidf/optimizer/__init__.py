"""
OmniDF Query Optimizer Module.

Provides both rule-based and LLM-based query optimization:
- Rule-based: FilterPushDown, SemanticMapFusion, SemanticFilterFusion, SemanticJoinDecomposition
- LLM-based: Uses LLM to analyze plans and suggest optimizations
"""

from omnidf.optimizer.base import Rule, Optimizer
from omnidf.optimizer.rules import (
    FilterPushDown,
    SemanticMapFusion,
    SemanticFilterFusion,
    SemanticJoinDecomposition,
    ProjectionPushDown,
)
from omnidf.optimizer.llm_optimizer import LLMOptimizer

__all__ = [
    "Rule",
    "Optimizer",
    "FilterPushDown",
    "SemanticMapFusion",
    "SemanticFilterFusion",
    "SemanticJoinDecomposition",
    "ProjectionPushDown",
    "LLMOptimizer",
]
