"""
Rule-based optimization rules for OmniDF.
"""

from omnidf.optimizer.rules.filter_pushdown import FilterPushDown
from omnidf.optimizer.rules.semantic_fusion import SemanticMapFusion, SemanticFilterFusion
from omnidf.optimizer.rules.semantic_join_decomposition import SemanticJoinDecomposition
from omnidf.optimizer.rules.projection_pushdown import ProjectionPushDown

__all__ = [
    "FilterPushDown",
    "SemanticMapFusion",
    "SemanticFilterFusion",
    "SemanticJoinDecomposition",
    "ProjectionPushDown",
]
