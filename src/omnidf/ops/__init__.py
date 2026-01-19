"""
OmniDF Operators Module.

This module contains the physical implementations of both relational and semantic operators.
- Relational operators: Executed via Pandas (optimized for vectorized operations)
- Semantic operators: Executed via LLM calls or embedding-based methods
"""

from omnidf.ops.relational import RelationalExecutor
from omnidf.ops.semantic_filter import SemanticFilterExecutor
from omnidf.ops.semantic_map import SemanticMapExecutor
from omnidf.ops.semantic_join import SemanticJoinExecutor
from omnidf.ops.semantic_dedup import SemanticDedupExecutor

__all__ = [
    "RelationalExecutor",
    "SemanticFilterExecutor",
    "SemanticMapExecutor",
    "SemanticJoinExecutor",
    "SemanticDedupExecutor",
]
