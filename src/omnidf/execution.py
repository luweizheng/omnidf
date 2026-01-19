"""
Execution engine for OmniDF.

Design Principles:
1. Relational operators executed via Pandas (vectorized operations)
2. Semantic operators executed via modular ops (LLM, embedding, etc.)
3. Bottom-up execution (execute children first)
4. Caching support for intermediate results

This module orchestrates execution by delegating to:
- ops/relational.py: Pandas-based relational operators
- ops/semantic_*.py: Semantic operator implementations
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

from omnidf.plan import (
    PlanNode,
    NodeType,
    Source,
    Filter,
    SemanticFilter,
    SemanticMap,
    SemanticJoin,
    SemanticDedup,
    Join,
    Project,
    Aggregate,
    Sort,
    Limit,
)
from omnidf.ops.relational import RelationalExecutor
from omnidf.ops.semantic_filter import SemanticFilterExecutor
from omnidf.ops.semantic_map import SemanticMapExecutor
from omnidf.ops.semantic_join import SemanticJoinExecutor
from omnidf.ops.semantic_dedup import SemanticDedupExecutor

if TYPE_CHECKING:
    from omnidf.models import LLMClient


class Executor:
    """
    Executes query plans.
    
    Orchestrates execution by delegating to specialized operator executors:
    - RelationalExecutor: Pandas-based relational operations
    - SemanticFilterExecutor, SemanticMapExecutor, etc.: Semantic operations
    """
    
    def __init__(
        self,
        llm_client: Optional["LLMClient"] = None,
        use_mock: bool = True,
    ):
        """
        Initialize executor.
        
        Args:
            llm_client: LLM client for semantic operations (uses default if None)
            use_mock: If True, use mock implementations for semantic ops
        """
        self._llm_client = llm_client
        self.use_mock = use_mock
        self._cache: Dict[str, pd.DataFrame] = {}
        
        # Initialize operator executors
        self._relational = RelationalExecutor()
        self._semantic_filter = SemanticFilterExecutor(llm_client=llm_client, use_mock=use_mock)
        self._semantic_map = SemanticMapExecutor(llm_client=llm_client, use_mock=use_mock)
        self._semantic_join = SemanticJoinExecutor(llm_client=llm_client, use_mock=use_mock)
        self._semantic_dedup = SemanticDedupExecutor(llm_client=llm_client, use_mock=use_mock)
    
    @property
    def llm_client(self) -> "LLMClient":
        """Get the LLM client, using default if not set."""
        if self._llm_client is None:
            from omnidf.settings import get_client
            return get_client()
        return self._llm_client
    
    # Backward compatibility alias
    @property
    def use_mock_llm(self) -> bool:
        """Backward compatibility alias for use_mock."""
        return self.use_mock
    
    @use_mock_llm.setter
    def use_mock_llm(self, value: bool):
        """Backward compatibility alias for use_mock."""
        self.use_mock = value
        # Update all semantic executors
        self._semantic_filter.use_mock = value
        self._semantic_map.use_mock = value
        self._semantic_join.use_mock = value
        self._semantic_dedup.use_mock = value
    
    def execute(self, plan: PlanNode) -> pd.DataFrame:
        """
        Execute a query plan and return the result.
        
        Args:
            plan: The query plan to execute
        
        Returns:
            pandas DataFrame with the query results
        """
        return self._execute_node(plan)
    
    def _execute_node(self, node: PlanNode) -> pd.DataFrame:
        """Execute a single node, recursively executing children first."""
        
        # Check cache
        if node.node_id in self._cache:
            return self._cache[node.node_id]
        
        # Dispatch based on node type
        if isinstance(node, Source):
            result = self._execute_source(node)
        elif isinstance(node, Filter):
            result = self._execute_filter(node)
        elif isinstance(node, SemanticFilter):
            result = self._execute_semantic_filter(node)
        elif isinstance(node, SemanticMap):
            result = self._execute_semantic_map(node)
        elif isinstance(node, SemanticJoin):
            result = self._execute_semantic_join(node)
        elif isinstance(node, SemanticDedup):
            result = self._execute_semantic_dedup(node)
        elif isinstance(node, Join):
            result = self._execute_join(node)
        elif isinstance(node, Project):
            result = self._execute_project(node)
        elif isinstance(node, Aggregate):
            result = self._execute_aggregate(node)
        elif isinstance(node, Sort):
            result = self._execute_sort(node)
        elif isinstance(node, Limit):
            result = self._execute_limit(node)
        else:
            raise ValueError(f"Unknown node type: {type(node)}")
        
        # Cache result
        self._cache[node.node_id] = result
        return result
    
    def _execute_source(self, node: Source) -> pd.DataFrame:
        """Execute a Source node."""
        return self._relational.execute_source(node)
    
    def _execute_filter(self, node: Filter) -> pd.DataFrame:
        """Execute a relational Filter node."""
        child_df = self._execute_node(node.child)
        return self._relational.execute_filter(child_df, node)
    
    def _execute_semantic_filter(self, node: SemanticFilter) -> pd.DataFrame:
        """Execute a SemanticFilter node."""
        child_df = self._execute_node(node.child)
        return self._semantic_filter.execute(child_df, node)
    
    def _execute_semantic_map(self, node: SemanticMap) -> pd.DataFrame:
        """Execute a SemanticMap node."""
        child_df = self._execute_node(node.child)
        return self._semantic_map.execute(child_df, node)
    
    def _execute_semantic_join(self, node: SemanticJoin) -> pd.DataFrame:
        """Execute a SemanticJoin node."""
        left_df = self._execute_node(node.left)
        right_df = self._execute_node(node.right)
        return self._semantic_join.execute(left_df, right_df, node)
    
    def _execute_semantic_dedup(self, node: SemanticDedup) -> pd.DataFrame:
        """Execute a SemanticDedup node."""
        child_df = self._execute_node(node.child)
        return self._semantic_dedup.execute(child_df, node)
    
    def _execute_join(self, node: Join) -> pd.DataFrame:
        """Execute a relational Join node."""
        left_df = self._execute_node(node.left)
        right_df = self._execute_node(node.right)
        return self._relational.execute_join(left_df, right_df, node)
    
    def _execute_project(self, node: Project) -> pd.DataFrame:
        """Execute a Project node."""
        child_df = self._execute_node(node.child)
        return self._relational.execute_project(child_df, node)
    
    def _execute_aggregate(self, node: Aggregate) -> pd.DataFrame:
        """Execute an Aggregate node."""
        child_df = self._execute_node(node.child)
        return self._relational.execute_aggregate(child_df, node)
    
    def _execute_sort(self, node: Sort) -> pd.DataFrame:
        """Execute a Sort node."""
        child_df = self._execute_node(node.child)
        return self._relational.execute_sort(child_df, node)
    
    def _execute_limit(self, node: Limit) -> pd.DataFrame:
        """Execute a Limit node."""
        child_df = self._execute_node(node.child)
        return self._relational.execute_limit(child_df, node)
    
    def clear_cache(self):
        """Clear the execution cache."""
        self._cache.clear()
