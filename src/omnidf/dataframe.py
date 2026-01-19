"""
OmniDataFrame - Lazy evaluation DataFrame with semantic operators.

Design Principles:
1. Lazy evaluation - operations build a DAG, execution happens on collect()
2. Pandas-compatible API for relational operations
3. Semantic operators for LLM-powered transformations
4. Immutable - each operation returns a new DataFrame
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from omnidf.plan import (
    PlanNode,
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
    plan_to_dict,
    plan_to_json,
)


class DataFrame:
    """
    OmniDataFrame - A lazy-evaluation DataFrame with semantic operators.
    
    This class wraps a query plan (DAG) and provides a pandas-like API.
    Operations are not executed until collect() is called.
    
    Example:
        >>> import omnidf as odf
        >>> import pandas as pd
        >>> data = pd.DataFrame({'name': ['Alice', 'Bob'], 'age': [25, 30]})
        >>> df = odf.DataFrame(data)
        >>> result = df[df['age'] > 25].collect()
    """
    
    def __init__(
        self,
        data: Optional[Union[pd.DataFrame, PlanNode]] = None,
        _plan: Optional[PlanNode] = None,
    ):
        """
        Create an OmniDataFrame.
        
        Args:
            data: Either a pandas DataFrame or a PlanNode
            _plan: Internal use - directly set the plan
        """
        if _plan is not None:
            self._plan = _plan
        elif isinstance(data, PlanNode):
            self._plan = data
        elif isinstance(data, pd.DataFrame):
            self._plan = Source(data=data)
        elif data is None:
            self._plan = Source(data=pd.DataFrame())
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        # Cache for the underlying data (used for predicate building)
        self._data_cache: Optional[pd.DataFrame] = None
        if isinstance(data, pd.DataFrame):
            self._data_cache = data
    
    @property
    def plan(self) -> PlanNode:
        """Get the query plan."""
        return self._plan
    
    @property
    def _data(self) -> pd.DataFrame:
        """
        Access to underlying data for predicate building.
        
        This is used for expressions like df[df._data['col'] > 5].
        Note: This returns the SOURCE data, not the result of the plan.
        """
        if self._data_cache is not None:
            return self._data_cache
        # Walk up to find source
        node = self._plan
        while node.children:
            node = node.children[0]
        if isinstance(node, Source) and node.data is not None:
            return node.data
        return pd.DataFrame()
    
    # ==================== Relational Operations ====================
    
    def __getitem__(self, key: Union[str, List[str], pd.Series]) -> DataFrame:
        """
        Select columns or filter rows.
        
        - df['col'] or df[['col1', 'col2']] -> Project
        - df[df['col'] > 5] -> Filter (when key is a boolean Series)
        """
        if isinstance(key, str):
            # Single column selection
            new_plan = Project(child=self._plan, columns=[key])
            return DataFrame(_plan=new_plan)
        elif isinstance(key, list):
            # Multiple column selection
            new_plan = Project(child=self._plan, columns=key)
            return DataFrame(_plan=new_plan)
        elif isinstance(key, pd.Series) and key.dtype == bool:
            # Boolean filter - need to capture the predicate
            # We store the predicate info for optimization
            predicate_str = self._extract_predicate_str(key)
            columns_used = self._extract_columns_from_series(key)
            
            new_plan = Filter(
                child=self._plan,
                predicate=predicate_str,
                predicate_expr=key,
                columns_used=columns_used,
            )
            return DataFrame(_plan=new_plan)
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")
    
    def _extract_predicate_str(self, series: pd.Series) -> str:
        """Extract a string representation of the predicate from a boolean Series."""
        # This is a simplified version - in practice, we'd need more sophisticated
        # predicate extraction or require users to pass predicates explicitly
        if hasattr(series, 'name') and series.name:
            return f"filter on {series.name}"
        return "boolean_filter"
    
    def _extract_columns_from_series(self, series: pd.Series) -> List[str]:
        """Extract column names used in a predicate."""
        # Simplified - would need AST analysis for complex predicates
        if hasattr(series, 'name') and series.name:
            return [str(series.name)]
        return []
    
    def filter(
        self,
        predicate: Union[str, Callable[[pd.DataFrame], pd.Series]],
        columns_used: Optional[List[str]] = None,
    ) -> DataFrame:
        """
        Filter rows based on a predicate.
        
        Args:
            predicate: Either a string expression or a callable
            columns_used: Columns referenced in the predicate
        
        Example:
            >>> df.filter("age > 25", columns_used=["age"])
            >>> df.filter(lambda df: df['age'] > 25, columns_used=["age"])
        """
        new_plan = Filter(
            child=self._plan,
            predicate=predicate if isinstance(predicate, str) else str(predicate),
            predicate_expr=predicate,
            columns_used=columns_used or [],
        )
        return DataFrame(_plan=new_plan)
    
    def select(self, *columns: str) -> DataFrame:
        """Select specific columns."""
        new_plan = Project(child=self._plan, columns=list(columns))
        return DataFrame(_plan=new_plan)
    
    def merge(
        self,
        right: DataFrame,
        on: Optional[Union[str, List[str]]] = None,
        left_on: Optional[Union[str, List[str]]] = None,
        right_on: Optional[Union[str, List[str]]] = None,
        how: str = "inner",
        suffixes: tuple = ("_x", "_y"),
    ) -> DataFrame:
        """
        Merge with another DataFrame (relational join).
        
        Args:
            right: Right DataFrame to join with
            on: Column(s) to join on (if same name in both)
            left_on: Column(s) from left DataFrame
            right_on: Column(s) from right DataFrame
            how: Join type - 'inner', 'left', 'right', 'outer', 'cross'
            suffixes: Suffixes for overlapping column names
        """
        new_plan = Join(
            left=self._plan,
            right=right._plan,
            on=on,
            left_on=left_on,
            right_on=right_on,
            how=how,
            suffixes=suffixes,
        )
        return DataFrame(_plan=new_plan)
    
    def join(
        self,
        right: DataFrame,
        on: Optional[Union[str, List[str]]] = None,
        how: str = "inner",
    ) -> DataFrame:
        """Alias for merge."""
        return self.merge(right, on=on, how=how)
    
    def groupby(self, by: Union[str, List[str]]) -> GroupBy:
        """Group by columns for aggregation."""
        if isinstance(by, str):
            by = [by]
        return GroupBy(self, by)
    
    def sort_values(
        self,
        by: Union[str, List[str]],
        ascending: Union[bool, List[bool]] = True,
    ) -> DataFrame:
        """Sort by columns."""
        if isinstance(by, str):
            by = [by]
        new_plan = Sort(child=self._plan, by=by, ascending=ascending)
        return DataFrame(_plan=new_plan)
    
    def head(self, n: int = 5) -> DataFrame:
        """Get first n rows."""
        new_plan = Limit(child=self._plan, n=n)
        return DataFrame(_plan=new_plan)
    
    def limit(self, n: int, offset: int = 0) -> DataFrame:
        """Limit rows with optional offset."""
        new_plan = Limit(child=self._plan, n=n, offset=offset)
        return DataFrame(_plan=new_plan)
    
    # ==================== Semantic Operations ====================
    
    def semantic_filter(
        self,
        user_instruction: str,
        input_columns: Optional[List[str]] = None,
        model: str = "gpt-4.1",
    ) -> DataFrame:
        """
        Filter rows using LLM-based semantic understanding.
        
        Args:
            user_instruction: Natural language description of filter condition
            input_columns: Columns to pass to the LLM
            model: LLM model to use
        
        Example:
            >>> df.semantic_filter(
            ...     "The description mentions renewable energy",
            ...     input_columns=["description"]
            ... )
        """
        new_plan = SemanticFilter(
            child=self._plan,
            user_instruction=user_instruction,
            input_columns=input_columns or [],
            model=model,
        )
        return DataFrame(_plan=new_plan)
    
    def sem_filter(
        self,
        user_instruction: str,
        input_columns: Optional[List[str]] = None,
        model: str = "gpt-4.1",
    ) -> DataFrame:
        """Alias for semantic_filter."""
        return self.semantic_filter(user_instruction, input_columns, model)
    
    def semantic_map(
        self,
        user_instruction: str,
        output_column: str,
        input_columns: Optional[List[str]] = None,
        model: str = "gpt-4.1",
    ) -> DataFrame:
        """
        Transform/extract information using LLM.
        
        Args:
            user_instruction: Natural language description of transformation
            output_column: Name of the new column to create
            input_columns: Columns to pass to the LLM
            model: LLM model to use
        
        Example:
            >>> df.semantic_map(
            ...     "Extract the sentiment (positive/negative/neutral)",
            ...     output_column="sentiment",
            ...     input_columns=["review"]
            ... )
        """
        new_plan = SemanticMap(
            child=self._plan,
            user_instruction=user_instruction,
            input_columns=input_columns or [],
            output_column=output_column,
            model=model,
        )
        return DataFrame(_plan=new_plan)
    
    def sem_map(
        self,
        user_instruction: str,
        output_column: str,
        input_columns: Optional[List[str]] = None,
        model: str = "gpt-4.1",
    ) -> DataFrame:
        """Alias for semantic_map."""
        return self.semantic_map(user_instruction, output_column, input_columns, model)
    
    def semantic_join(
        self,
        right: DataFrame,
        join_instruction: str,
        model: str = "gpt-4.1",
    ) -> DataFrame:
        """
        Join with another DataFrame using LLM-based matching.
        
        Args:
            right: Right DataFrame to join with
            join_instruction: Natural language description of join condition
            model: LLM model to use
        
        Example:
            >>> reviews.semantic_join(
            ...     reviews,
            ...     join_instruction='Reviews express the same sentiment'
            ... )
        """
        new_plan = SemanticJoin(
            left=self._plan,
            right=right._plan,
            join_instruction=join_instruction,
            model=model,
        )
        return DataFrame(_plan=new_plan)
    
    def sem_join(
        self,
        right: DataFrame,
        join_instruction: str,
        model: str = "gpt-4.1",
    ) -> DataFrame:
        """Alias for semantic_join."""
        return self.semantic_join(right, join_instruction, model)
    
    def semantic_dedup(
        self,
        user_instruction: str,
        input_columns: Optional[List[str]] = None,
        model: str = "gpt-4.1",
    ) -> DataFrame:
        """
        Deduplicate rows using LLM-based semantic similarity.
        
        Args:
            user_instruction: Natural language description of what makes rows duplicates
            input_columns: Columns to consider for deduplication
            model: LLM model to use
        """
        new_plan = SemanticDedup(
            child=self._plan,
            user_instruction=user_instruction,
            input_columns=input_columns or [],
            model=model,
        )
        return DataFrame(_plan=new_plan)
    
    def sem_dedup(
        self,
        user_instruction: str,
        input_columns: Optional[List[str]] = None,
        model: str = "gpt-4.1",
    ) -> DataFrame:
        """Alias for semantic_dedup."""
        return self.semantic_dedup(user_instruction, input_columns, model)
    
    # ==================== Plan Inspection ====================
    
    def explain(self, format: str = "tree") -> str:
        """
        Show the query plan.
        
        Args:
            format: 'tree' for tree view, 'json' for JSON representation
        """
        if format == "json":
            return plan_to_json(self._plan)
        return self._plan.pretty_print()
    
    def to_plan_dict(self) -> Dict[str, Any]:
        """Get the plan as a dictionary (for LLM optimization)."""
        return plan_to_dict(self._plan)
    
    def __repr__(self) -> str:
        return f"OmniDataFrame(plan={self._plan})"
    
    # ==================== Execution ====================
    
    def collect(self, optimize: bool = True) -> pd.DataFrame:
        """
        Execute the query plan and return a pandas DataFrame.
        
        Args:
            optimize: Whether to apply query optimizations before execution
        """
        from omnidf.execution import Executor
        from omnidf.optimizer import Optimizer
        
        plan = self._plan
        if optimize:
            optimizer = Optimizer()
            plan = optimizer.optimize(plan)
        
        executor = Executor()
        return executor.execute(plan)
    
    def execute(self, optimize: bool = True) -> pd.DataFrame:
        """Alias for collect."""
        return self.collect(optimize=optimize)


class GroupBy:
    """GroupBy object for aggregation operations."""
    
    def __init__(self, df: DataFrame, by: List[str]):
        self._df = df
        self._by = by
    
    def agg(self, aggregations: Dict[str, str]) -> DataFrame:
        """
        Aggregate with specified functions.
        
        Args:
            aggregations: Dict mapping column names to aggregation functions
                         e.g., {'age': 'mean', 'score': 'sum'}
        """
        new_plan = Aggregate(
            child=self._df._plan,
            group_by=self._by,
            aggregations=aggregations,
        )
        return DataFrame(_plan=new_plan)
    
    def sum(self) -> DataFrame:
        """Sum all numeric columns."""
        return self.agg({})  # Will be handled by executor
    
    def mean(self) -> DataFrame:
        """Mean of all numeric columns."""
        return self.agg({})
    
    def count(self) -> DataFrame:
        """Count rows in each group."""
        return self.agg({})
