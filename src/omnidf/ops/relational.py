"""
Relational Operators Implementation.

These operators are executed using Pandas, leveraging its vectorized operations
for optimal performance. No row-by-row processing is used.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from omnidf.plan import (
    Source,
    Filter,
    Join,
    Project,
    Aggregate,
    Sort,
    Limit,
)


class RelationalExecutor:
    """
    Executes relational operators using Pandas.
    
    All operations use Pandas' vectorized operations for efficiency.
    """
    
    def execute_source(self, node: Source) -> pd.DataFrame:
        """Execute a Source node."""
        if node.data is not None:
            return node.data.copy()
        elif node.data_ref:
            if node.data_ref.endswith('.csv'):
                return pd.read_csv(node.data_ref)
            elif node.data_ref.endswith('.parquet'):
                return pd.read_parquet(node.data_ref)
            elif node.data_ref.endswith('.json'):
                return pd.read_json(node.data_ref)
            else:
                raise ValueError(f"Unsupported data reference: {node.data_ref}")
        else:
            return pd.DataFrame()
    
    def execute_filter(self, df: pd.DataFrame, node: Filter) -> pd.DataFrame:
        """
        Execute a relational Filter node using Pandas vectorized operations.
        
        Supports:
        - Callable predicates (lambda functions)
        - String predicates (evaluated via df.eval or df.query)
        """
        if node.predicate_expr is not None:
            if callable(node.predicate_expr):
                mask = node.predicate_expr(df)
                return df[mask].reset_index(drop=True)
            elif isinstance(node.predicate_expr, pd.Series):
                try:
                    mask = eval(node.predicate, {"df": df})
                    return df[mask].reset_index(drop=True)
                except Exception:
                    return df
        
        if node.predicate:
            try:
                # Use Pandas eval for vectorized evaluation
                mask = df.eval(node.predicate)
                return df[mask].reset_index(drop=True)
            except Exception:
                try:
                    # Fallback to query
                    return df.query(node.predicate).reset_index(drop=True)
                except Exception:
                    pass
        
        return df
    
    def execute_join(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        node: Join,
    ) -> pd.DataFrame:
        """
        Execute a relational Join node using Pandas merge.
        
        Leverages Pandas' optimized merge implementation.
        """
        return pd.merge(
            left_df,
            right_df,
            on=node.on,
            left_on=node.left_on,
            right_on=node.right_on,
            how=node.how,
            suffixes=node.suffixes,
        )
    
    def execute_project(self, df: pd.DataFrame, node: Project) -> pd.DataFrame:
        """
        Execute a Project node using Pandas column selection.
        """
        available_cols = [c for c in node.columns if c in df.columns]
        if available_cols:
            return df[available_cols]
        return df
    
    def execute_aggregate(self, df: pd.DataFrame, node: Aggregate) -> pd.DataFrame:
        """
        Execute an Aggregate node using Pandas groupby.
        
        Uses Pandas' optimized groupby aggregation.
        """
        if node.group_by:
            grouped = df.groupby(node.group_by)
            if node.aggregations:
                return grouped.agg(node.aggregations).reset_index()
            return grouped.size().reset_index(name='count')
        else:
            if node.aggregations:
                return df.agg(node.aggregations).to_frame().T
            return pd.DataFrame({'count': [len(df)]})
    
    def execute_sort(self, df: pd.DataFrame, node: Sort) -> pd.DataFrame:
        """
        Execute a Sort node using Pandas sort_values.
        """
        return df.sort_values(by=node.by, ascending=node.ascending).reset_index(drop=True)
    
    def execute_limit(self, df: pd.DataFrame, node: Limit) -> pd.DataFrame:
        """
        Execute a Limit node using Pandas iloc.
        """
        return df.iloc[node.offset:node.offset + node.n].reset_index(drop=True)
