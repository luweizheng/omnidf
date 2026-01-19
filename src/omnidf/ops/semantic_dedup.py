"""
Semantic Dedup Operator Implementation.

Removes duplicate rows based on semantic similarity evaluated by LLM or other methods.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

from omnidf.ops.base import SemanticOperatorBase
from omnidf.plan import SemanticDedup

if TYPE_CHECKING:
    from omnidf.models import LLMClient


class SemanticDedupExecutor(SemanticOperatorBase):
    """
    Executes semantic deduplication operations.
    
    Removes semantically duplicate rows from a DataFrame.
    Uses batch LLM calls for efficiency.
    
    Note: This is O(n^2) in comparisons. For large datasets, consider
    using embedding-based clustering first to reduce comparisons.
    """
    
    SYSTEM_PROMPT = (
        "You are a duplicate detection assistant. Your task is to determine if two data rows "
        "are semantically duplicates. Respond with ONLY 'yes' or 'no'."
    )
    
    @classmethod
    def format_prompts(
        cls,
        instruction: str,
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        input_columns: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Format prompts for batch semantic dedup operation.
        """
        prompts = []
        
        for row1, row2 in pairs:
            if input_columns:
                context1 = {col: row1[col] for col in input_columns if col in row1}
                context2 = {col: row2[col] for col in input_columns if col in row2}
            else:
                context1 = row1
                context2 = row2
            
            prompt = f"""Determine if these two rows are semantic duplicates based on: "{instruction}"

Row 1: {context1}
Row 2: {context2}

Answer with only 'yes' or 'no'."""
            prompts.append(prompt)
        
        return prompts
    
    @classmethod
    def parse_response(cls, response: str) -> bool:
        """Parse a yes/no response from LLM."""
        answer = response.lower().strip()
        return answer == 'yes' or answer.startswith('yes')
    
    def execute(
        self,
        df: pd.DataFrame,
        node: SemanticDedup,
    ) -> pd.DataFrame:
        """
        Execute semantic dedup on DataFrame.
        
        Args:
            df: Input DataFrame
            node: SemanticDedup plan node with instruction and columns
        
        Returns:
            DataFrame with semantic duplicates removed
        """
        if self.use_mock:
            return self.execute_mock(df, node)
        
        if df.empty or len(df) <= 1:
            return df
        
        records = df.to_dict('records')
        n = len(records)
        
        # Track which rows to keep (not marked as duplicates)
        keep_mask = [True] * n
        
        # Build pairs for comparison (only compare with rows not yet marked as duplicates)
        pairs = []
        pair_indices = []
        
        for i in range(n):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, n):
                if not keep_mask[j]:
                    continue
                pairs.append((records[i], records[j]))
                pair_indices.append((i, j))
        
        if not pairs:
            return df
        
        # Generate prompts
        instruction = node.user_instruction or "These rows represent the same entity or concept"
        prompts = self.format_prompts(
            instruction=instruction,
            pairs=pairs,
            input_columns=node.input_columns if node.input_columns else None,
        )
        
        # Batch LLM call
        responses = self.llm_client.batch_complete(prompts, model=node.model)
        
        # Mark duplicates
        for (i, j), resp in zip(pair_indices, responses):
            if self.parse_response(resp.content):
                # j is a duplicate of i, mark j for removal
                keep_mask[j] = False
        
        return df[keep_mask].reset_index(drop=True)
    
    def execute_mock(
        self,
        df: pd.DataFrame,
        node: SemanticDedup,
    ) -> pd.DataFrame:
        """
        Mock implementation for testing.
        
        Uses Pandas drop_duplicates on input columns.
        """
        if node.input_columns:
            return df.drop_duplicates(subset=node.input_columns).reset_index(drop=True)
        return df.drop_duplicates().reset_index(drop=True)
    
    def execute_single_pair(
        self,
        row1: Dict[str, Any],
        row2: Dict[str, Any],
        instruction: str,
        input_columns: Optional[List[str]] = None,
        model: Optional[str] = None,
    ) -> bool:
        """
        Check if two rows are semantic duplicates.
        
        Args:
            row1: First row dictionary
            row2: Second row dictionary
            instruction: Dedup instruction
            input_columns: Columns to compare
            model: Model to use for LLM call
        
        Returns:
            True if rows are semantic duplicates
        """
        if self.use_mock:
            return False
        
        prompts = self.format_prompts(
            instruction=instruction,
            pairs=[(row1, row2)],
            input_columns=input_columns,
        )
        
        responses = self.llm_client.batch_complete(prompts, model=model)
        return self.parse_response(responses[0].content)
