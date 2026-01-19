"""
Semantic Join Operator Implementation.

Joins rows based on semantic conditions evaluated by LLM or other methods.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

from omnidf.ops.base import SemanticOperatorBase
from omnidf.plan import SemanticJoin

if TYPE_CHECKING:
    from omnidf.models import LLMClient


class SemanticJoinExecutor(SemanticOperatorBase):
    """
    Executes semantic join operations.
    
    Joins two DataFrames based on semantic conditions.
    Uses batch LLM calls for efficiency.
    
    Note: This is O(n*m) in LLM calls. Consider using SemanticJoinDecomposition
    optimization to reduce to O(n+m) when possible.
    """
    
    SYSTEM_PROMPT = (
        "You are a data matching assistant. Your task is to determine if two data rows "
        "should be joined based on a semantic condition. Respond with ONLY 'yes' or 'no'."
    )
    
    @classmethod
    def format_prompts(
        cls,
        instruction: str,
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
    ) -> List[str]:
        """
        Format prompts for batch semantic join operation.
        """
        prompts = []
        
        for left_row, right_row in pairs:
            processed_instruction = instruction
            for col, val in left_row.items():
                processed_instruction = processed_instruction.replace(f"{{{col}:left}}", str(val))
            for col, val in right_row.items():
                processed_instruction = processed_instruction.replace(f"{{{col}:right}}", str(val))
            
            prompt = f"""Determine if these two rows should be joined based on: "{processed_instruction}"

Left row: {left_row}
Right row: {right_row}

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
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        node: SemanticJoin,
    ) -> pd.DataFrame:
        """
        Execute semantic join on two DataFrames.
        
        Args:
            left_df: Left DataFrame
            right_df: Right DataFrame
            node: SemanticJoin plan node with join instruction
        
        Returns:
            Joined DataFrame
        """
        if self.use_mock:
            return self.execute_mock(left_df, right_df, node)
        
        if left_df.empty or right_df.empty:
            left_cols = [f"{c}_left" for c in left_df.columns]
            right_cols = [f"{c}_right" for c in right_df.columns]
            return pd.DataFrame(columns=left_cols + right_cols)
        
        # Build all pairs for batch processing
        pairs = []
        pair_indices = []  # Track (left_idx, right_idx) for each pair
        
        left_records = left_df.to_dict('records')
        right_records = right_df.to_dict('records')
        
        for left_idx, left_row in enumerate(left_records):
            for right_idx, right_row in enumerate(right_records):
                pairs.append((left_row, right_row))
                pair_indices.append((left_idx, right_idx))
        
        # Generate prompts
        prompts = self.format_prompts(
            instruction=node.join_instruction,
            pairs=pairs,
        )
        
        # Batch LLM call
        responses = self.llm_client.batch_complete(prompts, model=node.model)
        
        # Build result from matching pairs
        results = []
        for (left_idx, right_idx), resp in zip(pair_indices, responses):
            if self.parse_response(resp.content):
                left_row = left_records[left_idx]
                right_row = right_records[right_idx]
                combined = {
                    **{f"{k}_left": v for k, v in left_row.items()},
                    **{f"{k}_right": v for k, v in right_row.items()}
                }
                results.append(combined)
        
        if results:
            return pd.DataFrame(results)
        
        left_cols = [f"{c}_left" for c in left_df.columns]
        right_cols = [f"{c}_right" for c in right_df.columns]
        return pd.DataFrame(columns=left_cols + right_cols)
    
    def execute_mock(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        node: SemanticJoin,
    ) -> pd.DataFrame:
        """
        Mock implementation for testing.
        
        Returns cross join of both DataFrames.
        """
        left_df = left_df.copy()
        right_df = right_df.copy()
        left_df['_key'] = 1
        right_df['_key'] = 1
        result = pd.merge(left_df, right_df, on='_key', suffixes=('_left', '_right'))
        result = result.drop('_key', axis=1)
        return result
    
    def execute_single_pair(
        self,
        left_row: Dict[str, Any],
        right_row: Dict[str, Any],
        instruction: str,
        model: Optional[str] = None,
    ) -> bool:
        """
        Execute semantic join check on a single pair of rows.
        
        Args:
            left_row: Left row dictionary
            right_row: Right row dictionary
            instruction: Join instruction
            model: Model to use for LLM call
        
        Returns:
            True if rows should be joined
        """
        if self.use_mock:
            return True
        
        prompts = self.format_prompts(
            instruction=instruction,
            pairs=[(left_row, right_row)],
        )
        
        responses = self.llm_client.batch_complete(prompts, model=model)
        return self.parse_response(responses[0].content)
