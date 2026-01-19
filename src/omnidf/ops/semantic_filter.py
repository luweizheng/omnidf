"""
Semantic Filter Operator Implementation.

Filters rows based on semantic conditions evaluated by LLM or other methods.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

from omnidf.ops.base import SemanticOperatorBase
from omnidf.plan import SemanticFilter

if TYPE_CHECKING:
    from omnidf.models import LLMClient


class SemanticFilterExecutor(SemanticOperatorBase):
    """
    Executes semantic filter operations.
    
    Filters DataFrame rows based on semantic conditions.
    Uses batch LLM calls for efficiency.
    """
    
    SYSTEM_PROMPT = (
        "You are a data filtering assistant. Your task is to determine if a data row "
        "matches a given condition. Respond with ONLY 'yes' or 'no'."
    )
    
    @classmethod
    def format_prompts(
        cls,
        instruction: str,
        rows: List[Dict[str, Any]],
        input_columns: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Format prompts for batch semantic filter operation.
        """
        prompts = []
        for row_data in rows:
            if input_columns:
                context = {col: row_data[col] for col in input_columns if col in row_data}
            else:
                context = row_data
            
            prompt = f"""Determine if the following data matches this condition: "{instruction}"

Data: {context}

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
        node: SemanticFilter,
    ) -> pd.DataFrame:
        """
        Execute semantic filter on DataFrame.
        
        Args:
            df: Input DataFrame
            node: SemanticFilter plan node with instruction and columns
        
        Returns:
            Filtered DataFrame
        """
        if self.use_mock:
            return self.execute_mock(df, node)
        
        if df.empty:
            return df
        
        # Convert DataFrame rows to list of dicts for prompt formatting
        rows = df.to_dict('records')
        
        # Generate prompts
        prompts = self.format_prompts(
            instruction=node.user_instruction,
            rows=rows,
            input_columns=node.input_columns if node.input_columns else None,
        )
        
        # Batch LLM call
        responses = self.llm_client.batch_complete(prompts, model=node.model)
        
        # Filter rows based on responses
        mask = [
            self.parse_response(resp.content)
            for resp in responses
        ]
        
        return df[mask].reset_index(drop=True)
    
    def execute_mock(
        self,
        df: pd.DataFrame,
        node: SemanticFilter,
    ) -> pd.DataFrame:
        """
        Mock implementation for testing.
        
        Returns all rows (no actual filtering).
        """
        return df
    
    def execute_single(
        self,
        row: Dict[str, Any],
        instruction: str,
        input_columns: Optional[List[str]] = None,
        model: Optional[str] = None,
    ) -> bool:
        """
        Execute semantic filter on a single row.
        
        Args:
            row: Dictionary representing a single row
            instruction: Filter instruction
            input_columns: Columns to include in context
            model: Model to use for LLM call
        
        Returns:
            True if row matches the filter condition
        """
        if self.use_mock:
            return True
        
        prompts = self.format_prompts(
            instruction=instruction,
            rows=[row],
            input_columns=input_columns,
        )
        
        responses = self.llm_client.batch_complete(prompts, model=model)
        return self.parse_response(responses[0].content)
