"""
Semantic Map Operator Implementation.

Transforms or extracts information from rows using LLM or other methods.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

from omnidf.ops.base import SemanticOperatorBase
from omnidf.plan import SemanticMap

if TYPE_CHECKING:
    from omnidf.models import LLMClient


class SemanticMapExecutor(SemanticOperatorBase):
    """
    Executes semantic map operations.
    
    Transforms DataFrame by adding new columns based on LLM extraction/transformation.
    Uses batch LLM calls for efficiency.
    """
    
    SYSTEM_PROMPT = (
        "You are a data extraction assistant. Your task is to extract or transform "
        "information from data according to the given instruction. "
        "Provide ONLY the extracted/transformed value, nothing else."
    )
    
    @classmethod
    def format_prompts(
        cls,
        instruction: str,
        rows: List[Dict[str, Any]],
        input_columns: Optional[List[str]] = None,
        output_column: Optional[str] = None,
    ) -> List[str]:
        """
        Format prompts for batch semantic map operation.
        """
        prompts = []
        output_hint = f" Output the result for '{output_column}'." if output_column else ""
        
        for row_data in rows:
            if input_columns:
                context = {col: row_data[col] for col in input_columns if col in row_data}
            else:
                context = row_data
            
            prompt = f"""Based on the following data, {instruction}{output_hint}

Data: {context}

Provide only the extracted/transformed value."""
            prompts.append(prompt)
        
        return prompts
    
    def execute(
        self,
        df: pd.DataFrame,
        node: SemanticMap,
    ) -> pd.DataFrame:
        """
        Execute semantic map on DataFrame.
        
        Args:
            df: Input DataFrame
            node: SemanticMap plan node with instruction and columns
        
        Returns:
            DataFrame with new column(s) added
        """
        if self.use_mock:
            return self.execute_mock(df, node)
        
        if df.empty:
            result = df.copy()
            result[node.output_column] = []
            return result
        
        # Convert DataFrame rows to list of dicts for prompt formatting
        rows = df.to_dict('records')
        
        # Generate prompts
        prompts = self.format_prompts(
            instruction=node.user_instruction,
            rows=rows,
            input_columns=node.input_columns if node.input_columns else None,
            output_column=node.output_column,
        )
        
        # Batch LLM call
        responses = self.llm_client.batch_complete(prompts, model=node.model)
        
        # Extract values from responses
        new_values = [resp.content.strip() for resp in responses]
        
        result = df.copy()
        
        # Handle multiple output columns (from fusion)
        output_cols = node.output_column.split(',')
        if len(output_cols) == 1:
            result[node.output_column] = new_values
        else:
            # For fused maps, assign same value to all columns
            # (Real implementation would parse structured output)
            for col in output_cols:
                col = col.strip()
                result[col] = new_values
        
        return result
    
    def execute_mock(
        self,
        df: pd.DataFrame,
        node: SemanticMap,
    ) -> pd.DataFrame:
        """
        Mock implementation for testing.
        
        Adds output column with placeholder values.
        """
        result = df.copy()
        
        # Handle multiple output columns (from fusion)
        output_cols = node.output_column.split(',')
        for col in output_cols:
            col = col.strip()
            result[col] = "mock_value"
        
        return result
    
    def execute_single(
        self,
        row: Dict[str, Any],
        instruction: str,
        input_columns: Optional[List[str]] = None,
        output_column: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Execute semantic map on a single row.
        
        Args:
            row: Dictionary representing a single row
            instruction: Transformation instruction
            input_columns: Columns to include in context
            output_column: Name of output column
            model: Model to use for LLM call
        
        Returns:
            Extracted/transformed value
        """
        if self.use_mock:
            return "mock_value"
        
        prompts = self.format_prompts(
            instruction=instruction,
            rows=[row],
            input_columns=input_columns,
            output_column=output_column,
        )
        
        responses = self.llm_client.batch_complete(prompts, model=model)
        return responses[0].content.strip()
