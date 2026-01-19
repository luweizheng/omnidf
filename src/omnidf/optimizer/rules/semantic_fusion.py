"""
Semantic operator fusion rules.

Fuse consecutive semantic operations into single LLM calls.
"""

from omnidf.optimizer.base import Rule
from omnidf.plan import PlanNode, SemanticFilter, SemanticMap


class SemanticMapFusion(Rule):
    """
    Fuse consecutive SemanticMap operations into a single LLM call.
    
    Before: SemanticMap -> SemanticMap
    After:  FusedSemanticMap (single LLM call)
    
    This reduces the number of LLM API calls.
    """
    
    @property
    def name(self) -> str:
        return "SemanticMapFusion"
    
    def matches(self, node: PlanNode) -> bool:
        """Match consecutive SemanticMap nodes with the same model."""
        if not isinstance(node, SemanticMap):
            return False
        
        child = node.child
        if not isinstance(child, SemanticMap):
            return False
        
        # Only fuse if using the same model
        return node.model == child.model
    
    def apply(self, node: PlanNode) -> PlanNode:
        """
        Fuse two SemanticMap operations.
        
        The fused instruction combines both transformations.
        """
        assert isinstance(node, SemanticMap)
        child = node.child
        assert isinstance(child, SemanticMap)
        
        # Combine instructions
        fused_instruction = (
            f"Perform the following transformations:\n"
            f"1. {child.user_instruction} (output as '{child.output_column}')\n"
            f"2. {node.user_instruction} (output as '{node.output_column}')"
        )
        
        # Combine input columns (child's inputs + any additional from parent)
        combined_inputs = list(child.input_columns)
        for col in node.input_columns:
            if col not in combined_inputs and col != child.output_column:
                combined_inputs.append(col)
        
        # Create fused node - outputs both columns
        fused_node = SemanticMap(
            child=child.child,
            user_instruction=fused_instruction,
            input_columns=combined_inputs,
            output_column=f"{child.output_column},{node.output_column}",
            model=node.model,
        )
        
        return fused_node


class SemanticFilterFusion(Rule):
    """
    Fuse consecutive SemanticFilter operations into a single LLM call.
    
    Before: SemanticFilter -> SemanticFilter
    After:  FusedSemanticFilter (single LLM call with combined conditions)
    """
    
    @property
    def name(self) -> str:
        return "SemanticFilterFusion"
    
    def matches(self, node: PlanNode) -> bool:
        """Match consecutive SemanticFilter nodes with the same model."""
        if not isinstance(node, SemanticFilter):
            return False
        
        child = node.child
        if not isinstance(child, SemanticFilter):
            return False
        
        return node.model == child.model
    
    def apply(self, node: PlanNode) -> PlanNode:
        """Fuse two SemanticFilter operations."""
        assert isinstance(node, SemanticFilter)
        child = node.child
        assert isinstance(child, SemanticFilter)
        
        # Combine instructions with AND semantics
        fused_instruction = (
            f"The row must satisfy ALL of the following conditions:\n"
            f"1. {child.user_instruction}\n"
            f"2. {node.user_instruction}"
        )
        
        # Combine input columns
        combined_inputs = list(child.input_columns)
        for col in node.input_columns:
            if col not in combined_inputs:
                combined_inputs.append(col)
        
        fused_node = SemanticFilter(
            child=child.child,
            user_instruction=fused_instruction,
            input_columns=combined_inputs,
            model=node.model,
        )
        
        return fused_node
