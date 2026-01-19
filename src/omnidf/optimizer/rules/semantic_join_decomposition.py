"""
SemanticJoinDecomposition optimization rule.

Decompose SemanticJoin into SemanticMap + Relational Join to reduce LLM calls.
"""

from omnidf.optimizer.base import Rule
from omnidf.plan import PlanNode, SemanticJoin, SemanticMap, Join


class SemanticJoinDecomposition(Rule):
    """
    Decompose SemanticJoin into SemanticMap + Relational Join.
    
    Before: SemanticJoin(left, right, "same sentiment")
    After:  Join(SemanticMap(left, "extract sentiment"), 
                 SemanticMap(right, "extract sentiment"),
                 on="sentiment")
    
    This is beneficial when:
    - The semantic comparison can be decomposed into attribute extraction
    - The extracted attributes can be compared with equality
    
    This reduces O(n*m) LLM calls to O(n+m) calls.
    """
    
    @property
    def name(self) -> str:
        return "SemanticJoinDecomposition"
    
    def matches(self, node: PlanNode) -> bool:
        """
        Match SemanticJoin nodes that can be decomposed.
        
        Heuristic: Look for patterns like "same X" or "equal X" in the instruction.
        """
        if not isinstance(node, SemanticJoin):
            return False
        
        instruction = node.join_instruction.lower()
        
        # Simple heuristic - look for decomposable patterns
        decomposable_patterns = [
            "same sentiment",
            "same category",
            "same topic",
            "equal sentiment",
            "matching sentiment",
        ]
        
        return any(pattern in instruction for pattern in decomposable_patterns)
    
    def apply(self, node: PlanNode) -> PlanNode:
        """
        Decompose SemanticJoin into SemanticMap + Join.
        """
        assert isinstance(node, SemanticJoin)
        
        # Extract the attribute being compared
        instruction = node.join_instruction.lower()
        
        # Determine what attribute to extract
        if "sentiment" in instruction:
            attribute = "sentiment"
            extract_instruction = "Extract the sentiment of this text. Output exactly one of: positive, negative, neutral"
        elif "category" in instruction:
            attribute = "category"
            extract_instruction = "Extract the category of this item."
        elif "topic" in instruction:
            attribute = "topic"
            extract_instruction = "Extract the main topic of this text."
        else:
            attribute = "extracted_value"
            extract_instruction = "Extract the key attribute for comparison."
        
        # Create SemanticMap for left side
        left_map = SemanticMap(
            child=node.left,
            user_instruction=extract_instruction,
            input_columns=[],  # Will use all columns
            output_column=f"_left_{attribute}",
            model=node.model,
        )
        
        # Create SemanticMap for right side
        right_map = SemanticMap(
            child=node.right,
            user_instruction=extract_instruction,
            input_columns=[],
            output_column=f"_right_{attribute}",
            model=node.model,
        )
        
        # Create relational join on the extracted attribute
        join_node = Join(
            left=left_map,
            right=right_map,
            left_on=f"_left_{attribute}",
            right_on=f"_right_{attribute}",
            how="inner",
        )
        
        return join_node
