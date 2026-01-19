"""
FilterPushDown optimization rule.

Push relational filters down past semantic operators to reduce LLM calls.
"""

from omnidf.optimizer.base import Rule
from omnidf.plan import PlanNode, Filter, SemanticFilter, SemanticMap


class FilterPushDown(Rule):
    """
    Push relational filters down past semantic operators.
    
    This reduces the number of rows processed by expensive LLM calls.
    
    Before: SemanticFilter -> Filter(relational)
    After:  Filter(relational) -> SemanticFilter
    
    Conditions:
    - The relational filter must not depend on columns produced by the semantic operator
    - The filter columns must be available in the child of the semantic operator
    """
    
    @property
    def name(self) -> str:
        return "FilterPushDown"
    
    def matches(self, node: PlanNode) -> bool:
        """
        Match pattern: Filter on top of SemanticFilter/SemanticMap
        """
        if not isinstance(node, Filter):
            return False
        
        child = node.child
        if child is None:
            return False
        
        # Can push down past semantic operators
        if isinstance(child, (SemanticFilter, SemanticMap)):
            # Check if filter columns are available before the semantic op
            filter_cols = set(node.columns_used)
            
            # For SemanticMap, we cannot push if filter uses the output column
            if isinstance(child, SemanticMap):
                if child.output_column in filter_cols:
                    return False
            
            return True
        
        return False
    
    def apply(self, node: PlanNode) -> PlanNode:
        """
        Push the filter below the semantic operator.
        """
        assert isinstance(node, Filter)
        semantic_op = node.child
        
        # Create new filter with semantic op's child as its child
        new_filter = Filter(
            child=semantic_op.children[0],
            predicate=node.predicate,
            predicate_expr=node.predicate_expr,
            columns_used=node.columns_used,
        )
        
        # Create new semantic op with the filter as its child
        new_semantic = semantic_op.with_children([new_filter])
        
        return new_semantic
