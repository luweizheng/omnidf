"""
ProjectionPushDown optimization rule.
"""

from omnidf.optimizer.base import Rule
from omnidf.plan import PlanNode, Project, Filter, SemanticFilter, SemanticMap


class ProjectionPushDown(Rule):
    """
    Push projections down to reduce data size early.
    
    Before: Project -> Filter
    After:  Filter -> Project (if filter doesn't need projected-out columns)
    """
    
    @property
    def name(self) -> str:
        return "ProjectionPushDown"
    
    def matches(self, node: PlanNode) -> bool:
        if not isinstance(node, Project):
            return False
        
        child = node.child
        if not isinstance(child, (Filter, SemanticFilter, SemanticMap)):
            return False
        
        return True
    
    def apply(self, node: PlanNode) -> PlanNode:
        # For now, just return the node unchanged
        # Full implementation would check column dependencies
        return node
