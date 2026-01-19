"""
Base classes for query optimization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from omnidf.plan import PlanNode


class Rule(ABC):
    """
    Base class for optimization rules.
    
    Each rule defines a pattern to match and a transformation to apply.
    Rules are applied bottom-up (children first) by default.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the rule for logging/debugging."""
        pass
    
    @abstractmethod
    def matches(self, node: PlanNode) -> bool:
        """Check if this rule can be applied to the given node."""
        pass
    
    @abstractmethod
    def apply(self, node: PlanNode) -> PlanNode:
        """Apply the transformation and return the new node."""
        pass
    
    def __repr__(self) -> str:
        return f"Rule({self.name})"


class Optimizer:
    """
    Query optimizer that applies transformation rules.
    
    Supports both rule-based and LLM-based optimization.
    """
    
    def __init__(self, rules: Optional[List[Rule]] = None):
        """
        Initialize optimizer with rules.
        
        Args:
            rules: List of optimization rules. If None, uses default rules.
        """
        if rules is None:
            self.rules = self._default_rules()
        else:
            self.rules = rules
    
    def _default_rules(self) -> List[Rule]:
        """Get the default set of optimization rules."""
        from omnidf.optimizer.rules import (
            FilterPushDown,
            SemanticMapFusion,
            SemanticFilterFusion,
            SemanticJoinDecomposition,
        )
        return [
            FilterPushDown(),
            SemanticMapFusion(),
            SemanticFilterFusion(),
            SemanticJoinDecomposition(),
        ]
    
    def optimize(self, plan: PlanNode, max_iterations: int = 10) -> PlanNode:
        """
        Optimize the query plan by applying rules until fixpoint.
        
        Args:
            plan: The query plan to optimize
            max_iterations: Maximum number of optimization passes
        
        Returns:
            Optimized query plan
        """
        current = plan
        
        for _ in range(max_iterations):
            new_plan = self._apply_rules_once(current)
            if self._plans_equal(current, new_plan):
                break
            current = new_plan
        
        return current
    
    def _apply_rules_once(self, node: PlanNode) -> PlanNode:
        """
        Apply rules once in bottom-up order.
        """
        # First, recursively optimize children
        if node.children:
            new_children = [self._apply_rules_once(child) for child in node.children]
            node = node.with_children(new_children)
        
        # Then try to apply rules to this node
        for rule in self.rules:
            if rule.matches(node):
                node = rule.apply(node)
                # After applying a rule, we might be able to apply more
                # But we only do one pass here
                break
        
        return node
    
    def _plans_equal(self, plan1: PlanNode, plan2: PlanNode) -> bool:
        """Check if two plans are structurally equal."""
        return plan1.node_id == plan2.node_id
    
    def optimize_with_rules(
        self,
        plan: PlanNode,
        rule_names: List[str],
    ) -> PlanNode:
        """
        Optimize using only specific rules.
        
        Args:
            plan: The query plan to optimize
            rule_names: Names of rules to apply
        """
        selected_rules = [r for r in self.rules if r.name in rule_names]
        temp_optimizer = Optimizer(rules=selected_rules)
        return temp_optimizer.optimize(plan)
    
    def get_optimization_suggestions(self, plan: PlanNode) -> List[Dict[str, Any]]:
        """
        Get suggestions for optimizations without applying them.
        
        Returns:
            List of dicts with rule name, matched node, and description
        """
        suggestions = []
        
        def visit(node: PlanNode):
            for rule in self.rules:
                if rule.matches(node):
                    suggestions.append({
                        "rule": rule.name,
                        "node_id": node.node_id,
                        "node_type": node.node_type.value,
                        "description": f"Can apply {rule.name} to {node._node_str()}",
                    })
            for child in node.children:
                visit(child)
        
        visit(plan)
        return suggestions
