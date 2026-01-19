"""
LLM-based query optimizer.

Uses an LLM to analyze query plans and suggest optimizations.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

from omnidf.plan import PlanNode, plan_to_json, plan_from_dict
from omnidf.optimizer.base import Optimizer

if TYPE_CHECKING:
    from omnidf.models import LLMClient


class LLMOptimizer:
    """
    LLM-based query optimizer.
    
    Uses an LLM to suggest and apply query optimizations.
    The LLM receives the plan in JSON format and returns optimization suggestions.
    
    The optimizer model is separate from the oracle/proxy models used for
    semantic operations - it's specifically for query plan optimization.
    """
    
    def __init__(
        self,
        llm_client: Optional["LLMClient"] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize LLM optimizer.
        
        Args:
            llm_client: LLM client to use. If None, uses the default client.
            model: Override model for optimization. If None, uses client's optimizer_model.
        """
        self._llm_client = llm_client
        self._model = model
    
    @property
    def llm_client(self) -> "LLMClient":
        """Get the LLM client."""
        if self._llm_client is None:
            from omnidf.settings import get_client
            return get_client()
        return self._llm_client
    
    def get_plan_prompt(self, plan: PlanNode) -> str:
        """
        Generate a prompt for the LLM to optimize the plan.
        """
        plan_json = plan_to_json(plan, indent=2)
        
        prompt = f"""You are a query optimizer for a multi-modal DataFrame system called OmniDF.

The following query plan needs optimization. The plan is represented as a DAG where:
- Relational operators (filter, join, project, etc.) are executed by Pandas (fast, cheap)
- Semantic operators (semantic_filter, semantic_map, semantic_join) use LLM calls (slow, expensive)

Your goal is to minimize:
1. Number of LLM calls (most expensive)
2. Data processed by LLM calls (push filters down to reduce rows)
3. Overall query execution time

Current plan:
```json
{plan_json}
```

Available optimizations:
1. **FilterPushDown**: Push relational filters before semantic operators to reduce rows processed by LLM
2. **SemanticMapFusion**: Combine consecutive semantic_map operations into single LLM call
3. **SemanticFilterFusion**: Combine consecutive semantic_filter operations into single LLM call
4. **SemanticJoinDecomposition**: Convert semantic_join (O(n*m) LLM calls) to semantic_map + relational join (O(n+m) calls)

Analyze the plan and suggest which optimizations to apply. Return your response as valid JSON only:
{{
    "analysis": "Brief analysis of the current plan and optimization opportunities",
    "optimizations": [
        {{
            "rule": "RuleName",
            "target_node_id": "node_id",
            "reason": "Why this optimization helps"
        }}
    ]
}}

IMPORTANT: Return ONLY valid JSON, no markdown code blocks or extra text."""
        return prompt
    
    def optimize(self, plan: PlanNode) -> Tuple[PlanNode, Dict[str, Any]]:
        """
        Optimize the plan using LLM suggestions.
        
        The LLM analyzes the plan and suggests which optimization rules to apply.
        Then the rule-based optimizer applies those rules.
        
        Returns:
            Tuple of (optimized_plan, optimization_info)
        """
        # Generate prompt
        prompt = self.get_plan_prompt(plan)
        
        # Call LLM for optimization suggestions
        response = self.llm_client.complete_for_optimization(prompt)
        
        # Parse LLM response
        try:
            # Try to extract JSON from response
            content = response.content.strip()
            
            # Handle markdown code blocks
            if content.startswith("```"):
                # Extract content between code blocks
                lines = content.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                content = "\n".join(json_lines)
            
            llm_response = json.loads(content)
        except json.JSONDecodeError:
            # If parsing fails, fall back to rule-based optimization
            rule_optimizer = Optimizer()
            optimized = rule_optimizer.optimize(plan)
            return optimized, {
                "method": "rule_based_fallback",
                "reason": "Failed to parse LLM response",
                "raw_response": response.content,
            }
        
        # Apply suggested optimizations
        optimized_plan = self.apply_llm_suggested_plan(plan, llm_response)
        
        info = {
            "method": "llm_guided",
            "analysis": llm_response.get("analysis", ""),
            "optimizations": llm_response.get("optimizations", []),
            "model": response.model,
        }
        
        return optimized_plan, info
    
    def apply_llm_suggested_plan(
        self,
        original_plan: PlanNode,
        llm_response: Dict[str, Any],
    ) -> PlanNode:
        """
        Apply LLM-suggested optimizations.
        
        Args:
            original_plan: The original query plan
            llm_response: The LLM's optimization response
        
        Returns:
            The optimized plan
        """
        if "optimized_plan" in llm_response:
            # LLM provided a full plan - reconstruct it
            try:
                return plan_from_dict(llm_response["optimized_plan"])
            except Exception:
                pass  # Fall through to rule-based
        
        if "optimizations" in llm_response:
            # LLM provided optimization suggestions - apply them using rule-based optimizer
            optimizer = Optimizer()
            plan = original_plan
            
            for opt in llm_response["optimizations"]:
                rule_name = opt.get("rule")
                if rule_name:
                    plan = optimizer.optimize_with_rules(plan, [rule_name])
            
            return plan
        
        return original_plan
    
    def get_optimization_analysis(self, plan: PlanNode) -> Dict[str, Any]:
        """
        Get LLM analysis of the plan without applying optimizations.
        
        Useful for understanding what optimizations are possible.
        """
        prompt = self.get_plan_prompt(plan)
        response = self.llm_client.complete_for_optimization(prompt)
        
        try:
            content = response.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                json_lines = []
                in_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_block = not in_block
                        continue
                    if in_block:
                        json_lines.append(line)
                content = "\n".join(json_lines)
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "analysis": "Failed to parse LLM response",
                "raw_response": response.content,
            }
