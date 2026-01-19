"""
Mock LLM client for testing.
"""

from __future__ import annotations

import json
from typing import List, Optional

from omnidf.models.base import LLMClient, LLMResponse
from omnidf.models.config import LLMConfig


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing without actual API calls.
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.call_count = 0
        self.prompts_received: List[str] = []
    
    def complete(self, prompt: str, model: Optional[str] = None) -> LLMResponse:
        """Mock single completion."""
        self.call_count += 1
        self.prompts_received.append(prompt)
        
        # Generate mock response based on prompt content
        content = self._generate_mock_response(prompt)
        
        return LLMResponse(
            content=content,
            model=model or self.config.oracle_model,
            usage={"prompt_tokens": len(prompt) // 4, "completion_tokens": len(content) // 4},
        )
    
    def batch_complete(
        self,
        prompts: List[str],
        model: Optional[str] = None,
    ) -> List[LLMResponse]:
        """Mock batch completion."""
        return [self.complete(p, model) for p in prompts]
    
    def complete_for_optimization(self, prompt: str) -> LLMResponse:
        """Mock optimization completion."""
        self.call_count += 1
        self.prompts_received.append(prompt)
        
        # Parse the plan from prompt and generate optimization response
        content = self._generate_optimization_response(prompt)
        
        return LLMResponse(
            content=content,
            model=self.config.optimizer_model,
        )
    
    def _generate_mock_response(self, prompt: str) -> str:
        """Generate mock response based on prompt content."""
        prompt_lower = prompt.lower()
        
        # Semantic filter responses
        if "yes" in prompt_lower and "no" in prompt_lower:
            # Binary classification prompt
            if any(kw in prompt_lower for kw in ["solar", "eco", "green", "renewable"]):
                return "yes"
            if any(kw in prompt_lower for kw in ["nolan", "christopher"]):
                return "yes"
            if any(kw in prompt_lower for kw in ["action", "sci-fi"]):
                return "yes"
            return "yes"  # Default to yes for testing
        
        # Semantic map responses
        if "sentiment" in prompt_lower:
            if any(kw in prompt_lower for kw in ["fantastic", "amazing", "best", "brilliant"]):
                return "positive"
            if any(kw in prompt_lower for kw in ["terrible", "boring", "disappointed", "waste"]):
                return "negative"
            return "neutral"
        
        if "extract" in prompt_lower or "transform" in prompt_lower:
            return "extracted_value"
        
        return "mock_response"
    
    def _generate_optimization_response(self, prompt: str) -> str:
        """Generate mock optimization response."""
        # Check what optimizations are applicable based on prompt
        optimizations = []
        
        if "semantic_filter" in prompt.lower() and "filter" in prompt.lower():
            # Check for filter pushdown opportunity
            if '"type": "filter"' in prompt and '"type": "semantic_filter"' in prompt:
                optimizations.append({
                    "rule": "FilterPushDown",
                    "reason": "Push relational filter before semantic filter to reduce LLM calls"
                })
        
        if prompt.lower().count('"type": "semantic_filter"') >= 2:
            optimizations.append({
                "rule": "SemanticFilterFusion",
                "reason": "Fuse consecutive semantic filters into single LLM call"
            })
        
        if '"type": "semantic_join"' in prompt.lower():
            if "sentiment" in prompt.lower() or "same" in prompt.lower():
                optimizations.append({
                    "rule": "SemanticJoinDecomposition",
                    "reason": "Decompose semantic join into semantic map + relational join"
                })
        
        response = {
            "analysis": "Analyzed query plan for optimization opportunities",
            "optimizations": optimizations,
        }
        
        return json.dumps(response, indent=2)
    
    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.prompts_received.clear()
