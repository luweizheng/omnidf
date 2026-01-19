"""
Base classes for LLM clients.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def complete(self, prompt: str, model: Optional[str] = None) -> LLMResponse:
        """Single completion request."""
        pass
    
    @abstractmethod
    def batch_complete(
        self,
        prompts: List[str],
        model: Optional[str] = None,
    ) -> List[LLMResponse]:
        """Batch completion requests."""
        pass
    
    @abstractmethod
    def complete_for_optimization(self, prompt: str) -> LLMResponse:
        """Completion for query optimization (uses optimizer model)."""
        pass
