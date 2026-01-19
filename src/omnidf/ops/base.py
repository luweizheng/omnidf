"""
Base class for Semantic Operators.

Provides common functionality for all semantic operator implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from omnidf.models import LLMClient


class SemanticOperatorBase(ABC):
    """
    Base class for semantic operator implementations.
    
    Semantic operators can be implemented using:
    - LLM calls (default)
    - Embedding-based methods
    - Other ML models
    
    This base class provides common functionality and defines the interface.
    """
    
    def __init__(
        self,
        llm_client: Optional["LLMClient"] = None,
        use_mock: bool = False,
    ):
        """
        Initialize semantic operator.
        
        Args:
            llm_client: LLM client for API calls. If None, uses default client.
            use_mock: If True, use mock implementation instead of real LLM calls.
        """
        self._llm_client = llm_client
        self.use_mock = use_mock
    
    @property
    def llm_client(self) -> "LLMClient":
        """Get the LLM client, using default if not set."""
        if self._llm_client is None:
            from omnidf.settings import get_client
            return get_client()
        return self._llm_client
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> pd.DataFrame:
        """Execute the semantic operation."""
        pass
    
    @abstractmethod
    def execute_mock(self, *args, **kwargs) -> pd.DataFrame:
        """Execute mock implementation for testing."""
        pass
