"""
LLM Configuration for OmniDF.

Supports separate API configurations for oracle and proxy models,
allowing oracle to use cloud API and proxy to use internal enterprise API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for a single model endpoint."""
    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    def __post_init__(self):
        # Try to get from environment if not provided
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if self.api_base is None:
            self.api_base = os.environ.get("OPENAI_API_BASE")


@dataclass
class LLMConfig:
    """
    Configuration for LLM models.
    
    Supports separate API configurations for oracle and proxy models:
    - oracle: Primary model for semantic operations (cloud API)
    - proxy: Fallback model (can use internal enterprise API)
    - optimizer: Model for query optimization
    """
    # Oracle model configuration
    oracle_model: str = "gpt-4.1"
    oracle_api_key: Optional[str] = None
    oracle_api_base: Optional[str] = None
    
    # Proxy model configuration (optional, can use different API endpoint)
    proxy_model: Optional[str] = None
    proxy_api_key: Optional[str] = None
    proxy_api_base: Optional[str] = None
    
    # Optimizer model configuration
    optimizer_model: str = "gpt-4.1"
    optimizer_api_key: Optional[str] = None
    optimizer_api_base: Optional[str] = None
    
    # Batch settings
    max_batch_size: int = 20
    max_concurrent_requests: int = 10
    
    # Generation settings
    temperature: float = 0.0
    max_tokens: int = 1024
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        # Try to get oracle API config from environment if not provided
        if self.oracle_api_key is None:
            self.oracle_api_key = os.environ.get("OPENAI_API_KEY")
        if self.oracle_api_base is None:
            self.oracle_api_base = os.environ.get("OPENAI_API_BASE")
        
        # Proxy defaults to oracle config if not specified
        if self.proxy_model and self.proxy_api_key is None:
            self.proxy_api_key = self.oracle_api_key
        if self.proxy_model and self.proxy_api_base is None:
            self.proxy_api_base = self.oracle_api_base
        
        # Optimizer defaults to oracle config if not specified
        if self.optimizer_api_key is None:
            self.optimizer_api_key = self.oracle_api_key
        if self.optimizer_api_base is None:
            self.optimizer_api_base = self.oracle_api_base
    
    def get_oracle_config(self) -> ModelConfig:
        """Get configuration for oracle model."""
        return ModelConfig(
            model=self.oracle_model,
            api_key=self.oracle_api_key,
            api_base=self.oracle_api_base,
        )
    
    def get_proxy_config(self) -> Optional[ModelConfig]:
        """Get configuration for proxy model."""
        if self.proxy_model is None:
            return None
        return ModelConfig(
            model=self.proxy_model,
            api_key=self.proxy_api_key,
            api_base=self.proxy_api_base,
        )
    
    def get_optimizer_config(self) -> ModelConfig:
        """Get configuration for optimizer model."""
        return ModelConfig(
            model=self.optimizer_model,
            api_key=self.optimizer_api_key,
            api_base=self.optimizer_api_base,
        )
