"""
OmniDF Settings Module.

Provides a unified configuration API for OmniDF, including LLM model settings.

Supports separate API configurations for oracle and proxy models:
- oracle: Primary model (can use cloud API)
- proxy: Fallback model (can use internal enterprise API)
- optimizer: Query optimization model

Usage:
    import omnidf as odf
    
    # Simple configuration (same API for all models)
    odf.settings.configure(
        oracle_model="gpt-4o-mini",
        oracle_api_key="your-cloud-key",
        oracle_api_base="https://api.openai.com/v1",
    )
    
    # Advanced configuration (different APIs for oracle and proxy)
    odf.settings.configure(
        oracle_model="gpt-4o",
        oracle_api_key="cloud-api-key",
        oracle_api_base="https://api.openai.com/v1",
        proxy_model="gpt-3.5-turbo",
        proxy_api_key="enterprise-api-key",
        proxy_api_base="https://internal.company.com/v1",
    )
"""

from __future__ import annotations

from typing import Optional

from omnidf.models.config import LLMConfig
from omnidf.models.base import LLMClient
from omnidf.models.litellm_client import LiteLLMClient
from omnidf.models.mock_client import MockLLMClient


# Global client instance
_default_client: Optional[LLMClient] = None
_current_config: Optional[LLMConfig] = None


def get_client() -> LLMClient:
    """Get the current LLM client."""
    global _default_client
    if _default_client is None:
        # Default to mock client for safety
        _default_client = MockLLMClient()
    return _default_client


def set_client(client: LLMClient):
    """Set the LLM client."""
    global _default_client
    _default_client = client


def get_config() -> Optional[LLMConfig]:
    """Get the current LLM configuration."""
    return _current_config


def configure(
    # Oracle model configuration
    oracle_model: str = "gpt-4.1",
    oracle_api_key: Optional[str] = None,
    oracle_api_base: Optional[str] = None,
    # Proxy model configuration (optional, can use different API)
    proxy_model: Optional[str] = None,
    proxy_api_key: Optional[str] = None,
    proxy_api_base: Optional[str] = None,
    # Optimizer model configuration
    optimizer_model: str = "gpt-4.1",
    optimizer_api_key: Optional[str] = None,
    optimizer_api_base: Optional[str] = None,
    # General settings
    use_mock: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_batch_size: int = 20,
    max_concurrent_requests: int = 10,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    # Backward compatibility
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> LLMClient:
    """
    Configure OmniDF settings, including LLM models.
    
    Supports separate API configurations for oracle, proxy, and optimizer models.
    This allows using cloud API for oracle and internal enterprise API for proxy.
    
    Args:
        oracle_model: Primary model for semantic operations (default: "gpt-4.1")
        oracle_api_key: API key for oracle model (defaults to OPENAI_API_KEY env var)
        oracle_api_base: API base URL for oracle model (defaults to OPENAI_API_BASE env var)
        proxy_model: Optional fallback model (e.g., "gpt-3.5-turbo")
        proxy_api_key: API key for proxy model (defaults to oracle_api_key)
        proxy_api_base: API base URL for proxy model (defaults to oracle_api_base)
        optimizer_model: Model for query optimization (default: "gpt-4.1")
        optimizer_api_key: API key for optimizer model (defaults to oracle_api_key)
        optimizer_api_base: API base URL for optimizer model (defaults to oracle_api_base)
        use_mock: If True, use mock client for testing
        temperature: Generation temperature (default: 0.0)
        max_tokens: Maximum tokens per response (default: 1024)
        max_batch_size: Maximum batch size for batch requests (default: 20)
        max_concurrent_requests: Maximum concurrent requests (default: 10)
        max_retries: Maximum retries on failure (default: 3)
        retry_delay: Delay between retries in seconds (default: 1.0)
        api_key: Deprecated, use oracle_api_key instead
        api_base: Deprecated, use oracle_api_base instead
    
    Returns:
        The configured LLM client
    
    Example:
        >>> import omnidf as odf
        >>> # Simple: same API for all models
        >>> odf.settings.configure(
        ...     oracle_model="gpt-4o-mini",
        ...     oracle_api_key="your-key",
        ... )
        >>> # Advanced: different APIs for oracle and proxy
        >>> odf.settings.configure(
        ...     oracle_model="gpt-4o",
        ...     oracle_api_key="cloud-key",
        ...     oracle_api_base="https://api.openai.com/v1",
        ...     proxy_model="internal-llm",
        ...     proxy_api_key="enterprise-key",
        ...     proxy_api_base="https://internal.company.com/v1",
        ... )
    """
    global _default_client, _current_config
    
    # Backward compatibility: use api_key/api_base as oracle defaults
    if api_key and not oracle_api_key:
        oracle_api_key = api_key
    if api_base and not oracle_api_base:
        oracle_api_base = api_base
    
    config = LLMConfig(
        oracle_model=oracle_model,
        oracle_api_key=oracle_api_key,
        oracle_api_base=oracle_api_base,
        proxy_model=proxy_model,
        proxy_api_key=proxy_api_key,
        proxy_api_base=proxy_api_base,
        optimizer_model=optimizer_model,
        optimizer_api_key=optimizer_api_key,
        optimizer_api_base=optimizer_api_base,
        temperature=temperature,
        max_tokens=max_tokens,
        max_batch_size=max_batch_size,
        max_concurrent_requests=max_concurrent_requests,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    
    _current_config = config
    
    if use_mock:
        client = MockLLMClient(config)
    else:
        client = LiteLLMClient(config)
    
    _default_client = client
    return client


def reset():
    """Reset settings to defaults."""
    global _default_client, _current_config
    _default_client = None
    _current_config = None
