"""
LiteLLM-based LLM client implementation.
"""

from __future__ import annotations

import asyncio
import os
from typing import Dict, List, Optional

from omnidf.models.base import LLMClient, LLMResponse
from omnidf.models.config import LLMConfig

try:
    import litellm
    from litellm import completion, batch_completion, acompletion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


class LiteLLMClient(LLMClient):
    """
    LLM client using LiteLLM for unified API access.
    
    Supports:
    - OpenAI, Anthropic, Azure, and other providers via LiteLLM
    - Batch requests for efficiency
    - Oracle/Proxy model fallback
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is required for LLM operations. "
                "Install with: pip install litellm"
            )
        
        self.config = config or LLMConfig()
        
        # Set default API key from oracle config (for compatibility)
        if self.config.oracle_api_key:
            os.environ["OPENAI_API_KEY"] = self.config.oracle_api_key
        
        # Configure LiteLLM
        litellm.set_verbose = False
    
    def _get_api_config(self, use_proxy: bool = False, use_optimizer: bool = False) -> tuple:
        """Get API key and base for the specified model type."""
        if use_optimizer:
            return self.config.optimizer_api_key, self.config.optimizer_api_base
        elif use_proxy and self.config.proxy_model:
            return self.config.proxy_api_key, self.config.proxy_api_base
        else:
            return self.config.oracle_api_key, self.config.oracle_api_base
    
    def _get_model(self, model: Optional[str] = None) -> str:
        """Get the model to use, with fallback logic."""
        if model:
            return model
        return self.config.oracle_model
    
    def _create_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Create messages for chat completion."""
        return [{"role": "user", "content": prompt}]
    
    def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        use_proxy: bool = False,
    ) -> LLMResponse:
        """
        Single completion request.
        
        Args:
            prompt: The prompt to complete
            model: Override model (optional)
            use_proxy: If True and proxy_model is set, use proxy model
        """
        selected_model = model
        if selected_model is None:
            if use_proxy and self.config.proxy_model:
                selected_model = self.config.proxy_model
            else:
                selected_model = self.config.oracle_model
        
        # Get API config for the selected model type
        api_key, api_base = self._get_api_config(use_proxy=use_proxy)
        
        try:
            kwargs = {
                "model": selected_model,
                "messages": self._create_messages(prompt),
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            # Add api_base and api_key if configured
            if api_base:
                kwargs["api_base"] = api_base
            if api_key:
                kwargs["api_key"] = api_key
            
            response = completion(**kwargs)
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=selected_model,
                usage=dict(response.usage) if response.usage else None,
                raw_response=response,
            )
        except Exception as e:
            # Fallback to proxy model if available
            if not use_proxy and self.config.proxy_model:
                return self.complete(prompt, model=self.config.proxy_model, use_proxy=True)
            raise e
    
    def batch_complete(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        use_proxy: bool = False,
    ) -> List[LLMResponse]:
        """
        Batch completion requests using LiteLLM's batch_completion.
        
        Args:
            prompts: List of prompts to complete
            model: Override model (optional)
            use_proxy: If True and proxy_model is set, use proxy model
        """
        if not prompts:
            return []
        
        selected_model = model
        if selected_model is None:
            if use_proxy and self.config.proxy_model:
                selected_model = self.config.proxy_model
            else:
                selected_model = self.config.oracle_model
        
        # Create messages for each prompt
        messages_list = [self._create_messages(p) for p in prompts]
        
        # Get API config for the selected model type
        api_key, api_base = self._get_api_config(use_proxy=use_proxy)
        
        try:
            # Use LiteLLM batch completion
            kwargs = {
                "model": selected_model,
                "messages": messages_list,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            # Add api_base and api_key if configured
            if api_base:
                kwargs["api_base"] = api_base
            if api_key:
                kwargs["api_key"] = api_key
            
            responses = batch_completion(**kwargs)
            
            results = []
            for resp in responses:
                if hasattr(resp, 'choices') and resp.choices:
                    content = resp.choices[0].message.content or ""
                    results.append(LLMResponse(
                        content=content,
                        model=selected_model,
                        usage=dict(resp.usage) if resp.usage else None,
                        raw_response=resp,
                    ))
                else:
                    # Handle error response
                    results.append(LLMResponse(
                        content="",
                        model=selected_model,
                        raw_response=resp,
                    ))
            
            # If batch_completion returned empty results, fallback to sequential
            if all(r.content == "" for r in results):
                return self._sequential_complete(prompts, selected_model)
            
            return results
            
        except Exception as e:
            # Fallback to sequential completion
            try:
                return self._sequential_complete(prompts, selected_model)
            except Exception:
                # Fallback to proxy model if available
                if not use_proxy and self.config.proxy_model:
                    return self.batch_complete(prompts, model=self.config.proxy_model, use_proxy=True)
                raise e
    
    def _sequential_complete(
        self,
        prompts: List[str],
        model: str,
    ) -> List[LLMResponse]:
        """
        Sequential completion as fallback when batch completion fails.
        """
        results = []
        for prompt in prompts:
            resp = self.complete(prompt, model=model)
            results.append(resp)
        return results
    
    async def abatch_complete(
        self,
        prompts: List[str],
        model: Optional[str] = None,
    ) -> List[LLMResponse]:
        """
        Async batch completion for better concurrency.
        """
        selected_model = model or self.config.oracle_model
        
        async def single_complete(prompt: str) -> LLMResponse:
            response = await acompletion(
                model=selected_model,
                messages=self._create_messages(prompt),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return LLMResponse(
                content=response.choices[0].message.content,
                model=selected_model,
                usage=dict(response.usage) if response.usage else None,
                raw_response=response,
            )
        
        # Run all completions concurrently
        tasks = [single_complete(p) for p in prompts]
        return await asyncio.gather(*tasks)
    
    def complete_for_optimization(self, prompt: str) -> LLMResponse:
        """
        Completion for query optimization using the optimizer model.
        Uses optimizer-specific API configuration.
        """
        api_key, api_base = self._get_api_config(use_optimizer=True)
        
        kwargs = {
            "model": self.config.optimizer_model,
            "messages": self._create_messages(prompt),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if api_base:
            kwargs["api_base"] = api_base
        if api_key:
            kwargs["api_key"] = api_key
        
        response = completion(**kwargs)
        
        return LLMResponse(
            content=response.choices[0].message.content,
            model=self.config.optimizer_model,
            usage=dict(response.usage) if response.usage else None,
            raw_response=response,
        )
