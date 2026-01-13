"""
Anthropic Claude client implementation.
"""

from typing import Optional
from anthropic import AsyncAnthropic

from .base_client import BaseLLMClient
from config.config import ModelConfig, ANTHROPIC_CONFIG


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic's Claude API."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the Anthropic client.
        
        Args:
            config: Model configuration, defaults to ANTHROPIC_CONFIG
        """
        config = config or ANTHROPIC_CONFIG
        super().__init__(config)
        
        if not config.api_key:
            raise ValueError("Anthropic API key not configured")
        
        self.client = AsyncAnthropic(api_key=config.api_key)
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response using Claude.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            
        Returns:
            Generated text response
        """
        kwargs = {
            "model": self.model_id,
            "max_tokens": max_tokens or self.config.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        if temperature is not None:
            kwargs["temperature"] = temperature
        else:
            kwargs["temperature"] = self.config.temperature
        
        response = await self.client.messages.create(**kwargs)
        
        return response.content[0].text
