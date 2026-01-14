"""
xAI Grok client implementation.
Uses the OpenAI-compatible API endpoint.
"""

from typing import Optional
from openai import AsyncOpenAI

from .base_client import BaseLLMClient
from config.config import ModelConfig, XAI_CONFIG


class XAIClient(BaseLLMClient):
    """Client for xAI's Grok API (OpenAI-compatible)."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the xAI Grok client.
        
        Args:
            config: Model configuration, defaults to XAI_CONFIG
        """
        config = config or XAI_CONFIG
        super().__init__(config)
        
        if not config.api_key:
            raise ValueError("xAI API key not configured")
        
        # xAI uses an OpenAI-compatible API
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url="https://api.x.ai/v1"
        )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response using Grok.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            
        Returns:
            Generated text response
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens
        )
        
        return response.choices[0].message.content
