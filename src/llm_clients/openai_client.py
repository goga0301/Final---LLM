"""
OpenAI GPT client implementation.
Supports newer models that require max_completion_tokens parameter.
"""

from typing import Optional
from openai import AsyncOpenAI

from .base_client import BaseLLMClient
from config.config import ModelConfig, OPENAI_CONFIG


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI's GPT API."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the OpenAI client.
        
        Args:
            config: Model configuration, defaults to OPENAI_CONFIG
        """
        config = config or OPENAI_CONFIG
        super().__init__(config)
        
        if not config.api_key:
            raise ValueError("OpenAI API key not configured")
        
        self.client = AsyncOpenAI(api_key=config.api_key)
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response using OpenAI GPT.
        
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
        
        # Use max_completion_tokens for newer models (required for GPTo, gpt-5, etc.)
        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_completion_tokens=max_tokens or self.config.max_tokens
        )
        
        return response.choices[0].message.content
