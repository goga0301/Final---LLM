"""
Google Gemini client implementation using the new google-genai SDK.
"""

from typing import Optional
from google import genai
from google.genai import types

from .base_client import BaseLLMClient
from config.config import ModelConfig, GOOGLE_CONFIG


class GoogleClient(BaseLLMClient):
    """Client for Google's Gemini API."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the Google Gemini client.
        
        Args:
            config: Model configuration, defaults to GOOGLE_CONFIG
        """
        config = config or GOOGLE_CONFIG
        super().__init__(config)
        
        if not config.api_key:
            raise ValueError("Google API key not configured")
        
        self.client = genai.Client(api_key=config.api_key)
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a response using Gemini.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            
        Returns:
            Generated text response
        """
        # Build generation config
        generation_config = types.GenerateContentConfig(
            temperature=temperature or self.config.temperature,
            max_output_tokens=max_tokens or self.config.max_tokens,
            system_instruction=system_prompt if system_prompt else None
        )
        
        # Generate response using async method
        response = await self.client.aio.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=generation_config
        )
        
        return response.text
