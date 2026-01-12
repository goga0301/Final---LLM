"""
Abstract base class for LLM clients.
Provides a unified interface for interacting with different LLM providers.
"""

from abc import ABC, abstractmethod
from typing import Type, TypeVar, Optional, Any
import json
import asyncio
from pydantic import BaseModel

from config.config import ModelConfig, SYSTEM_CONFIG

T = TypeVar('T', bound=BaseModel)


class BaseLLMClient(ABC):
    """Abstract base class for LLM API clients."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the LLM client.
        
        Args:
            config: Model configuration including API key and settings
        """
        self.config = config
        self.name = config.name
        self.model_id = config.model_id
        
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate a text response from the LLM.
        
        Args:
            prompt: The user prompt/question
            system_prompt: Optional system prompt to set context
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            The generated text response
        """
        pass
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> T:
        """
        Generate a structured response matching a Pydantic schema.
        
        Args:
            prompt: The user prompt/question
            schema: Pydantic model class for the expected response
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Instance of the schema class with the response data
        """
        # Build a prompt that requests JSON output
        json_schema = schema.model_json_schema()
        structured_prompt = f"""{prompt}

Please respond with a valid JSON object that matches this schema:
```json
{json.dumps(json_schema, indent=2)}
```

Important: Respond ONLY with the JSON object, no additional text or markdown formatting."""

        enhanced_system = system_prompt or ""
        enhanced_system += "\nYou must respond with valid JSON only. No markdown, no explanations outside the JSON."
        
        response = await self.generate(
            prompt=structured_prompt,
            system_prompt=enhanced_system.strip(),
            temperature=temperature or 0.3,  # Low temperature for reliable JSON output
            max_tokens=max_tokens
        )
        
        # Parse the response
        parsed = self._parse_json_response(response)
        return schema.model_validate(parsed)
    
    def _parse_json_response(self, response: str) -> dict:
        """
        Parse JSON from LLM response, handling common formatting issues.
        
        Args:
            response: Raw response string from LLM
            
        Returns:
            Parsed dictionary
        """
        # Clean up the response
        text = response.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Try to find JSON object in the response
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                try:
                    return json.loads(text[start_idx:end_idx + 1])
                except json.JSONDecodeError:
                    pass
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response[:500]}")
    
    async def generate_with_retry(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None
    ) -> str:
        """
        Generate with automatic retry on failure.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            max_retries: Override default max retries
            
        Returns:
            Generated text response
        """
        retries = max_retries or SYSTEM_CONFIG.max_retries
        last_error = None
        
        for attempt in range(retries):
            try:
                return await self.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    delay = SYSTEM_CONFIG.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
        
        raise RuntimeError(f"Failed after {retries} attempts: {last_error}")
    
    async def generate_structured_with_retry(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None
    ) -> T:
        """
        Generate structured output with automatic retry.
        
        Args:
            prompt: The user prompt
            schema: Pydantic model for response
            system_prompt: Optional system prompt
            temperature: Override temperature
            max_tokens: Override max tokens
            max_retries: Override max retries
            
        Returns:
            Instance of schema class
        """
        retries = max_retries or SYSTEM_CONFIG.max_retries
        last_error = None
        
        for attempt in range(retries):
            try:
                return await self.generate_structured(
                    prompt=prompt,
                    schema=schema,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            except Exception as e:
                last_error = e
                if attempt < retries - 1:
                    delay = SYSTEM_CONFIG.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
        
        raise RuntimeError(f"Failed after {retries} attempts: {last_error}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, model={self.model_id})"
