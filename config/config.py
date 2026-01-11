"""
Configuration module for the Multi-LLM Debate System.
Handles API keys, model settings, and system configuration.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model."""
    name: str
    api_key: Optional[str]
    model_id: str
    max_tokens: int = 4096
    temperature: float = 1


@dataclass
class SystemConfig:
    """System-wide configuration settings."""
    # Timeout for API calls in seconds
    api_timeout: int = 120
    
    # Maximum retries for failed API calls
    max_retries: int = 3
    
    # Delay between retries in seconds
    retry_delay: float = 1.0
    
    # Whether to run in debug mode
    debug: bool = False
    
    # Results output directory
    results_dir: str = "results"


# Model configurations
OPENAI_CONFIG = ModelConfig(
    name="GPT",
    api_key=os.getenv("OPENAI_API_KEY"),
    model_id="gpt-5.1-2025-11-13",
    max_tokens=4096,
    temperature=1
)

ANTHROPIC_CONFIG = ModelConfig(
    name="Claude",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    model_id="claude-3-haiku-20240307",
    max_tokens=4096,
    temperature=1
)

GOOGLE_CONFIG = ModelConfig(
    name="Gemini",
    api_key=os.getenv("GOOGLE_API_KEY"),
    model_id="gemini-3-flash-preview",
    max_tokens=4096,
    temperature=1
)

XAI_CONFIG = ModelConfig(
    name="Grok",
    api_key=os.getenv("XAI_API_KEY"),
    model_id="grok-3-mini",
    max_tokens=4096,
    temperature=1
)

# System configuration
SYSTEM_CONFIG = SystemConfig(
    api_timeout=120,
    max_retries=3,
    retry_delay=1.0,
    debug=os.getenv("DEBUG", "false").lower() == "true",
    results_dir="results"
)

# All available models
ALL_MODELS = {
    "gpt4": OPENAI_CONFIG,
    "claude": ANTHROPIC_CONFIG,
    "gemini": GOOGLE_CONFIG,
    "grok": XAI_CONFIG
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a specific model."""
    if model_name.lower() not in ALL_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(ALL_MODELS.keys())}")
    return ALL_MODELS[model_name.lower()]


def validate_api_keys() -> dict[str, bool]:
    """Check which API keys are configured."""
    return {
        name: config.api_key is not None and len(config.api_key) > 0
        for name, config in ALL_MODELS.items()
    }
