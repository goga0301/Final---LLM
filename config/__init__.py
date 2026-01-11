"""Configuration package for the Multi-LLM Debate System."""

from .config import (
    ModelConfig,
    SystemConfig,
    OPENAI_CONFIG,
    ANTHROPIC_CONFIG,
    GOOGLE_CONFIG,
    XAI_CONFIG,
    SYSTEM_CONFIG,
    ALL_MODELS,
    get_model_config,
    validate_api_keys
)

__all__ = [
    "ModelConfig",
    "SystemConfig",
    "OPENAI_CONFIG",
    "ANTHROPIC_CONFIG",
    "GOOGLE_CONFIG",
    "XAI_CONFIG",
    "SYSTEM_CONFIG",
    "ALL_MODELS",
    "get_model_config",
    "validate_api_keys"
]
