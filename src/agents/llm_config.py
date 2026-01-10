"""
Centralized LLM configuration and model registry.

Supports multiple providers: Azure OpenAI, OpenAI, Anthropic, Google Vertex AI.
This is the SINGLE SOURCE OF TRUTH for all LLM configuration in the app.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    AZURE_OPENAI = "azure-openai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VERTEX_AI = "vertex-ai"


# Default configuration values
DEFAULT_AZURE_ENDPOINT = "https://kallamai.openai.azure.com/"
DEFAULT_AZURE_API_VERSION = "2025-01-01-preview"
DEFAULT_VERTEX_LOCATION = "us-central1"
DEFAULT_TEMPERATURE = 0.1


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    provider: LLMProvider
    model_id: str
    display_name: str
    is_fast: bool = False  # True for models suitable for orchestrator routing
    supports_tools: bool = True


# Model registry with all supported models - SINGLE SOURCE OF TRUTH
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # Azure OpenAI
    "gpt-4.1": ModelConfig(
        LLMProvider.AZURE_OPENAI, "gpt-4.1", "Azure GPT-4.1"
    ),
    "gpt-4.1-mini": ModelConfig(
        LLMProvider.AZURE_OPENAI, "gpt-4.1-mini", "Azure GPT-4.1 Mini", is_fast=True
    ),
    "gpt-5.1": ModelConfig(
        LLMProvider.AZURE_OPENAI, "gpt-5.1", "Azure GPT-5.1"
    ),
    # Anthropic Claude (via Vertex AI)
    "claude-sonnet-4.5": ModelConfig(
        LLMProvider.VERTEX_AI, "claude-sonnet-4-5", "Claude Sonnet 4.5"
    ),
    "claude-opus-4.5": ModelConfig(
        LLMProvider.VERTEX_AI, "claude-opus-4-5", "Claude Opus 4.5"
    ),
    "claude-haiku-4.5": ModelConfig(
        LLMProvider.VERTEX_AI, "claude-haiku-4-5", "Claude Haiku 4.5", is_fast=True
    ),
    # Google Vertex AI
    "gemini-2.5-flash": ModelConfig(
        LLMProvider.VERTEX_AI, "gemini-2.5-flash", "Gemini 2.5 Flash", is_fast=True
    ),
    "gemini-3-pro": ModelConfig(
        LLMProvider.VERTEX_AI, "gemini-3-pro-preview", "Gemini 3 Pro"
    ),
}

# Fast models for orchestrator routing (subset of MODEL_REGISTRY)
FAST_MODELS = {k: v for k, v in MODEL_REGISTRY.items() if v.is_fast}


def create_llm(
    model_key: str,
    temperature: float = DEFAULT_TEMPERATURE,
    azure_endpoint: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    azure_api_version: str = DEFAULT_AZURE_API_VERSION,
    vertex_project: Optional[str] = None,
    vertex_location: Optional[str] = None,
) -> BaseChatModel:
    """Create an LLM instance from a model key.

    Args:
        model_key: Key from MODEL_REGISTRY (e.g., "claude-sonnet-4.5")
        temperature: Temperature for generation
        azure_endpoint: Optional Azure endpoint override
        azure_api_key: Optional Azure API key override
        azure_api_version: Azure API version
        vertex_project: Optional Vertex AI project override
        vertex_location: Optional Vertex AI location override

    Returns:
        Configured LLM instance

    Raises:
        ValueError: If model_key is not found in registry
    """
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")

    config = MODEL_REGISTRY[model_key]
    return create_llm_from_config(
        config,
        temperature=temperature,
        azure_endpoint=azure_endpoint,
        azure_api_key=azure_api_key,
        azure_api_version=azure_api_version,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
    )


def create_llm_from_config(
    config: ModelConfig,
    temperature: float = DEFAULT_TEMPERATURE,
    azure_endpoint: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    azure_api_version: str = DEFAULT_AZURE_API_VERSION,
    vertex_project: Optional[str] = None,
    vertex_location: Optional[str] = None,
) -> BaseChatModel:
    """Create an LLM instance from a ModelConfig.

    Args:
        config: ModelConfig instance
        temperature: Temperature for generation
        azure_endpoint: Optional Azure endpoint override
        azure_api_key: Optional Azure API key override
        azure_api_version: Azure API version
        vertex_project: Optional Vertex AI project override
        vertex_location: Optional Vertex AI location override

    Returns:
        Configured LLM instance
    """
    if config.provider == LLMProvider.ANTHROPIC:
        return ChatAnthropic(
            model=config.model_id,
            temperature=temperature,
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        )

    elif config.provider == LLMProvider.AZURE_OPENAI:
        return AzureChatOpenAI(
            azure_endpoint=azure_endpoint
            or os.getenv("AZURE_OPENAI_ENDPOINT", DEFAULT_AZURE_ENDPOINT),
            api_version=azure_api_version,
            api_key=azure_api_key or os.getenv("AZURE_OPENAI_API_KEY", ""),
            model=config.model_id,
            temperature=temperature,
        )

    elif config.provider == LLMProvider.VERTEX_AI:
        return ChatVertexAI(
            model_name=config.model_id,
            project=vertex_project or os.getenv("GOOGLE_VERTEX_PROJECT", ""),
            location=vertex_location or os.getenv("GOOGLE_VERTEX_LOCATION", DEFAULT_VERTEX_LOCATION),
            temperature=temperature,
        )

    elif config.provider == LLMProvider.OPENAI:
        return ChatOpenAI(
            model=config.model_id,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )

    else:
        raise ValueError(f"Unknown provider: {config.provider}")


def get_fast_model_for_provider(provider: LLMProvider) -> Optional[ModelConfig]:
    """Get the fastest model for a given provider.

    Args:
        provider: The LLM provider

    Returns:
        ModelConfig for the fastest model of that provider, or None
    """
    for config in FAST_MODELS.values():
        if config.provider == provider:
            return config
    return None


def get_default_fast_model() -> ModelConfig:
    """Get the default fast model for orchestrator routing."""
    return MODEL_REGISTRY["claude-haiku-4.5"]


def get_available_models() -> List[Tuple[str, str, ModelConfig]]:
    """Get list of available models for UI selection.

    Returns:
        List of (model_key, display_name, config) tuples
    """
    return [(key, config.display_name, config) for key, config in MODEL_REGISTRY.items()]


def get_model_by_key(model_key: str) -> Optional[ModelConfig]:
    """Get model config by key.

    Args:
        model_key: Key from MODEL_REGISTRY

    Returns:
        ModelConfig or None if not found
    """
    return MODEL_REGISTRY.get(model_key)


def get_default_provider() -> str:
    """Determine the best available provider based on environment variables.

    Returns:
        Provider string (e.g., "anthropic", "azure-openai", "vertex-ai")
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    elif os.getenv("AZURE_OPENAI_API_KEY"):
        return "azure-openai"
    elif os.getenv("GOOGLE_VERTEX_PROJECT"):
        return "vertex-ai"
    elif os.getenv("OPENAI_API_KEY"):
        return "openai"
    else:
        return "vertex-ai"  # Default to Vertex AI


def get_default_model_key() -> str:
    """Get the default model key based on available providers.

    Returns:
        Model key string
    """
    provider = get_default_provider()
    if provider == "azure-openai":
        return "gpt-4.1-mini"
    elif provider == "anthropic":
        return "claude-sonnet-4.5"
    elif provider == "vertex-ai":
        return "claude-haiku-4.5"
    else:
        return "claude-haiku-4.5"
