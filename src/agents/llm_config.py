"""
Centralized LLM configuration and model registry.

Supports multiple providers: Azure OpenAI, OpenAI, Anthropic, Google Vertex AI.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

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


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    provider: LLMProvider
    model_id: str
    display_name: str
    is_fast: bool = False  # True for models suitable for orchestrator routing
    supports_tools: bool = True


# Model registry with all supported models
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # Azure OpenAI
    "gpt-4.1": ModelConfig(
        LLMProvider.AZURE_OPENAI, "gpt-4.1", "Azure GPT-4.1"
    ),
    "gpt-4.1-mini": ModelConfig(
        LLMProvider.AZURE_OPENAI, "gpt-4.1-mini", "Azure GPT-4.1 Mini", is_fast=True
    ),
    "o4-mini": ModelConfig(
        LLMProvider.AZURE_OPENAI, "o4-mini", "Azure o4-mini"
    ),
    "gpt-5": ModelConfig(
        LLMProvider.AZURE_OPENAI, "gpt-5", "Azure GPT-5"
    ),
    "gpt-5-mini": ModelConfig(
        LLMProvider.AZURE_OPENAI, "gpt-5-mini", "Azure GPT-5 Mini"
    ),
    # Anthropic Claude
    "claude-sonnet-4": ModelConfig(
        LLMProvider.ANTHROPIC, "claude-sonnet-4-20250514", "Claude Sonnet 4"
    ),
    "claude-sonnet-4.5": ModelConfig(
        LLMProvider.ANTHROPIC, "claude-sonnet-4-5-20251101", "Claude Sonnet 4.5"
    ),
    "claude-opus-4.5": ModelConfig(
        LLMProvider.ANTHROPIC, "claude-opus-4-5-20251101", "Claude Opus 4.5"
    ),
    "claude-haiku-3.5": ModelConfig(
        LLMProvider.ANTHROPIC, "claude-3-5-haiku-20241022", "Claude Haiku 3.5", is_fast=True
    ),
    # Google Vertex AI
    "gemini-2.0-flash-lite": ModelConfig(
        LLMProvider.VERTEX_AI, "gemini-2.0-flash-lite-001", "Gemini 2.0 Flash Lite", is_fast=True
    ),
    "gemini-2.5-pro": ModelConfig(
        LLMProvider.VERTEX_AI, "gemini-2.5-pro", "Gemini 2.5 Pro"
    ),
    "gemini-3-pro": ModelConfig(
        LLMProvider.VERTEX_AI, "gemini-3.0-pro", "Gemini 3 Pro"
    ),
}

# Fast models for orchestrator routing (subset of MODEL_REGISTRY)
FAST_MODELS = {k: v for k, v in MODEL_REGISTRY.items() if v.is_fast}


def create_llm(
    model_key: str,
    temperature: float = 0.1,
    azure_endpoint: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    azure_api_version: str = "2025-01-01-preview",
) -> BaseChatModel:
    """Create an LLM instance from a model key.

    Args:
        model_key: Key from MODEL_REGISTRY (e.g., "claude-sonnet-4.5")
        temperature: Temperature for generation
        azure_endpoint: Optional Azure endpoint override
        azure_api_key: Optional Azure API key override
        azure_api_version: Azure API version

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
    )


def create_llm_from_config(
    config: ModelConfig,
    temperature: float = 0.1,
    azure_endpoint: Optional[str] = None,
    azure_api_key: Optional[str] = None,
    azure_api_version: str = "2025-01-01-preview",
) -> BaseChatModel:
    """Create an LLM instance from a ModelConfig.

    Args:
        config: ModelConfig instance
        temperature: Temperature for generation
        azure_endpoint: Optional Azure endpoint override
        azure_api_key: Optional Azure API key override
        azure_api_version: Azure API version

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
            or os.getenv("AZURE_OPENAI_ENDPOINT", "https://kallamai.openai.azure.com/"),
            api_version=azure_api_version,
            api_key=azure_api_key or os.getenv("AZURE_OPENAI_API_KEY", ""),
            model=config.model_id,
            temperature=temperature,
        )

    elif config.provider == LLMProvider.VERTEX_AI:
        return ChatVertexAI(
            model_name=config.model_id,
            project=os.getenv("GOOGLE_VERTEX_PROJECT", ""),
            location=os.getenv("GOOGLE_VERTEX_LOCATION", "us-central1"),
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
    """Get the default fast model for orchestrator routing.

    Returns Claude Haiku 3.5 as the default fast model.
    """
    return MODEL_REGISTRY["claude-haiku-3.5"]
