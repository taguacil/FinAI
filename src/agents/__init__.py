# AI Agents Package
"""
Multi-agent system for portfolio management.

Provides:
- PortfolioAgent: Main facade for backward compatibility
- OrchestratorAgent: Query classification and routing
- TransactionAgent: CRUD operations on transactions
- AnalyticsAgent: Market data and portfolio analysis
"""

from .analytics_agent import AnalyticsAgent
from .base_agent import BaseAgent
from .llm_config import (
    FAST_MODELS,
    MODEL_REGISTRY,
    LLMProvider,
    ModelConfig,
    create_llm,
    create_llm_from_config,
    get_default_fast_model,
    get_fast_model_for_provider,
)
from .orchestrator_agent import OrchestratorAgent, QueryCategory, QueryClassification
from .portfolio_agent import PortfolioAgent
from .shared_state import SharedAgentState
from .transaction_agent import TransactionAgent

__all__ = [
    # Main facade
    "PortfolioAgent",
    # Specialist agents
    "TransactionAgent",
    "AnalyticsAgent",
    "OrchestratorAgent",
    # Base classes
    "BaseAgent",
    "SharedAgentState",
    # LLM configuration
    "LLMProvider",
    "ModelConfig",
    "MODEL_REGISTRY",
    "FAST_MODELS",
    "create_llm",
    "create_llm_from_config",
    "get_default_fast_model",
    "get_fast_model_for_provider",
    # Orchestrator types
    "QueryCategory",
    "QueryClassification",
]
