"""
Portfolio AI agent using a multi-agent architecture for financial advice and portfolio management.

This module provides a backward-compatible facade that internally uses:
- OrchestratorAgent: Routes queries to the appropriate specialist
- TransactionAgent: Handles CRUD operations on transactions
- AnalyticsAgent: Handles market data and portfolio analysis
"""

import os
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage

from ..data_providers.manager import DataProviderManager
from ..portfolio.manager import PortfolioManager
from ..utils.metrics import FinancialMetricsCalculator
from .analytics_agent import AnalyticsAgent
from .llm_config import (
    DEFAULT_AZURE_API_VERSION,
    DEFAULT_AZURE_ENDPOINT,
    DEFAULT_VERTEX_LOCATION,
    MODEL_REGISTRY,
    LLMProvider,
    create_llm,
    get_default_model_key,
    get_default_provider,
    get_model_by_key,
)
from .orchestrator_agent import OrchestratorAgent
from .shared_state import SharedAgentState
from .transaction_agent import TransactionAgent

if TYPE_CHECKING:
    from ..services.market_data_service import MarketDataService


class PortfolioAgent:
    """AI agent for portfolio management and financial advice.

    This class serves as a backward-compatible facade for the multi-agent system.
    It internally uses an orchestrator to route queries to specialist agents:
    - TransactionAgent: For buy/sell/modify/delete operations
    - AnalyticsAgent: For portfolio analysis and market data

    Supports both DataProviderManager (legacy) and MarketDataService (new).
    """

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        data_manager: Union[DataProviderManager, "MarketDataService"],
        metrics_calculator: FinancialMetricsCalculator,
        openai_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_model: Optional[str] = None,
        azure_api_version: str = "2025-01-01-preview",
    ):
        """Initialize the portfolio agent.

        Args:
            portfolio_manager: PortfolioManager instance
            data_manager: DataProviderManager or MarketDataService instance
            metrics_calculator: FinancialMetricsCalculator instance
            openai_api_key: Optional OpenAI API key
            azure_endpoint: Optional Azure OpenAI endpoint
            azure_api_key: Optional Azure OpenAI API key
            azure_model: Optional Azure OpenAI model name
            azure_api_version: Azure API version
        """
        self.portfolio_manager = portfolio_manager
        self._data_manager = data_manager
        self.metrics_calculator = metrics_calculator
        self.azure_api_version = azure_api_version

        # Get or create MarketDataService
        from ..services.market_data_service import MarketDataService

        if isinstance(data_manager, MarketDataService):
            self.market_data_service = data_manager
        else:
            self.market_data_service = MarketDataService(data_manager)

        # Initialize shared state (memory shared across agents)
        self.shared_state = SharedAgentState()
        self.memory = self.shared_state.memory

        # Initialize LLM for specialist agents
        self.llm = None
        self._current_provider = None
        self._current_model = None
        self.set_llm_config(
            azure_endpoint=azure_endpoint
            or os.getenv("AZURE_OPENAI_ENDPOINT", "https://kallamai.openai.azure.com/"),
            azure_api_key=azure_api_key or os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_model=azure_model or "gpt-4.1-mini",
            openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY", ""),
        )

        # Create specialist agents
        self._create_agents()

    @property
    def data_manager(self) -> DataProviderManager:
        """Get the underlying DataProviderManager for compatibility."""
        if hasattr(self._data_manager, "data_manager"):
            return self._data_manager.data_manager
        return self._data_manager

    def _create_agents(self):
        """Create the specialist agents and orchestrator."""
        # Create Transaction Agent
        self.transaction_agent = TransactionAgent.create(
            portfolio_manager=self.portfolio_manager,
            market_data_service=self.market_data_service,
            metrics_calculator=self.metrics_calculator,
            llm=self.llm,
            memory=self.shared_state.memory,
        )

        # Create Analytics Agent
        self.analytics_agent = AnalyticsAgent.create(
            portfolio_manager=self.portfolio_manager,
            market_data_service=self.market_data_service,
            metrics_calculator=self.metrics_calculator,
            llm=self.llm,
            memory=self.shared_state.memory,
        )

        # Create Orchestrator with fast model
        self.orchestrator = OrchestratorAgent(
            transaction_agent=self.transaction_agent,
            analytics_agent=self.analytics_agent,
            memory=self.shared_state.memory,
            orchestrator_provider=self._get_orchestrator_provider(),
        )

    def _get_orchestrator_provider(self) -> str:
        """Determine the best provider for the orchestrator based on current config."""
        if self._current_provider:
            return self._current_provider

        # Check environment for available providers
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        elif os.getenv("AZURE_OPENAI_API_KEY"):
            return "azure-openai"
        elif os.getenv("GOOGLE_VERTEX_PROJECT"):
            return "vertex-ai"
        else:
            return "anthropic"  # Default

    def set_llm_config(
        self,
        model_key: Optional[str] = None,
        provider: Optional[str] = None,
        # Azure overrides
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_model: Optional[str] = None,
        # Vertex AI overrides
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        vertex_model: Optional[str] = None,
        # Legacy parameters (for backwards compatibility)
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        anthropic_api_key: Optional[str] = None,
        anthropic_model: Optional[str] = None,
    ) -> None:
        """Configure LLM provider and model using central config.

        Uses llm_config.py as the single source of truth.
        Supports both new model_key parameter and legacy provider-specific parameters.

        Args:
            model_key: Key from MODEL_REGISTRY (e.g., "claude-sonnet-4.5")
            provider: Provider name for legacy compatibility
            azure_endpoint: Optional Azure endpoint override
            azure_api_key: Optional Azure API key override
            azure_model: Optional Azure model override (legacy)
            vertex_project: Optional Vertex AI project override
            vertex_location: Optional Vertex AI location override
            vertex_model: Optional Vertex AI model override (legacy)
        """
        try:
            # Determine model key from parameters
            effective_model_key = model_key

            if not effective_model_key:
                # Legacy: map provider + model to model_key
                provider_normalized = (provider or "").lower()
                if provider_normalized in ("azure", "azure-openai", "azure_openai"):
                    # Find matching model in registry
                    model_id = azure_model or "gpt-4.1-mini"
                    for key, config in MODEL_REGISTRY.items():
                        if config.model_id == model_id and config.provider == LLMProvider.AZURE_OPENAI:
                            effective_model_key = key
                            break
                    if not effective_model_key:
                        effective_model_key = "gpt-4.1-mini"
                elif provider_normalized in ("vertex", "vertex-ai", "google", "google-vertex"):
                    model_id = vertex_model or "claude-haiku-4-5"
                    for key, config in MODEL_REGISTRY.items():
                        if config.model_id == model_id and config.provider == LLMProvider.VERTEX_AI:
                            effective_model_key = key
                            break
                    if not effective_model_key:
                        effective_model_key = "claude-haiku-4.5"
                elif provider_normalized == "anthropic":
                    effective_model_key = "claude-sonnet-4.5"
                else:
                    effective_model_key = get_default_model_key()

            # Create LLM using central config
            self.llm = create_llm(
                model_key=effective_model_key,
                azure_endpoint=azure_endpoint,
                azure_api_key=azure_api_key,
                vertex_project=vertex_project,
                vertex_location=vertex_location,
            )

            # Update tracking
            config = get_model_by_key(effective_model_key)
            if config:
                self._current_provider = config.provider.value
                self._current_model = config.model_id
            else:
                self._current_provider = get_default_provider()
                self._current_model = effective_model_key

            self._update_specialist_llms()

        except Exception as e:
            # Fallback to default model
            import logging
            logging.warning(f"Failed to set LLM config: {e}. Using default.")
            default_key = get_default_model_key()
            self.llm = create_llm(default_key)
            config = get_model_by_key(default_key)
            self._current_provider = config.provider.value if config else "vertex-ai"
            self._current_model = config.model_id if config else default_key

    def _update_specialist_llms(self):
        """Update the LLM for specialist agents after configuration change."""
        if hasattr(self, "orchestrator") and self.orchestrator:
            self.orchestrator.update_specialist_llm(self.llm)

    def chat(self, message: str) -> str:
        """Process a chat message and return response.

        Routes the message through the orchestrator to the appropriate
        specialist agent (Transaction or Analytics).

        Args:
            message: User's message

        Returns:
            Response from the appropriate agent
        """
        try:
            # Get current portfolio context
            context = self.shared_state.get_portfolio_context(self.portfolio_manager)

            # Route through orchestrator
            response = self.orchestrator.route_and_execute(message, context)

            # Invalidate context cache after potential modifications
            self.shared_state.invalidate_context()

            return response

        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try rephrasing your request."

    def initialize_conversation(self) -> str:
        """Initialize conversation with portfolio overview."""
        try:
            if not self.portfolio_manager.current_portfolio:
                return """Hello! I'm your AI financial advisor and portfolio manager.

I can help you with:
- Managing your investment portfolio
- Analyzing performance metrics
- Providing investment advice
- Researching stocks and market conditions
- Tracking transactions and positions

To get started, I can create a new portfolio for you or load an existing one. Just let me know what you'd like to do!

*Disclaimer: This is for educational purposes. Always consult qualified financial professionals for personalized investment advice.*"""

            else:
                # Get portfolio summary via analytics agent
                summary_response = self.analytics_agent.invoke(
                    "Give me a brief portfolio summary",
                    context=self.shared_state.get_portfolio_context(self.portfolio_manager),
                )

                return f"""Welcome back! Here's your current portfolio status:

{summary_response}

How can I help you with your investments today? I can:
- Add new transactions or update positions
- Analyze your portfolio performance
- Research investment opportunities
- Provide market insights and advice

*Disclaimer: This is for educational purposes. Always consult qualified financial professionals for personalized investment advice.*"""

        except Exception as e:
            return f"Hello! I'm ready to help with your portfolio. (Note: {str(e)})"

    def process_transaction_from_text(self, text: str) -> str:
        """Process natural language transaction input.

        Routes directly to the Transaction Agent.

        Args:
            text: Natural language transaction description

        Returns:
            Response from Transaction Agent
        """
        context = self.shared_state.get_portfolio_context(self.portfolio_manager)
        response = self.transaction_agent.invoke(
            f"Please help me add this transaction: {text}",
            context=context,
        )
        self.shared_state.invalidate_context()
        return response

    def get_investment_advice(self, query: str) -> str:
        """Get investment advice based on portfolio and market conditions.

        Routes to the Analytics Agent for comprehensive analysis.

        Args:
            query: Investment advice query

        Returns:
            Investment advice from Analytics Agent
        """
        advice_prompt = f"""Based on my current portfolio and current market conditions, please provide investment advice for: {query}

Please:
1. Research current market conditions and news related to this query
2. Analyze my portfolio's current allocation and risk profile
3. Provide specific, actionable recommendations
4. Include appropriate risk considerations and diversification advice
5. Suggest position sizing if recommending new investments

Remember to include proper disclaimers."""

        context = self.shared_state.get_portfolio_context(self.portfolio_manager)
        return self.analytics_agent.invoke(advice_prompt, context=context)

    def analyze_portfolio_performance(self, days: int = 365) -> str:
        """Analyze portfolio performance over specified period.

        Routes to the Analytics Agent.

        Args:
            days: Number of days to analyze

        Returns:
            Performance analysis from Analytics Agent
        """
        analysis_prompt = f"""Please provide a comprehensive analysis of my portfolio performance over the last {days} days. Include:

1. Performance metrics (returns, volatility, Sharpe ratio, etc.)
2. Comparison to major market indices
3. Risk analysis (drawdown, VaR, etc.)
4. Portfolio allocation breakdown
5. Recommendations for improvement

Also check data freshness and refresh if needed."""

        context = self.shared_state.get_portfolio_context(self.portfolio_manager)
        return self.analytics_agent.invoke(analysis_prompt, context=context)

    def clear_conversation(self):
        """Clear conversation memory."""
        self.shared_state.clear()

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        try:
            messages = self.memory.chat_memory.messages
            history = []

            for message in messages:
                if isinstance(message, HumanMessage):
                    history.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    history.append({"role": "assistant", "content": message.content})

            return history
        except Exception:
            return []

    # Legacy compatibility properties and methods

    @property
    def portfolio_tools(self):
        """Legacy compatibility: Return combined tools from both agents."""
        return self.transaction_agent.tools + self.analytics_agent.tools

    @property
    def tools(self):
        """Legacy compatibility: Return all tools."""
        return self.portfolio_tools

    @property
    def agent_executor(self):
        """Legacy compatibility: Return analytics agent executor as default."""
        return self.analytics_agent.agent_executor
