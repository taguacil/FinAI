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
from .llm_config import MODEL_REGISTRY, LLMProvider, create_llm, create_llm_from_config
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
        provider: Optional[str] = None,
        # Azure
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_model: Optional[str] = None,
        # OpenAI fallback
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
        # Anthropic
        anthropic_api_key: Optional[str] = None,
        anthropic_model: Optional[str] = None,
        # Google Vertex AI
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        vertex_model: Optional[str] = None,
    ) -> None:
        """Configure LLM provider and model.

        Supported providers: 'azure-openai', 'openai', 'anthropic', 'vertex-ai'.
        """
        try:
            from langchain_anthropic import ChatAnthropic
            from langchain_google_vertexai import ChatVertexAI
            from langchain_openai import AzureChatOpenAI, ChatOpenAI

            provider_normalized = (provider or "").lower()

            if provider_normalized == "anthropic":
                key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
                model = anthropic_model or "claude-sonnet-4-20250514"
                self.llm = ChatAnthropic(model=model, temperature=0.1, api_key=key)
                self._current_provider = "anthropic"
                self._current_model = model
                self._update_specialist_llms()
                return

            if provider_normalized in (
                "vertex",
                "vertex-ai",
                "google",
                "google-vertex",
            ):
                project = vertex_project or os.getenv("GOOGLE_VERTEX_PROJECT", "")
                location = vertex_location or os.getenv(
                    "GOOGLE_VERTEX_LOCATION", "us-central1"
                )
                model_name = vertex_model or "gemini-2.0-flash-lite-001"
                self.llm = ChatVertexAI(
                    model_name=model_name,
                    project=project,
                    location=location,
                    temperature=0.1,
                )
                self._current_provider = "vertex-ai"
                self._current_model = model_name
                self._update_specialist_llms()
                return

            if provider_normalized in ("azure", "azure-openai", "azure_openai") or (
                azure_endpoint and azure_api_key and azure_model
            ):
                self.llm = AzureChatOpenAI(
                    azure_endpoint=azure_endpoint
                    or os.getenv(
                        "AZURE_OPENAI_ENDPOINT", "https://kallamai.openai.azure.com/"
                    ),
                    api_version=self.azure_api_version,
                    api_key=azure_api_key or os.getenv("AZURE_OPENAI_API_KEY", ""),
                    model=azure_model or "gpt-4.1-mini",
                    temperature=0.1,
                )
                self._current_provider = "azure-openai"
                self._current_model = azure_model or "gpt-4.1-mini"
                self._update_specialist_llms()
                return

            # Default fallback to OpenAI
            key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
            self.llm = ChatOpenAI(model=openai_model, temperature=0.1, api_key=key)
            self._current_provider = "openai"
            self._current_model = openai_model

            self._update_specialist_llms()

        except Exception:
            # Last resort minimal fallback to avoid crashing UI
            from langchain_openai import ChatOpenAI

            key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
            self.llm = ChatOpenAI(model=openai_model, temperature=0.1, api_key=key)
            self._current_provider = "openai"
            self._current_model = openai_model

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
