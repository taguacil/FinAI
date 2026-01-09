"""
Analytics Agent for market data and portfolio analysis.

Handles data fetching, portfolio analysis, metrics calculation,
scenario simulations, and general financial research.
"""

from typing import List

from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool, Tool
from langchain_core.language_models import BaseChatModel

from ..data_providers.manager import DataProviderManager
from ..portfolio.manager import PortfolioManager
from ..services.market_data_service import MarketDataService
from ..utils.metrics import FinancialMetricsCalculator
from .base_agent import BaseAgent
from .tools import (
    AdvancedWhatIfTool,
    CalculatorTool,
    GetCurrentPriceTool,
    GetPortfolioMetricsTool,
    GetPortfolioSummaryTool,
    GetTransactionHistoryTool,
    GetTransactionsTool,
    HypotheticalPositionTool,
    IngestPdfTool,
    SimulateWhatIfTool,
)
from .tools.market_data_tools import create_market_data_tools


class AnalyticsAgent(BaseAgent):
    """Specialist agent for market data and portfolio analytics.

    Responsibilities:
    - Portfolio performance analysis and metrics
    - Market data queries (prices, FX rates, historical data)
    - What-if scenarios and Monte Carlo simulations
    - Data freshness monitoring and refresh
    - PDF document analysis for financial research
    - Web search for market information
    """

    def get_agent_name(self) -> str:
        return "Analytics Agent"

    def get_system_prompt(self) -> str:
        return """You are an analytics specialist for a financial portfolio management system.

Your responsibilities:
- Portfolio performance analysis and metrics calculation
- Market data queries (current prices, historical prices, FX rates)
- What-if scenarios and Monte Carlo simulations
- Data freshness monitoring and coordinated refresh
- PDF document analysis for financial research
- Web search for market information and news

You have FULL access to the MarketDataService for:
- Current and historical prices via get_price_history
- FX rates with staleness tracking via get_fx_rate
- Batch price queries via get_batch_prices
- Data freshness status via get_data_freshness
- Forced data refresh via refresh_data

You also have READ-ONLY access to transaction data for context.

Key metrics you can calculate:
- Sharpe ratio, Sortino ratio, Calmar ratio
- Alpha, Beta, Information ratio
- Volatility, VaR, CVaR (Conditional Value at Risk)
- Maximum drawdown and drawdown duration
- Time-weighted and money-weighted returns

When analyzing portfolios:
1. Always check data freshness first using get_data_freshness
2. Refresh data if stale before important calculations
3. Provide clear explanations of metrics and their implications
4. Include appropriate risk disclaimers
5. Compare to benchmarks when relevant (default: SPY)

For scenario analysis:
- Use simulate_what_if for simple exclusion scenarios
- Use advanced_what_if for Monte Carlo simulations with custom assumptions
- Use test_hypothetical_position for testing new investment ideas

When providing analysis:
- Be specific and quantitative
- Explain what the numbers mean in practical terms
- Highlight key risks and opportunities
- Suggest actionable insights when appropriate

Data Freshness Guidelines:
- Prices older than 1 hour are considered stale
- Always note the data age when reporting prices
- Recommend refresh before major decisions

Remember: This is for educational purposes. Always recommend consulting with qualified financial professionals for personalized investment advice."""

    @classmethod
    def create_tools(
        cls,
        portfolio_manager: PortfolioManager,
        market_data_service: MarketDataService,
        metrics_calculator: FinancialMetricsCalculator,
        include_web_search: bool = True,
    ) -> List[BaseTool]:
        """Create analytics-specific tools.

        Args:
            portfolio_manager: PortfolioManager instance
            market_data_service: MarketDataService instance
            metrics_calculator: FinancialMetricsCalculator instance
            include_web_search: Whether to include web search tool

        Returns:
            List of tools for the Analytics Agent
        """
        data_manager = market_data_service.data_manager

        tools = [
            # Portfolio analysis
            GetPortfolioSummaryTool(portfolio_manager, metrics_calculator),
            GetPortfolioMetricsTool(portfolio_manager, metrics_calculator),
            # Read-only transaction access (for context)
            GetTransactionsTool(portfolio_manager),
            GetTransactionHistoryTool(portfolio_manager),
            # Scenarios
            SimulateWhatIfTool(portfolio_manager),
            AdvancedWhatIfTool(portfolio_manager),
            HypotheticalPositionTool(portfolio_manager),
            # Basic price lookup (legacy)
            GetCurrentPriceTool(data_manager),
            # Utilities
            CalculatorTool(),
            IngestPdfTool(),
        ]

        # Add MarketDataService tools
        market_tools = create_market_data_tools(
            market_data_service, portfolio_manager
        )
        tools.extend(market_tools)

        # Add web search if requested
        if include_web_search:
            web_tool = cls._create_web_search_tool()
            if web_tool:
                tools.append(web_tool)

        return tools

    @staticmethod
    def _create_web_search_tool() -> Tool:
        """Create web search tool using Vercel AI Gateway with Perplexity Sonar Pro."""

        def search_web(query: str) -> str:
            """Search the web for financial information using Perplexity Sonar Pro via Vercel AI Gateway."""
            try:
                import json
                import os

                import requests

                # Vercel AI Gateway configuration
                gateway_url = os.getenv(
                    "VERCEL_AI_GATEWAY_URL",
                    "https://gateway.ai.vercel.app/v1"
                )
                api_key = os.getenv("VERCEL_AI_GATEWAY_API_KEY", "")

                if not api_key:
                    return "Vercel AI Gateway API key not configured. Set VERCEL_AI_GATEWAY_API_KEY in your environment."

                # Prepare the request for Perplexity Sonar Pro with deep research
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }

                # Use sonar-pro model for deep research
                payload = {
                    "model": "sonar-pro",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a financial research assistant. Provide comprehensive, well-sourced answers about financial markets, stocks, economic indicators, and investment analysis. Include relevant data, statistics, and cite your sources."
                        },
                        {
                            "role": "user",
                            "content": query
                        }
                    ],
                    "search_domain_filter": [],  # No domain restrictions
                    "return_citations": True,
                    "search_recency_filter": "month",  # Focus on recent information
                }

                response = requests.post(
                    f"{gateway_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60  # Deep research may take longer
                )

                if response.status_code != 200:
                    return f"Web search failed (HTTP {response.status_code}): {response.text}"

                result = response.json()

                # Extract the response content
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

                if not content:
                    return "No results found for this query."

                # Format the response
                lines = ["Web Research Results:", ""]
                lines.append(content)

                # Add citations if available
                citations = result.get("citations", [])
                if citations:
                    lines.append("")
                    lines.append("Sources:")
                    for i, citation in enumerate(citations[:10], 1):
                        lines.append(f"{i}. {citation}")

                return "\n".join(lines)

            except requests.exceptions.Timeout:
                return "Web search timed out. Please try a more specific query."
            except Exception as e:
                return f"Error performing web search: {str(e)}"

        return Tool(
            name="web_search",
            description="Perform deep web research for financial information using Perplexity Sonar Pro. Use this for comprehensive analysis of stocks, market trends, company news, economic indicators, and investment research. Returns well-sourced, detailed answers.",
            func=search_web,
        )

    @classmethod
    def create(
        cls,
        portfolio_manager: PortfolioManager,
        market_data_service: MarketDataService,
        metrics_calculator: FinancialMetricsCalculator,
        llm: BaseChatModel,
        memory: ConversationBufferMemory,
        include_web_search: bool = True,
    ) -> "AnalyticsAgent":
        """Factory method to create an AnalyticsAgent with its tools.

        Args:
            portfolio_manager: PortfolioManager instance
            market_data_service: MarketDataService instance
            metrics_calculator: FinancialMetricsCalculator instance
            llm: Language model instance
            memory: Shared conversation memory
            include_web_search: Whether to include web search tool

        Returns:
            Configured AnalyticsAgent instance
        """
        tools = cls.create_tools(
            portfolio_manager,
            market_data_service,
            metrics_calculator,
            include_web_search,
        )
        return cls(
            portfolio_manager=portfolio_manager,
            market_data_service=market_data_service,
            metrics_calculator=metrics_calculator,
            llm=llm,
            memory=memory,
            tools=tools,
        )
