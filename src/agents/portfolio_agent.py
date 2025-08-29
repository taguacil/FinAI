"""
Portfolio AI agent using LangGraph for financial advice and portfolio management.
"""

import os
from typing import Dict, List, Optional

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI

from ..data_providers.manager import DataProviderManager
from ..portfolio.manager import PortfolioManager
from ..utils.metrics import FinancialMetricsCalculator
from .tools import create_portfolio_tools


class PortfolioAgent:
    """AI agent for portfolio management and financial advice."""

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        data_manager: DataProviderManager,
        metrics_calculator: FinancialMetricsCalculator,
        openai_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_model: Optional[str] = None,
        azure_api_version: str = "2025-01-01-preview",
    ):
        """Initialize the portfolio agent."""

        self.portfolio_manager = portfolio_manager
        self.data_manager = data_manager
        self.metrics_calculator = metrics_calculator

        # Initialize LLM (prefer Azure OpenAI if configured)
        self.llm = None
        self.azure_api_version = azure_api_version
        self.set_llm_config(
            azure_endpoint=azure_endpoint
            or os.getenv("AZURE_OPENAI_ENDPOINT", "https://kallamai.openai.azure.com/"),
            azure_api_key=azure_api_key or os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_model=azure_model or "gpt-4.1-mini",
            openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY", ""),
        )

        # Create tools
        self.portfolio_tools = create_portfolio_tools(
            portfolio_manager, data_manager, metrics_calculator
        )

        # Add web search tool for financial news and market data
        self.web_search_tool = self._create_web_search_tool()

        # All tools
        self.tools = self.portfolio_tools + [self.web_search_tool]

        # Memory for conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # Create agent
        self.agent_executor = self._create_agent()

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
            provider_normalized = (provider or "").lower()

            if provider_normalized == "anthropic":
                key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
                model = anthropic_model or "claude-3-5-sonnet-20240620"
                self.llm = ChatAnthropic(model=model, temperature=0.1, api_key=key)
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
                # Credentials are expected via GOOGLE_APPLICATION_CREDENTIALS or default ADC
                self.llm = ChatVertexAI(
                    model_name=model_name,
                    project=project,
                    location=location,
                    temperature=0.1,
                )
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
                return

            # Default fallback to OpenAI
            key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
            self.llm = ChatOpenAI(model=openai_model, temperature=0.1, api_key=key)

        except Exception:
            # Last resort minimal fallback to avoid crashing UI
            key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
            self.llm = ChatOpenAI(model=openai_model, temperature=0.1, api_key=key)

    def _create_web_search_tool(self) -> Tool:
        """Create web search tool for financial information."""

        def search_web(query: str) -> str:
            """Search the web for financial information using Tavily."""
            try:
                import os

                from langchain_community.tools.tavily_search import TavilySearchResults

                tavily_key = os.getenv("TAVILY_API_KEY", "")
                if not tavily_key:
                    return "❌ Tavily API key not configured. Set TAVILY_API_KEY in your environment."
                tool = TavilySearchResults(max_results=5)
                results = tool.run(query)
                if not results:
                    return "No results found."
                lines = ["🔍 Tavily Web Search:"]
                for r in results:
                    title = r.get("title") or r.get("url")
                    snippet = r.get("content") or r.get("snippet") or ""
                    url = r.get("url") or ""
                    lines.append(f"• {title}\n  {snippet}\n  {url}")
                return "\n".join(lines)
            except Exception as e:
                return f"❌ Error searching web: {str(e)}"

        return Tool(
            name="web_search",
            description="Search the web for current financial news, market information, and investment analysis. Use this for real-time market data, company news, economic indicators, and investment advice.",
            func=search_web,
        )

    def _create_agent(self) -> AgentExecutor:
        """Create the portfolio agent."""

        system_prompt = """You are a professional financial advisor and portfolio manager AI assistant.

        Your capabilities include:
        - Managing investment portfolios (adding transactions, tracking positions)
        - Analyzing portfolio performance (calculating metrics like Sharpe ratio, alpha, beta, volatility)
        - Providing investment advice based on current market conditions
        - Searching for financial instruments and getting current prices
        - Accessing real-time financial news and market data through web search

        Guidelines:
        1. Always prioritize the user's financial goals and risk tolerance
        2. Provide clear, actionable advice with proper risk disclaimers
        3. Use portfolio tools to track and analyze the user's investments
        4. Search the web for current market conditions when giving advice
        5. Calculate and present relevant financial metrics
        6. Be conversational but professional
        7. Always include appropriate risk warnings and disclaimers

        CRITICAL RULE: FOLLOW USER INSTRUCTIONS EXACTLY!
        - If the user says "bond", use instrument_type="bond"
        - If the user says "stock", use instrument_type="stock"
        - If the user says "EUR" currency, use currency="EUR"
        - If the user specifies "equity", use instrument_type="stock"
        - NEVER change what the user explicitly specified
        - Only use tool calls and web search to fill in MISSING information, not to override user specifications

        When the user mentions buying/selling stocks, bonds, or updating their portfolio:
        - Parse the transaction details carefully (symbol, quantity, price, date, ISIN if provided)
        - RESPECT the user's explicit instrument type (bond, stock, etf, etc.)
        - RESPECT the user's explicit currency (EUR, USD, etc.)
        - For bonds with ISINs (especially XS-prefixed ISINs), always use the ISIN parameter
        - For bond prices expressed as percentages (e.g., 98.85%), use the percentage value directly
        - Use the add_transaction tool to record the transaction with the user's exact specifications
        - Update the portfolio summary afterward

        CRITICAL TRANSACTION PARSING RULES:
        1. **USER SPECIFICATIONS ARE ABSOLUTE**: Never change what the user explicitly stated
        2. **Extract EXACTLY what the user said**:
           - ISIN if mentioned (e.g., "ISIN XS2472298335", "ISIN US0378331005")
           - Symbol if mentioned (e.g., "AAPL", "TSLA", "MSFT")
           - Instrument type if mentioned (e.g., "bond", "stock", "equity")
           - Currency if mentioned (e.g., "EUR", "USD", "GBP")
           - Company/instrument name if mentioned (e.g., "Apple Inc.", "Tesla Inc.")
        3. **Use tool calls ONLY for missing information**:
           - If user says "bond in EUR" → instrument_type="bond", currency="EUR"
           - If user says "XS1234567890 bond" → isin="XS1234567890", instrument_type="bond"
           - If missing symbol, search for it. If missing price, ask for it.
        4. **Preserve exact values**: bond prices as percentages (98.85% → price: 98.85)

        Instrument Type Guidelines (use ONLY when user doesn't specify):
        - **Stocks/Equities**: Individual company shares (AAPL, MSFT, TSLA)
        - **ETFs**: Exchange-traded funds (SPY, QQQ, VTI, ARKK)
        - **Bonds**: Fixed income instruments (TLT, IEF, BND, XS-prefixed ISINs)
        - **Crypto**: Cryptocurrencies (BTC, ETH, SOL)
        - **Cash**: Deposits, withdrawals, fees

        Use web search ONLY to find missing information (never to override user specifications):
        - Search for "Apple Inc. stock symbol" to find "AAPL" (if symbol missing)
        - Search for "Apple Inc. ISIN" to find "US0378331005" (if ISIN missing)
        - Search for "XS2472298335 bond details" to find bond information (if details missing)

        Examples of FOLLOWING USER INSTRUCTIONS EXACTLY:
        - "Buy 50 AAPL bonds in EUR"
          → symbol: "AAPL", quantity: 50, instrument_type: "bond", currency: "EUR" (user said "bonds", use that!)
        - "Buy 50 bonds using ISIN XS2472298335 in EUR at 98.85%"
          → isin: "XS2472298335", quantity: 50, price: 98.85, instrument_type: "bond", currency: "EUR"
        - "Add 100 TLT as equity in USD"
          → symbol: "TLT", quantity: 100, instrument_type: "stock", currency: "USD" (user said "equity"!)
        - "Buy 50000 EUR bonds with ISIN XS2472298335 at 98.85%"
          → isin: "XS2472298335", quantity: 50000, price: 98.85, instrument_type: "bond", currency: "EUR"
        - "Purchase 100 Apple stock at $150"
          → symbol: "AAPL", quantity: 100, price: 150, instrument_type: "stock" (user said "stock")
        - "Buy 50 Microsoft bonds in EUR at 95%"
          → symbol: "MSFT", quantity: 50, price: 95, instrument_type: "bond", currency: "EUR"

        Examples when user doesn't specify type (then auto-detect):
        - "Buy 50 AAPL at $150"
          → symbol: "AAPL", quantity: 50, price: 150 (system auto-detects: instrument_type: "stock")
        - "Buy 100 using ISIN XS2472298335 at 98.85%"
          → isin: "XS2472298335", quantity: 100, price: 98.85 (system auto-detects: instrument_type: "bond")
        - "Buy 5 BTC at $45000"
          → symbol: "BTC", quantity: 5, price: 45000 (system auto-detects: instrument_type: "crypto")

        When asked for investment advice:
        - Search for current market conditions and news
        - Analyze the user's portfolio performance and risk metrics
        - Consider diversification and risk management
        - Provide specific, actionable recommendations

        Remember: This is for educational purposes. Always recommend consulting with qualified financial professionals for personalized advice.
        """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_openai_tools_agent(llm=self.llm, tools=self.tools, prompt=prompt)

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            max_iterations=5,
        )

    def chat(self, message: str) -> str:
        """Process a chat message and return response."""
        try:
            # Add current portfolio context if available
            context = self._get_portfolio_context()
            if context:
                enhanced_message = (
                    f"Current portfolio context: {context}\n\nUser message: {message}"
                )
            else:
                enhanced_message = message

            response = self.agent_executor.invoke({"input": enhanced_message})
            return response.get("output", "I'm sorry, I couldn't process that request.")

        except Exception as e:
            return f"❌ I encountered an error: {str(e)}. Please try rephrasing your request."

    def _get_portfolio_context(self) -> str:
        """Get brief portfolio context for the agent."""
        try:
            if not self.portfolio_manager.current_portfolio:
                return "No portfolio is currently loaded."

            portfolio = self.portfolio_manager.current_portfolio
            total_value = self.portfolio_manager.get_portfolio_value()
            positions_count = len(portfolio.positions)

            return (
                f"Portfolio '{portfolio.name}' loaded with "
                f"${total_value:,.2f} total value across {positions_count} positions."
            )

        except Exception:
            return ""

    def initialize_conversation(self) -> str:
        """Initialize conversation with portfolio overview."""
        try:
            if not self.portfolio_manager.current_portfolio:
                return """👋 Hello! I'm your AI financial advisor and portfolio manager.

I can help you with:
- 📊 Managing your investment portfolio
- 📈 Analyzing performance metrics
- 💡 Providing investment advice
- 🔍 Researching stocks and market conditions
- 📝 Tracking transactions and positions

To get started, I can create a new portfolio for you or load an existing one. Just let me know what you'd like to do!

*Disclaimer: This is for educational purposes. Always consult qualified financial professionals for personalized investment advice.*"""

            else:
                # Get portfolio summary
                summary_tool = next(
                    (
                        tool
                        for tool in self.portfolio_tools
                        if tool.name == "get_portfolio_summary"
                    ),
                    None,
                )

                if summary_tool:
                    summary = summary_tool._run(include_metrics=True)
                    return f"""👋 Welcome back! Here's your current portfolio status:

{summary}

How can I help you with your investments today? I can:
- Add new transactions or update positions
- Analyze your portfolio performance
- Research investment opportunities
- Provide market insights and advice

*Disclaimer: This is for educational purposes. Always consult qualified financial professionals for personalized investment advice.*"""

                else:
                    return "👋 Welcome back! Your portfolio is loaded. How can I help you today?"

        except Exception as e:
            return f"👋 Hello! I'm ready to help with your portfolio. (Note: {str(e)})"

    def process_transaction_from_text(self, text: str) -> str:
        """Process natural language transaction input."""
        # This would be enhanced with better NLP parsing
        # For now, direct the user to use specific format
        return self.chat(f"Please help me add this transaction: {text}")

    def get_investment_advice(self, query: str) -> str:
        """Get investment advice based on portfolio and market conditions."""
        advice_prompt = f"""Based on my current portfolio and current market conditions, please provide investment advice for: {query}

Please:
1. Research current market conditions and news related to this query
2. Analyze my portfolio's current allocation and risk profile
3. Provide specific, actionable recommendations
4. Include appropriate risk considerations and diversification advice
5. Suggest position sizing if recommending new investments

Remember to include proper disclaimers."""

        return self.chat(advice_prompt)

    def analyze_portfolio_performance(self, days: int = 365) -> str:
        """Analyze portfolio performance over specified period."""
        analysis_prompt = f"""Please provide a comprehensive analysis of my portfolio performance over the last {days} days. Include:

1. Performance metrics (returns, volatility, Sharpe ratio, etc.)
2. Comparison to major market indices
3. Risk analysis (drawdown, VaR, etc.)
4. Portfolio allocation breakdown
5. Recommendations for improvement

Also search for current market conditions that might affect my portfolio."""

        return self.chat(analysis_prompt)

    def clear_conversation(self):
        """Clear conversation memory."""
        self.memory.clear()

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
