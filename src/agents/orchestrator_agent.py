"""
Orchestrator Agent for query classification and routing.

Uses a fast, lightweight model to classify incoming queries and route them
to the appropriate specialist agent (Transaction or Analytics).
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

from langchain.memory import ConversationBufferMemory
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from .analytics_agent import AnalyticsAgent
from .llm_config import (
    FAST_MODELS,
    MODEL_REGISTRY,
    LLMProvider,
    create_llm_from_config,
    get_default_fast_model,
)
from .transaction_agent import TransactionAgent


class QueryCategory(str, Enum):
    """Categories for query classification."""

    TRANSACTIONAL = "transactional"
    ANALYTICS = "analytics"
    GREETING = "greeting"
    UNCLEAR = "unclear"


@dataclass
class QueryClassification:
    """Result of query classification."""

    category: QueryCategory
    confidence: float
    reasoning: str


class OrchestratorAgent:
    """Lightweight orchestrator for query classification and routing.

    Uses a fast model (e.g., Claude Haiku, GPT-4.1 Mini) for classification,
    then routes to the appropriate specialist agent.
    """

    # Classification prompt template
    CLASSIFICATION_PROMPT = """You are a query classifier for a financial portfolio management system.

Classify the user's query into one of these categories:
- "transactional": Adding, modifying, or deleting transactions; searching for instruments to add; any CRUD operations on portfolio data
- "analytics": Market data queries, portfolio analysis, metrics, scenarios, what-if analysis, price history, performance analysis, research
- "greeting": Simple greetings, introductions, or general conversation starters
- "unclear": Cannot determine intent or ambiguous query

Classification Guidelines:
- "Buy", "Sell", "Add", "Delete", "Modify", "Remove" → transactional
- "Show", "Analyze", "Calculate", "What is", "How much", "Performance" → analytics
- "Search for [symbol] to add" → transactional (searching with intent to add)
- "What's the price of" → analytics (just looking up data)
- "Hello", "Hi", "Help" → greeting

Examples:
- "Buy 100 shares of AAPL at $150" → transactional
- "What is my portfolio return?" → analytics
- "Add a deposit of $5000" → transactional
- "Show me the price history of TSLA" → analytics
- "Delete the transaction from yesterday" → transactional
- "Run a Monte Carlo simulation" → analytics
- "Search for Microsoft stock" → transactional (likely to add)
- "What's my Sharpe ratio?" → analytics
- "Hi, what can you do?" → greeting
- "Modify the last transaction" → transactional

Query: {query}

Respond with ONLY a JSON object (no markdown, no explanation):
{{"category": "transactional|analytics|greeting|unclear", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

    def __init__(
        self,
        transaction_agent: TransactionAgent,
        analytics_agent: AnalyticsAgent,
        memory: ConversationBufferMemory,
        orchestrator_provider: str = "anthropic",
        fast_model_key: Optional[str] = None,
    ):
        """Initialize the orchestrator.

        Args:
            transaction_agent: TransactionAgent instance
            analytics_agent: AnalyticsAgent instance
            memory: Shared conversation memory
            orchestrator_provider: Provider for fast model ("anthropic", "azure-openai", "vertex-ai")
            fast_model_key: Optional specific fast model key from MODEL_REGISTRY
        """
        self.transaction_agent = transaction_agent
        self.analytics_agent = analytics_agent
        self.memory = memory
        self.llm = self._create_fast_llm(orchestrator_provider, fast_model_key)

    def _create_fast_llm(
        self,
        provider: str,
        model_key: Optional[str] = None,
    ) -> BaseChatModel:
        """Create a fast, lightweight LLM for routing.

        Args:
            provider: LLM provider name
            model_key: Optional specific model key

        Returns:
            Configured fast LLM instance
        """
        # If specific model key provided, use it
        if model_key and model_key in MODEL_REGISTRY:
            config = MODEL_REGISTRY[model_key]
            return create_llm_from_config(config, temperature=0.0)

        # Otherwise, find fastest model for the provider
        provider_mapping = {
            "anthropic": LLMProvider.ANTHROPIC,
            "azure-openai": LLMProvider.AZURE_OPENAI,
            "azure": LLMProvider.AZURE_OPENAI,
            "vertex-ai": LLMProvider.VERTEX_AI,
            "google": LLMProvider.VERTEX_AI,
        }

        llm_provider = provider_mapping.get(provider.lower())

        if llm_provider:
            # Find fast model for this provider
            for config in FAST_MODELS.values():
                if config.provider == llm_provider:
                    return create_llm_from_config(config, temperature=0.0)

        # Default to Claude Haiku
        return create_llm_from_config(get_default_fast_model(), temperature=0.0)

    def classify_query(self, query: str) -> QueryClassification:
        """Classify a user query into transaction or analytics category.

        Args:
            query: User's query string

        Returns:
            QueryClassification with category, confidence, and reasoning
        """
        try:
            prompt = self.CLASSIFICATION_PROMPT.format(query=query)
            response = self.llm.invoke([HumanMessage(content=prompt)])

            # Parse JSON response
            content = response.content.strip()

            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)

            category = QueryCategory(data.get("category", "unclear"))
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")

            return QueryClassification(
                category=category,
                confidence=confidence,
                reasoning=reasoning,
            )

        except Exception as e:
            logging.warning(f"Classification failed: {e}, defaulting to analytics")
            return QueryClassification(
                category=QueryCategory.UNCLEAR,
                confidence=0.3,
                reasoning=f"Classification error: {str(e)}",
            )

    def route_and_execute(self, query: str, portfolio_context: str = "") -> str:
        """Classify, route to appropriate agent, and return response.

        Args:
            query: User's query string
            portfolio_context: Current portfolio context for agents

        Returns:
            Response from the appropriate specialist agent
        """
        classification = self.classify_query(query)

        logging.info(
            f"Query classified as {classification.category.value} "
            f"(confidence: {classification.confidence:.2f}): {classification.reasoning}"
        )

        if classification.category == QueryCategory.TRANSACTIONAL:
            return self.transaction_agent.invoke(query, context=portfolio_context)

        elif classification.category == QueryCategory.ANALYTICS:
            return self.analytics_agent.invoke(query, context=portfolio_context)

        elif classification.category == QueryCategory.GREETING:
            return self._handle_greeting(query)

        else:
            # Unclear - try analytics as fallback (most informative)
            logging.info("Unclear query, routing to analytics agent as fallback")
            return self.analytics_agent.invoke(query, context=portfolio_context)

    def _handle_greeting(self, query: str) -> str:
        """Handle greeting queries with a friendly response.

        Args:
            query: User's greeting

        Returns:
            Friendly greeting response
        """
        return """Hello! I'm your AI financial advisor and portfolio manager.

I can help you with:

**Transactions:**
- Buy or sell stocks, bonds, ETFs, and crypto
- Add deposits, withdrawals, and dividends
- Modify or delete existing transactions
- Search for instruments by symbol or company name

**Analytics:**
- Portfolio performance analysis and metrics
- Historical price data and FX rates
- What-if scenarios and Monte Carlo simulations
- Market research and news

What would you like to do today?

*Disclaimer: This is for educational purposes. Always consult qualified financial professionals for personalized investment advice.*"""

    def update_specialist_llm(self, llm: BaseChatModel):
        """Update the LLM for both specialist agents.

        Args:
            llm: New language model instance
        """
        self.transaction_agent.update_llm(llm)
        self.analytics_agent.update_llm(llm)

    def update_orchestrator_llm(
        self,
        provider: str = "anthropic",
        model_key: Optional[str] = None,
    ):
        """Update the orchestrator's fast LLM.

        Args:
            provider: LLM provider name
            model_key: Optional specific model key
        """
        self.llm = self._create_fast_llm(provider, model_key)
