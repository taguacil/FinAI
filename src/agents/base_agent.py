"""
Base agent class for the multi-agent system.

Provides common functionality for all specialist agents.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..portfolio.manager import PortfolioManager
from ..services.market_data_service import MarketDataService
from ..utils.metrics import FinancialMetricsCalculator


class BaseAgent(ABC):
    """Abstract base class for all specialist agents.

    Provides common initialization, agent creation, and invocation logic.
    Subclasses must implement get_system_prompt() and get_agent_name().
    """

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        market_data_service: MarketDataService,
        metrics_calculator: FinancialMetricsCalculator,
        llm: BaseChatModel,
        memory: ConversationBufferMemory,
        tools: List[BaseTool],
    ):
        """Initialize the base agent.

        Args:
            portfolio_manager: PortfolioManager instance
            market_data_service: MarketDataService instance
            metrics_calculator: FinancialMetricsCalculator instance
            llm: Language model instance
            memory: Shared conversation memory
            tools: List of tools available to this agent
        """
        self.portfolio_manager = portfolio_manager
        self.market_data_service = market_data_service
        self.metrics_calculator = metrics_calculator
        self.llm = llm
        self.memory = memory  # Shared across agents
        self.tools = tools
        self.agent_executor = self._create_agent()

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the specialized system prompt for this agent.

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def get_agent_name(self) -> str:
        """Return the agent's display name.

        Returns:
            Agent name string
        """
        pass

    def _create_agent(self) -> AgentExecutor:
        """Create the agent with specialized prompt and tools.

        Returns:
            Configured AgentExecutor instance
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.get_system_prompt()),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt,
        )

        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            max_iterations=5,
        )

    def invoke(self, message: str, context: Optional[str] = None) -> str:
        """Invoke this agent with a message.

        Args:
            message: User message to process
            context: Optional portfolio context to prepend

        Returns:
            Agent response string
        """
        try:
            enhanced_message = message
            if context:
                enhanced_message = f"Context: {context}\n\nUser message: {message}"

            response = self.agent_executor.invoke({"input": enhanced_message})
            return response.get("output", "I couldn't process that request.")

        except Exception as e:
            return f"Error in {self.get_agent_name()}: {str(e)}"

    def update_llm(self, llm: BaseChatModel):
        """Update the LLM and recreate the agent.

        Args:
            llm: New language model instance
        """
        self.llm = llm
        self.agent_executor = self._create_agent()
