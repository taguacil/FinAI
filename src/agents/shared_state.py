"""
Shared state management for multi-agent system.

Provides shared conversation memory and portfolio context across all agents.
"""

from datetime import datetime
from typing import Optional

from langchain.memory import ConversationBufferMemory

from ..portfolio.manager import PortfolioManager


class SharedAgentState:
    """Manages shared state across all agents in a session.

    This class maintains:
    - Shared conversation memory (single instance across all agents)
    - Portfolio context cache with automatic refresh
    """

    # Context refresh interval in seconds
    CONTEXT_REFRESH_INTERVAL = 30

    def __init__(self):
        """Initialize shared state."""
        # Shared conversation memory - single instance for all agents
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

        # Portfolio context cache
        self._portfolio_context: Optional[str] = None
        self._context_timestamp: Optional[datetime] = None

    def get_portfolio_context(self, portfolio_manager: PortfolioManager) -> str:
        """Get cached portfolio context, refreshing if stale.

        Args:
            portfolio_manager: PortfolioManager instance

        Returns:
            Portfolio context string for agent prompts
        """
        now = datetime.now()

        # Refresh if cache is stale or empty
        if (
            self._portfolio_context is None
            or self._context_timestamp is None
            or (now - self._context_timestamp).seconds > self.CONTEXT_REFRESH_INTERVAL
        ):
            self._portfolio_context = self._build_context(portfolio_manager)
            self._context_timestamp = now

        return self._portfolio_context

    def invalidate_context(self):
        """Invalidate the portfolio context cache.

        Call this after transactions are added/modified/deleted.
        """
        self._portfolio_context = None
        self._context_timestamp = None

    def _build_context(self, portfolio_manager: PortfolioManager) -> str:
        """Build portfolio context string for agents.

        Args:
            portfolio_manager: PortfolioManager instance

        Returns:
            Portfolio context string
        """
        try:
            if not portfolio_manager.current_portfolio:
                return "No portfolio is currently loaded."

            portfolio = portfolio_manager.current_portfolio
            total_value = portfolio_manager.get_portfolio_value()
            positions_count = len(portfolio.positions)

            # Get cash balances summary
            cash_summary = ""
            if portfolio.cash_balances:
                balances = [
                    f"{currency.value}: {amount:,.2f}"
                    for currency, amount in portfolio.cash_balances.items()
                ]
                cash_summary = f" Cash: {', '.join(balances)}."

            return (
                f"Portfolio '{portfolio.name}' loaded with "
                f"{total_value:,.2f} {portfolio.base_currency.value} total value "
                f"across {positions_count} positions.{cash_summary}"
            )

        except Exception:
            return "Portfolio context unavailable."

    def clear(self):
        """Clear all shared state."""
        self.memory.clear()
        self._portfolio_context = None
        self._context_timestamp = None
