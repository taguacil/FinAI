"""
Portfolio analyzer for calculating metrics.

This class provides calculation methods for portfolio analysis.
Uses PortfolioHistory for value calculations and MarketDataStore for prices.
"""

import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import pandas as pd

from ..data_providers.manager import DataProviderManager
from .models import Currency, Portfolio, Position, TransactionType
from .storage import FileBasedStorage

if TYPE_CHECKING:
    from ..services.market_data_service import MarketDataService
    from .portfolio_history import PortfolioHistory


class PortfolioAnalyzer:
    """Pure calculation engine - no data fetching, only calculations.

    Supports both DataProviderManager (legacy) and MarketDataService (new).
    """

    def __init__(
        self,
        data_manager: Union[DataProviderManager, "MarketDataService"],
        storage: FileBasedStorage,
    ):
        """Initialize the analyzer.

        Args:
            data_manager: Either DataProviderManager or MarketDataService
            storage: FileBasedStorage instance
        """
        self._data_manager = data_manager
        self.storage = storage

    @property
    def data_manager(self) -> DataProviderManager:
        """Get the underlying DataProviderManager for compatibility.

        If a MarketDataService was provided, returns its internal data_manager.
        """
        # Check if it's a MarketDataService by looking for data_manager attribute
        if hasattr(self._data_manager, "data_manager"):
            return self._data_manager.data_manager
        return self._data_manager

    def get_performance_metrics(
        self,
        portfolio: Portfolio,
        days: int = 365,
        portfolio_history: Optional["PortfolioHistory"] = None,
    ) -> Dict:
        """Calculate performance metrics using PortfolioHistory.

        Args:
            portfolio: The portfolio to analyze
            days: Number of days to analyze
            portfolio_history: PortfolioHistory instance for value calculations

        Returns:
            Dict with performance metrics
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        if not portfolio_history:
            return {"error": "PortfolioHistory required for metrics calculation"}

        df = portfolio_history.get_value_history(start_date, end_date)

        if df.empty or len(df) < 2:
            return {"error": "Insufficient data - update market data first"}

        first_value = Decimal(str(df["total_value"].iloc[0]))
        last_value = Decimal(str(df["total_value"].iloc[-1]))

        total_return = (
            ((last_value - first_value) / first_value * 100)
            if first_value > 0
            else Decimal("0")
        )

        # Calculate daily returns using pandas
        daily_returns = df["total_value"].pct_change().dropna()

        volatility = Decimal("0")
        if len(daily_returns) > 1:
            # Annualized volatility
            vol = daily_returns.std() * (252**0.5) * 100
            volatility = Decimal(str(vol))

        return {
            "total_return_percent": total_return,
            "annualized_volatility_percent": volatility,
            "current_value": last_value,
            "period_start_value": first_value,
            "days_analyzed": len(df),
        }

    def get_external_cash_flows_by_day(
        self, portfolio: Portfolio, start_date: date, end_date: date
    ) -> Dict[date, Decimal]:
        """Calculate daily cash flows in base currency."""
        flows: Dict[date, Decimal] = {}
        base = portfolio.base_currency

        for txn in portfolio.transactions:
            txn_date = txn.timestamp.date()
            if not (start_date <= txn_date <= end_date):
                continue
            if txn.transaction_type not in [TransactionType.DEPOSIT, TransactionType.WITHDRAWAL]:
                continue

            amount = txn.total_value
            if txn.currency != base:
                rate = self.data_manager.get_historical_fx_rate_on(txn_date, txn.currency, base)
                amount = amount * rate if rate else amount

            if txn.transaction_type == TransactionType.WITHDRAWAL:
                amount = -amount

            flows[txn_date] = flows.get(txn_date, Decimal("0")) + amount

        return flows

    # ==== PRIVATE HELPERS ====

    def _calculate_portfolio_value(self, portfolio: Portfolio) -> Decimal:
        """Total value in base currency."""
        total = Decimal("0")

        # Cash
        for currency, amount in portfolio.cash_balances.items():
            if currency == portfolio.base_currency:
                total += amount
            else:
                rate = self.data_manager.get_exchange_rate(currency, portfolio.base_currency)
                if rate:
                    total += amount * rate

        # Positions
        for pos in portfolio.positions.values():
            if pos.market_value:
                if pos.instrument.currency == portfolio.base_currency:
                    total += pos.market_value
                else:
                    rate = self.data_manager.get_exchange_rate(
                        pos.instrument.currency, portfolio.base_currency
                    )
                    if rate:
                        total += pos.market_value * rate

        return total

    def _calculate_cash_balance(self, portfolio: Portfolio) -> Decimal:
        """Cash balance in base currency."""
        cash = Decimal("0")
        for currency, amount in portfolio.cash_balances.items():
            if currency == portfolio.base_currency:
                cash += amount
            else:
                rate = self.data_manager.get_exchange_rate(currency, portfolio.base_currency)
                if rate:
                    cash += amount * rate
        return cash
