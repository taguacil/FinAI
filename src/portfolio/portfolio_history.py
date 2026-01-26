"""
Portfolio history calculator with on-demand value computation.

This module calculates portfolio state at any point in time from transactions + prices,
replacing the need for pre-computed snapshots.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from .market_data_store import MarketDataStore
from .models import Currency, Portfolio, Position, Transaction, TransactionType


@dataclass
class PositionState:
    """Lightweight position state without price data.

    This represents the position quantity and cost basis at a point in time,
    derived purely from transactions. Price data is looked up separately.
    """

    symbol: str
    quantity: Decimal
    average_cost: Decimal
    currency: Currency
    instrument_type: str = "stock"
    instrument_name: str = ""
    data_provider_symbol: Optional[str] = None
    isin: Optional[str] = None

    @property
    def cost_basis(self) -> Decimal:
        """Total cost basis of the position."""
        return self.quantity * self.average_cost


@dataclass
class CashState:
    """Cash balances at a point in time."""

    balances: Dict[Currency, Decimal] = field(default_factory=dict)

    def get_balance(self, currency: Currency) -> Decimal:
        """Get balance for a specific currency."""
        return self.balances.get(currency, Decimal("0"))

    def total_in_currency(
        self, target_currency: Currency, fx_rate_func: Callable[[Currency, Currency], Optional[Decimal]]
    ) -> Decimal:
        """Get total cash in target currency."""
        total = Decimal("0")
        for currency, amount in self.balances.items():
            if currency == target_currency:
                total += amount
            else:
                rate = fx_rate_func(currency, target_currency)
                if rate:
                    total += amount * rate
        return total


@dataclass
class PortfolioState:
    """Complete portfolio state at a point in time."""

    as_of_date: date
    positions: Dict[str, PositionState]
    cash: CashState
    base_currency: Currency

    def get_position(self, symbol: str) -> Optional[PositionState]:
        """Get position for a symbol."""
        return self.positions.get(symbol.upper())


class PortfolioHistory:
    """Calculates portfolio state at any point in time from transactions + prices.

    Uses memoization for position state at transaction dates (sparse) to avoid
    replaying all transactions for each query.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        market_data: MarketDataStore,
        fx_rate_func: Callable[[Currency, Currency], Optional[Decimal]],
        fx_rate_func_with_date: Optional[Callable[[Currency, Currency, date], Optional[Decimal]]] = None,
    ):
        """Initialize the portfolio history calculator.

        Args:
            portfolio: The portfolio to analyze
            market_data: MarketDataStore for price lookups
            fx_rate_func: Function to get FX rates (from_currency, to_currency) -> rate (for current date)
            fx_rate_func_with_date: Function to get historical FX rates (from_currency, to_currency, date) -> rate
        """
        self.portfolio = portfolio
        self.market_data = market_data
        self._fx_rate_func = fx_rate_func
        self._fx_rate_func_with_date = fx_rate_func_with_date

        # Cache of position states at transaction dates
        # Key: date -> PortfolioState
        self._state_cache: Dict[date, PortfolioState] = {}

        # Sorted list of transaction dates for binary search
        self._transaction_dates: List[date] = sorted(set(
            t.timestamp.date() for t in portfolio.transactions
        ))

    def _get_fx_rate(self, from_currency: Currency, to_currency: Currency, as_of: Optional[date] = None) -> Optional[Decimal]:
        """Get FX rate with fallback to 1.0 for same currency.

        Args:
            from_currency: Source currency
            to_currency: Target currency
            as_of: Optional date for historical rate lookup
        """
        if from_currency == to_currency:
            return Decimal("1")

        # Use date-aware function if available and date is specified
        if as_of and self._fx_rate_func_with_date:
            return self._fx_rate_func_with_date(from_currency, to_currency, as_of)

        return self._fx_rate_func(from_currency, to_currency)

    def _get_price_for_position(
        self, pos: PositionState, target_date: date
    ) -> Optional[Decimal]:
        """Get price for a position, trying all available identifiers.

        Lookup order:
        1. data_provider_symbol (if set)
        2. symbol (portfolio symbol)
        3. isin (if set)

        This ensures we find prices regardless of which identifier was used
        when storing the price data.

        Args:
            pos: PositionState with symbol, data_provider_symbol, and isin
            target_date: Date to get price for

        Returns:
            Price if found, None otherwise
        """
        # Try data_provider_symbol first (if set)
        if pos.data_provider_symbol:
            price = self.market_data.get_price_with_fallback(
                pos.data_provider_symbol, target_date
            )
            if price is not None:
                return price

        # Try portfolio symbol
        price = self.market_data.get_price_with_fallback(pos.symbol, target_date)
        if price is not None:
            return price

        # Try ISIN as fallback (if set)
        if pos.isin:
            price = self.market_data.get_price_with_fallback(pos.isin, target_date)
            if price is not None:
                return price

        return None

    def _replay_transactions_to_date(self, target_date: date) -> PortfolioState:
        """Replay transactions up to target_date to compute position state.

        Args:
            target_date: Date to compute state for

        Returns:
            PortfolioState at the target date
        """
        # Check cache first
        if target_date in self._state_cache:
            return self._state_cache[target_date]

        # Find nearest cached state before target_date
        nearest_cached_date: Optional[date] = None
        for cached_date in sorted(self._state_cache.keys(), reverse=True):
            if cached_date < target_date:
                nearest_cached_date = cached_date
                break

        # Start from cached state or empty
        if nearest_cached_date:
            base_state = self._state_cache[nearest_cached_date]
            positions = {s: PositionState(
                symbol=p.symbol,
                quantity=p.quantity,
                average_cost=p.average_cost,
                currency=p.currency,
                instrument_type=p.instrument_type,
                instrument_name=p.instrument_name,
                data_provider_symbol=p.data_provider_symbol,
                isin=p.isin,
            ) for s, p in base_state.positions.items()}
            cash_balances = dict(base_state.cash.balances)
            start_date = nearest_cached_date
        else:
            positions: Dict[str, PositionState] = {}
            cash_balances: Dict[Currency, Decimal] = {}
            start_date = None

        # Replay transactions from start_date to target_date
        for txn in sorted(self.portfolio.transactions, key=lambda t: t.timestamp):
            txn_date = txn.timestamp.date()

            # Skip if before start date (already in cached state)
            if start_date and txn_date <= start_date:
                continue

            # Stop if after target date
            if txn_date > target_date:
                break

            # Apply transaction
            self._apply_transaction(txn, positions, cash_balances)

        # Create state
        state = PortfolioState(
            as_of_date=target_date,
            positions=positions,
            cash=CashState(balances=cash_balances),
            base_currency=self.portfolio.base_currency,
        )

        # Cache the state
        self._state_cache[target_date] = state

        return state

    def _apply_transaction(
        self,
        txn: Transaction,
        positions: Dict[str, PositionState],
        cash_balances: Dict[Currency, Decimal],
    ) -> None:
        """Apply a transaction to position and cash state."""
        symbol = txn.instrument.symbol
        currency = txn.currency

        # Ensure cash balance exists
        if currency not in cash_balances:
            cash_balances[currency] = Decimal("0")

        if txn.transaction_type == TransactionType.BUY:
            if symbol in positions:
                # Update existing position
                pos = positions[symbol]
                total_cost = (pos.quantity * pos.average_cost) + txn.total_value
                total_quantity = pos.quantity + txn.quantity
                new_avg_cost = total_cost / total_quantity if total_quantity > 0 else Decimal("0")

                positions[symbol] = PositionState(
                    symbol=symbol,
                    quantity=total_quantity,
                    average_cost=new_avg_cost,
                    currency=pos.currency,
                    instrument_type=pos.instrument_type,
                    instrument_name=pos.instrument_name,
                    data_provider_symbol=pos.data_provider_symbol,
                    isin=pos.isin,
                )
            else:
                # Create new position
                positions[symbol] = PositionState(
                    symbol=symbol,
                    quantity=txn.quantity,
                    average_cost=txn.price,
                    currency=txn.instrument.currency,
                    instrument_type=txn.instrument.instrument_type.value,
                    instrument_name=txn.instrument.name,
                    data_provider_symbol=txn.instrument.data_provider_symbol,
                    isin=txn.instrument.isin,
                )

            # Decrease cash
            cash_balances[currency] -= txn.total_value

        elif txn.transaction_type == TransactionType.SELL:
            if symbol in positions:
                pos = positions[symbol]
                new_quantity = pos.quantity - txn.quantity
                if new_quantity <= 0:
                    del positions[symbol]
                else:
                    positions[symbol] = PositionState(
                        symbol=symbol,
                        quantity=new_quantity,
                        average_cost=pos.average_cost,  # Keep same avg cost
                        currency=pos.currency,
                        instrument_type=pos.instrument_type,
                        instrument_name=pos.instrument_name,
                        data_provider_symbol=pos.data_provider_symbol,
                        isin=pos.isin,
                    )

            # Increase cash from sale
            cash_balances[currency] += txn.total_value

        elif txn.transaction_type == TransactionType.DEPOSIT:
            cash_balances[currency] += txn.total_value

        elif txn.transaction_type == TransactionType.WITHDRAWAL:
            cash_balances[currency] -= txn.total_value

        elif txn.transaction_type == TransactionType.FEES:
            cash_balances[currency] -= txn.total_value

        elif txn.transaction_type in [TransactionType.DIVIDEND, TransactionType.INTEREST]:
            cash_balances[currency] += txn.total_value

    def get_positions_at_date(self, target_date: date) -> Dict[str, PositionState]:
        """Get position states at a specific date.

        Args:
            target_date: Date to get positions for

        Returns:
            Dict mapping symbol to PositionState
        """
        state = self._replay_transactions_to_date(target_date)
        return state.positions

    def get_cash_at_date(self, target_date: date) -> Dict[Currency, Decimal]:
        """Get cash balances at a specific date.

        Args:
            target_date: Date to get cash for

        Returns:
            Dict mapping currency to balance
        """
        state = self._replay_transactions_to_date(target_date)
        return state.cash.balances

    def get_value_at_date(self, target_date: date) -> Decimal:
        """Get total portfolio value at a specific date.

        Args:
            target_date: Date to get value for

        Returns:
            Total portfolio value in base currency
        """
        state = self._replay_transactions_to_date(target_date)
        base = self.portfolio.base_currency

        total = Decimal("0")

        # Create a date-aware FX rate function for cash conversion
        def fx_rate_for_date(from_curr: Currency, to_curr: Currency) -> Optional[Decimal]:
            return self._get_fx_rate(from_curr, to_curr, as_of=target_date)

        # Add cash
        total += state.cash.total_in_currency(base, fx_rate_for_date)

        # Add position values
        for symbol, pos in state.positions.items():
            price = self._get_price_for_position(pos, target_date)

            if price is not None and pos.quantity > 0:
                position_value = pos.quantity * price
                if pos.currency == base:
                    total += position_value
                else:
                    rate = self._get_fx_rate(pos.currency, base, as_of=target_date)
                    if rate:
                        total += position_value * rate

        return total

    def get_value_history(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Get portfolio value history as a DataFrame.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with columns: date, total_value, cash_value, positions_value
        """
        data: List[Dict] = []
        current = start_date

        while current <= end_date:
            state = self._replay_transactions_to_date(current)
            base = self.portfolio.base_currency

            # Create date-aware FX rate function for this date
            def fx_rate_for_current(from_curr: Currency, to_curr: Currency, dt: date = current) -> Optional[Decimal]:
                return self._get_fx_rate(from_curr, to_curr, as_of=dt)

            cash_value = state.cash.total_in_currency(base, fx_rate_for_current)
            positions_value = Decimal("0")

            for symbol, pos in state.positions.items():
                price = self._get_price_for_position(pos, current)

                if price is not None and pos.quantity > 0:
                    pv = pos.quantity * price
                    if pos.currency == base:
                        positions_value += pv
                    else:
                        rate = self._get_fx_rate(pos.currency, base, as_of=current)
                        if rate:
                            positions_value += pv * rate

            total_value = cash_value + positions_value

            data.append({
                "date": current,
                "total_value": float(total_value),
                "cash_value": float(cash_value),
                "positions_value": float(positions_value),
            })

            current += timedelta(days=1)

        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df

    def get_positions_history(
        self, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Get position value history for all symbols.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with date index and one column per symbol
        """
        # Collect all symbols that ever existed
        all_symbols: set = set()
        for txn in self.portfolio.transactions:
            if txn.timestamp.date() <= end_date:
                if txn.instrument.symbol != "CASH":
                    all_symbols.add(txn.instrument.symbol)

        if not all_symbols:
            return pd.DataFrame()

        # Build data for each date
        data: List[Dict] = []
        current = start_date

        while current <= end_date:
            positions = self.get_positions_at_date(current)

            row = {"date": current}
            for symbol in all_symbols:
                if symbol in positions:
                    pos = positions[symbol]
                    price = self._get_price_for_position(pos, current)

                    if price is not None:
                        row[symbol] = float(pos.quantity * price)
                    else:
                        row[symbol] = None
                else:
                    row[symbol] = 0.0

            data.append(row)
            current += timedelta(days=1)

        df = pd.DataFrame(data)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df

    def get_daily_returns(
        self, start_date: date, end_date: date
    ) -> pd.Series:
        """Get daily return series.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Series of daily returns
        """
        df = self.get_value_history(start_date, end_date)
        if df.empty or len(df) < 2:
            return pd.Series(dtype=float)

        returns = df["total_value"].pct_change().dropna()
        return returns

    def calculate_twr(
        self, start_date: date, end_date: date
    ) -> Optional[Decimal]:
        """Calculate Time-Weighted Return (TWR).

        TWR measures investment performance independent of cash flows.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            TWR as a decimal (e.g., 0.12 for 12%) or None if insufficient data
        """
        # Get values and cash flows
        df = self.get_value_history(start_date, end_date)
        if df.empty or len(df) < 2:
            return None

        # Get external cash flows
        cash_flow_dates: Dict[date, Decimal] = {}
        for txn in self.portfolio.transactions:
            txn_date = txn.timestamp.date()
            if not (start_date <= txn_date <= end_date):
                continue
            if txn.transaction_type == TransactionType.DEPOSIT:
                if txn_date not in cash_flow_dates:
                    cash_flow_dates[txn_date] = Decimal("0")
                cash_flow_dates[txn_date] += txn.total_value
            elif txn.transaction_type == TransactionType.WITHDRAWAL:
                if txn_date not in cash_flow_dates:
                    cash_flow_dates[txn_date] = Decimal("0")
                cash_flow_dates[txn_date] -= txn.total_value

        # Calculate sub-period returns
        # For simplicity, we use daily returns and compound them
        returns = df["total_value"].pct_change().dropna()

        if len(returns) == 0:
            return None

        # Compound returns: (1 + r1) * (1 + r2) * ... - 1
        cumulative = 1.0
        for r in returns:
            cumulative *= (1 + r)

        twr = cumulative - 1
        return Decimal(str(twr))

    def get_unrealized_pnl_at_date(self, target_date: date) -> Decimal:
        """Get total unrealized P&L at a specific date.

        Args:
            target_date: Date to calculate P&L for

        Returns:
            Total unrealized P&L in base currency
        """
        positions = self.get_positions_at_date(target_date)
        base = self.portfolio.base_currency

        total_pnl = Decimal("0")

        for symbol, pos in positions.items():
            if pos.quantity <= 0:
                continue

            price = self._get_price_for_position(pos, target_date)

            if price is not None:
                market_value = pos.quantity * price
                cost_basis = pos.cost_basis
                pnl = market_value - cost_basis

                if pos.currency == base:
                    total_pnl += pnl
                else:
                    rate = self._get_fx_rate(pos.currency, base)
                    if rate:
                        total_pnl += pnl * rate

        return total_pnl

    def clear_cache(self) -> None:
        """Clear the internal state cache."""
        self._state_cache.clear()

    def invalidate_from_date(self, from_date: date) -> None:
        """Invalidate cached states from a specific date onwards.

        Useful when transactions are added or modified.

        Args:
            from_date: Date from which to invalidate cache
        """
        dates_to_remove = [d for d in self._state_cache.keys() if d >= from_date]
        for d in dates_to_remove:
            del self._state_cache[d]
