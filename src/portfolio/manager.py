"""
Portfolio manager for handling portfolio operations and transactions.
"""

import logging
import uuid
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

from ..data_providers.manager import DataProviderManager
from .models import (
    Currency,
    FinancialInstrument,
    InstrumentType,
    Portfolio,
    PortfolioSnapshot,
    Position,
    Transaction,
    TransactionType,
)
from .storage import FileBasedStorage


class PortfolioManager:
    """Manages portfolio operations, transactions, and data updates."""

    def __init__(
        self,
        storage: Optional[FileBasedStorage] = None,
        data_manager: Optional[DataProviderManager] = None,
    ):
        """Initialize portfolio manager."""
        self.storage = storage or FileBasedStorage()
        self.data_manager = data_manager or DataProviderManager()
        self.current_portfolio: Optional[Portfolio] = None

    def create_portfolio(
        self, name: str, base_currency: Currency = Currency.USD
    ) -> Portfolio:
        """Create a new portfolio."""
        portfolio = Portfolio(
            id=str(uuid.uuid4()),
            name=name,
            base_currency=base_currency,
            created_at=datetime.now(),
        )

        self.storage.save_portfolio(portfolio)
        self.current_portfolio = portfolio
        logging.info(f"Created new portfolio: {name} ({portfolio.id})")
        return portfolio

    def load_portfolio(self, portfolio_id: str) -> Optional[Portfolio]:
        """Load an existing portfolio."""
        portfolio = self.storage.load_portfolio(portfolio_id)
        if portfolio:
            self.current_portfolio = portfolio
            logging.info(f"Loaded portfolio: {portfolio.name} ({portfolio_id})")
        return portfolio

    def list_portfolios(self) -> List[str]:
        """List all available portfolios."""
        return self.storage.list_portfolios()

    def add_transaction(
        self,
        symbol: str,
        transaction_type: TransactionType,
        quantity: Decimal,
        price: Decimal,
        timestamp: Optional[datetime] = None,
        fees: Decimal = Decimal("0"),
        currency: Optional[Currency] = None,
        notes: Optional[str] = None,
        isin: Optional[str] = None,
    ) -> bool:
        """Add a transaction to the current portfolio."""
        if not self.current_portfolio:
            logging.error("No portfolio loaded")
            return False

        # Get instrument info from data providers
        instrument_info = self.data_manager.get_instrument_info(symbol)
        if not instrument_info:
            # Create basic instrument info if not found
            instrument_info_dict = {
                "symbol": symbol.upper(),
                "name": symbol.upper(),
                "instrument_type": InstrumentType.STOCK,  # Default
                "currency": currency or Currency.USD,
                "isin": isin,
            }
        else:
            instrument_info_dict = {
                "symbol": instrument_info.symbol,
                "name": instrument_info.name,
                "instrument_type": instrument_info.instrument_type,
                "currency": instrument_info.currency,
                "exchange": instrument_info.exchange,
                "isin": instrument_info.isin or isin,
            }

        instrument = FinancialInstrument(**instrument_info_dict)

        transaction = Transaction(
            id=str(uuid.uuid4()),
            timestamp=timestamp or datetime.now(),
            instrument=instrument,
            transaction_type=transaction_type,
            quantity=quantity,
            price=price,
            fees=fees,
            currency=currency or instrument.currency,
            notes=notes,
        )

        self.current_portfolio.add_transaction(transaction)
        self.storage.save_portfolio(self.current_portfolio)

        logging.info(
            f"Added {transaction_type} transaction: {quantity} {symbol} @ {price}"
        )
        return True

    def buy_shares(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        timestamp: Optional[datetime] = None,
        fees: Decimal = Decimal("0"),
        notes: Optional[str] = None,
    ) -> bool:
        """Convenience method to buy shares."""
        return self.add_transaction(
            symbol=symbol,
            transaction_type=TransactionType.BUY,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            fees=fees,
            notes=notes,
        )

    def sell_shares(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        timestamp: Optional[datetime] = None,
        fees: Decimal = Decimal("0"),
        notes: Optional[str] = None,
    ) -> bool:
        """Convenience method to sell shares."""
        return self.add_transaction(
            symbol=symbol,
            transaction_type=TransactionType.SELL,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            fees=fees,
            notes=notes,
        )

    def add_dividend(
        self,
        symbol: str,
        amount: Decimal,
        timestamp: Optional[datetime] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Add a dividend payment."""
        return self.add_transaction(
            symbol=symbol,
            transaction_type=TransactionType.DIVIDEND,
            quantity=Decimal("1"),  # Dividend as single unit
            price=amount,
            timestamp=timestamp,
            notes=notes,
        )

    def deposit_cash(
        self,
        amount: Decimal,
        currency: Currency = Currency.USD,
        timestamp: Optional[datetime] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Deposit cash into portfolio."""
        return self.add_transaction(
            symbol="CASH",
            transaction_type=TransactionType.DEPOSIT,
            quantity=Decimal("1"),
            price=amount,
            timestamp=timestamp,
            currency=currency,
            notes=notes,
        )

    def withdraw_cash(
        self,
        amount: Decimal,
        currency: Currency = Currency.USD,
        timestamp: Optional[datetime] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Withdraw cash from portfolio."""
        return self.add_transaction(
            symbol="CASH",
            transaction_type=TransactionType.WITHDRAWAL,
            quantity=Decimal("1"),
            price=amount,
            timestamp=timestamp,
            currency=currency,
            notes=notes,
        )

    def update_current_prices(self) -> Dict[str, bool]:
        """Update current prices for all positions."""
        if not self.current_portfolio:
            return {}

        results = {}
        symbols = list(self.current_portfolio.positions.keys())

        if not symbols:
            return {}

        # Get current prices for all symbols
        prices = self.data_manager.get_multiple_current_prices(symbols)

        for symbol, price in prices.items():
            if symbol in self.current_portfolio.positions and price is not None:
                self.current_portfolio.positions[symbol].current_price = price
                self.current_portfolio.positions[symbol].last_updated = datetime.now()
                results[symbol] = True
                logging.debug(f"Updated price for {symbol}: {price}")
            else:
                results[symbol] = False
                logging.warning(f"Could not update price for {symbol}")

        # Save updated portfolio
        self.storage.save_portfolio(self.current_portfolio)
        return results

    def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value in base currency."""
        if not self.current_portfolio:
            return Decimal("0")

        # Get exchange rates for currency conversion
        exchange_rates = {}
        for currency in [
            Currency.EUR,
            Currency.GBP,
            Currency.JPY,
            Currency.CHF,
            Currency.CAD,
        ]:
            if currency != self.current_portfolio.base_currency:
                rate = self.data_manager.get_exchange_rate(
                    currency, self.current_portfolio.base_currency
                )
                if rate:
                    exchange_rates[currency.value] = rate

        return self.current_portfolio.get_total_value(exchange_rates)

    def create_snapshot(
        self, snapshot_date: Optional[date] = None
    ) -> PortfolioSnapshot:
        """Create a comprehensive portfolio snapshot for the given date."""
        if not self.current_portfolio:
            raise ValueError("No portfolio loaded")

        snapshot_date = snapshot_date or date.today()

        # Update current prices if taking snapshot for today
        if snapshot_date == date.today():
            self.update_current_prices()

        total_value = self.get_portfolio_value()

        # Calculate cash balance in base currency
        cash_balance = Decimal("0")
        for currency, amount in self.current_portfolio.cash_balances.items():
            if currency == self.current_portfolio.base_currency.value:
                cash_balance += amount
            else:
                rate = self.data_manager.get_exchange_rate(
                    Currency(currency), self.current_portfolio.base_currency
                )
                if rate:
                    cash_balance += amount * rate

        positions_value = total_value - cash_balance

        # Calculate performance metrics
        total_cost_basis = Decimal("0")
        total_unrealized_pnl = Decimal("0")

        for position in self.current_portfolio.positions.values():
            total_cost_basis += position.cost_basis
            if position.unrealized_pnl is not None:
                total_unrealized_pnl += position.unrealized_pnl

        total_unrealized_pnl_percent = (
            (total_unrealized_pnl / total_cost_basis * 100)
            if total_cost_basis > 0
            else Decimal("0")
        )

        # Create deep copy of positions for snapshot
        snapshot_positions = {}
        for symbol, position in self.current_portfolio.positions.items():
            snapshot_positions[symbol] = Position(
                instrument=position.instrument,
                quantity=position.quantity,
                average_cost=position.average_cost,
                current_price=position.current_price,
                last_updated=position.last_updated,
            )

        # Create deep copy of cash balances
        snapshot_cash_balances = {}
        for currency, amount in self.current_portfolio.cash_balances.items():
            snapshot_cash_balances[currency] = amount

        snapshot = PortfolioSnapshot(
            date=snapshot_date,
            total_value=total_value,
            cash_balance=cash_balance,
            positions_value=positions_value,
            base_currency=self.current_portfolio.base_currency,
            positions=snapshot_positions,
            cash_balances=snapshot_cash_balances,
            total_cost_basis=total_cost_basis,
            total_unrealized_pnl=total_unrealized_pnl,
            total_unrealized_pnl_percent=total_unrealized_pnl_percent,
        )

        # Save snapshot
        self.storage.save_snapshot(self.current_portfolio.id, snapshot)
        return snapshot

    def create_snapshots_since_last(
        self, end_date: Optional[date] = None
    ) -> List[PortfolioSnapshot]:
        """Create snapshots for all dates since the last snapshot up to the end_date."""
        if not self.current_portfolio:
            raise ValueError("No portfolio loaded")

        end_date = end_date or date.today()

        # Get the last snapshot date
        last_snapshot = self.storage.get_latest_snapshot(self.current_portfolio.id)
        start_date = None

        if last_snapshot:
            start_date = last_snapshot.date + timedelta(days=1)
        else:
            # If no previous snapshots, start from portfolio creation date
            start_date = self.current_portfolio.created_at.date()

        if start_date > end_date:
            return []  # No dates to snapshot

        snapshots = []
        current_date = start_date

        while current_date <= end_date:
            try:
                # Create a temporary portfolio state for this date
                temp_portfolio = self._get_portfolio_state_for_date(current_date)

                # Temporarily set current portfolio to the historical state
                original_portfolio = self.current_portfolio
                self.current_portfolio = temp_portfolio

                # Create snapshot for this date
                snapshot = self.create_snapshot(current_date)
                snapshots.append(snapshot)

                # Restore original portfolio
                self.current_portfolio = original_portfolio

            except Exception as e:
                logging.warning(f"Failed to create snapshot for {current_date}: {e}")

            current_date += timedelta(days=1)

        return snapshots

    def create_snapshots_for_range(
        self, start_date: date, end_date: date
    ) -> List[PortfolioSnapshot]:
        """Create snapshots for a specific date range."""
        if not self.current_portfolio:
            raise ValueError("No portfolio loaded")

        if start_date > end_date:
            raise ValueError("Start date must be before or equal to end date")

        snapshots = []
        current_date = start_date

        while current_date <= end_date:
            try:
                # Create a temporary portfolio state for this date
                temp_portfolio = self._get_portfolio_state_for_date(current_date)

                # Temporarily set current portfolio to the historical state
                original_portfolio = self.current_portfolio
                self.current_portfolio = temp_portfolio

                # Create snapshot for this date
                snapshot = self.create_snapshot(current_date)
                snapshots.append(snapshot)

                # Restore original portfolio
                self.current_portfolio = original_portfolio

            except Exception as e:
                logging.warning(f"Failed to create snapshot for {current_date}: {e}")

            current_date += timedelta(days=1)

        return snapshots

    def get_snapshot_summary(self, portfolio_id: Optional[str] = None) -> Dict:
        """Get a summary of all snapshots for a portfolio."""
        if not portfolio_id:
            if not self.current_portfolio:
                raise ValueError("No portfolio loaded")
            portfolio_id = self.current_portfolio.id

        snapshots = self.storage.load_snapshots(portfolio_id)

        if not snapshots:
            return {
                "total_snapshots": 0,
                "date_range": None,
                "latest_snapshot": None,
                "earliest_snapshot": None,
            }

        return {
            "total_snapshots": len(snapshots),
            "date_range": {
                "start": snapshots[0].date.isoformat(),
                "end": snapshots[-1].date.isoformat(),
            },
            "latest_snapshot": {
                "date": snapshots[-1].date.isoformat(),
                "total_value": float(snapshots[-1].total_value),
                "total_pnl": float(snapshots[-1].total_unrealized_pnl),
                "total_pnl_percent": float(snapshots[-1].total_unrealized_pnl_percent),
            },
            "earliest_snapshot": {
                "date": snapshots[0].date.isoformat(),
                "total_value": float(snapshots[0].total_value),
                "total_pnl": float(snapshots[0].total_unrealized_pnl),
                "total_pnl_percent": float(snapshots[0].total_unrealized_pnl_percent),
            },
        }

    def _get_portfolio_state_for_date(self, target_date: date) -> Portfolio:
        """Get portfolio state as it was on a specific date."""
        if not self.current_portfolio:
            raise ValueError("No portfolio loaded")

        # Create a copy of the current portfolio
        portfolio_copy = Portfolio(
            id=self.current_portfolio.id,
            name=self.current_portfolio.name,
            base_currency=self.current_portfolio.base_currency,
            created_at=self.current_portfolio.created_at,
            transactions=[],
            positions={},
            cash_balances={},
        )

        # Replay transactions up to the target date
        for transaction in sorted(
            self.current_portfolio.transactions, key=lambda t: t.timestamp
        ):
            if transaction.timestamp.date() <= target_date:
                portfolio_copy.add_transaction(transaction)

        # Update prices for the target date (if it's today, use current prices)
        if target_date == date.today():
            self._update_portfolio_prices(portfolio_copy)
        else:
            # For historical dates, we would need historical price data
            # For now, we'll use the last known prices
            self._update_portfolio_prices(portfolio_copy)

        return portfolio_copy

    def _update_portfolio_prices(self, portfolio: Portfolio) -> None:
        """Update current prices for all positions in a portfolio."""
        for position in portfolio.positions.values():
            try:
                price = self.data_manager.get_current_price(position.instrument.symbol)
                if price:
                    position.current_price = price
                    position.last_updated = datetime.now()
            except Exception as e:
                logging.warning(
                    f"Failed to update price for {position.instrument.symbol}: {e}"
                )

    def get_position_summary(self) -> List[Dict]:
        """Get summary of all positions."""
        if not self.current_portfolio:
            return []

        summary = []
        for symbol, position in self.current_portfolio.positions.items():
            summary.append(
                {
                    "symbol": symbol,
                    "name": position.instrument.name,
                    "quantity": position.quantity,
                    "average_cost": position.average_cost,
                    "current_price": position.current_price,
                    "market_value": position.market_value,
                    "cost_basis": position.cost_basis,
                    "unrealized_pnl": position.unrealized_pnl,
                    "unrealized_pnl_percent": position.unrealized_pnl_percent,
                    "currency": position.instrument.currency.value,
                    "instrument_type": position.instrument.instrument_type.value,
                }
            )

        return summary

    def get_transaction_history(self, days: Optional[int] = None) -> List[Dict]:
        """Get transaction history, optionally filtered by days."""
        if not self.current_portfolio:
            return []

        transactions = self.current_portfolio.transactions

        if days:
            cutoff_date = datetime.now() - timedelta(days=days)
            transactions = [txn for txn in transactions if txn.timestamp >= cutoff_date]

        return [
            {
                "id": txn.id,
                "timestamp": txn.timestamp,
                "symbol": txn.instrument.symbol,
                "name": txn.instrument.name,
                "type": txn.transaction_type.value,
                "quantity": txn.quantity,
                "price": txn.price,
                "fees": txn.fees,
                "total_value": txn.total_value,
                "currency": txn.currency.value,
                "notes": txn.notes,
            }
            for txn in sorted(transactions, key=lambda x: x.timestamp, reverse=True)
        ]

    def get_performance_metrics(self, days: int = 365) -> Dict:
        """Get basic performance metrics."""
        if not self.current_portfolio:
            return {}

        # Get snapshots for the period
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        snapshots = self.storage.load_snapshots(
            self.current_portfolio.id, start_date, end_date
        )

        if len(snapshots) < 2:
            return {"error": "Insufficient historical data"}

        first_value = snapshots[0].total_value
        last_value = snapshots[-1].total_value

        total_return = (
            ((last_value - first_value) / first_value * 100)
            if first_value > 0
            else Decimal("0")
        )

        # Calculate daily returns for volatility
        daily_returns = []
        for i in range(1, len(snapshots)):
            prev_value = snapshots[i - 1].total_value
            curr_value = snapshots[i].total_value
            if prev_value > 0:
                daily_return = (curr_value - prev_value) / prev_value
                daily_returns.append(float(daily_return))

        volatility = Decimal("0")
        if len(daily_returns) > 1:
            import statistics

            volatility = Decimal(
                str(statistics.stdev(daily_returns) * (252**0.5) * 100)
            )  # Annualized

        return {
            "total_return_percent": total_return,
            "annualized_volatility_percent": volatility,
            "current_value": last_value,
            "period_start_value": first_value,
            "days_analyzed": len(snapshots),
        }
