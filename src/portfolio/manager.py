"""
Portfolio manager for handling portfolio operations and transactions.
"""

import logging
import uuid
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from ..data_providers.manager import DataProviderManager
from .analyzer import PortfolioAnalyzer
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

if TYPE_CHECKING:
    from ..services.market_data_service import MarketDataService


class PortfolioManager:
    """Manages portfolio operations, transactions, and data updates.

    Supports both DataProviderManager (legacy) and MarketDataService (new).
    """

    def __init__(
        self,
        storage: Optional[FileBasedStorage] = None,
        data_manager: Optional[Union[DataProviderManager, "MarketDataService"]] = None,
    ):
        """Initialize portfolio manager.

        Args:
            storage: FileBasedStorage instance
            data_manager: DataProviderManager or MarketDataService instance
        """
        self.storage = storage or FileBasedStorage()
        self._data_manager = data_manager or DataProviderManager()
        self.analyzer = PortfolioAnalyzer(self._data_manager, self.storage)
        self.current_portfolio: Optional[Portfolio] = None

    @property
    def data_manager(self) -> DataProviderManager:
        """Get the underlying DataProviderManager for compatibility.

        If a MarketDataService was provided, returns its internal data_manager.
        """
        if hasattr(self._data_manager, "data_manager"):
            return self._data_manager.data_manager
        return self._data_manager

    @property
    def market_data_service(self) -> Optional["MarketDataService"]:
        """Get the MarketDataService if one was provided."""
        # Import here to avoid circular imports
        from ..services.market_data_service import MarketDataService

        if isinstance(self._data_manager, MarketDataService):
            return self._data_manager
        return None

    def create_portfolio(
        self, name: str, base_currency: Currency = Currency.USD
    ) -> Portfolio:
        """Create a new portfolio."""
        portfolio = Portfolio(
            id=str(uuid.uuid4()),
            name=name,
            base_currency=base_currency,
            created_at=datetime.now(),
            cash_balances={
                Currency.USD: Decimal("0"),
                Currency.GBP: Decimal("0"),
                Currency.EUR: Decimal("0"),
                Currency.CHF: Decimal("0"),
            },
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
            # Normalize any legacy symbols (strip leading '$', uppercase)
            try:
                normalized_positions: Dict[str, Position] = {}
                for sym, pos in list(self.current_portfolio.positions.items()):
                    norm = sym.strip().lstrip("$").upper()
                    if pos.instrument.symbol != norm:
                        pos.instrument.symbol = norm
                    normalized_positions[norm] = pos
                self.current_portfolio.positions = normalized_positions

                for txn in self.current_portfolio.transactions:
                    norm = txn.instrument.symbol.strip().lstrip("$").upper()
                    if txn.instrument.symbol != norm:
                        txn.instrument.symbol = norm

                # Persist normalization changes
                self.storage.save_portfolio(self.current_portfolio)
            except Exception as e:
                logging.warning(f"Failed to normalize symbols on load: {e}")
            logging.info(f"Loaded portfolio: {portfolio.name} ({portfolio_id})")
        return portfolio

    def list_portfolios(self) -> List[str]:
        """List all available portfolios."""
        return self.storage.list_portfolios()

    def delete_portfolio(self, portfolio_id: str, delete_all_data: bool = True) -> dict:
        """Delete a portfolio and all associated data.

        Args:
            portfolio_id: The portfolio ID to delete
            delete_all_data: If True, also deletes snapshots, backups, and exports

        Returns:
            Dictionary with deletion results
        """
        # Clear current portfolio if it's the one being deleted
        if self.current_portfolio and self.current_portfolio.id == portfolio_id:
            self.current_portfolio = None

        result = self.storage.delete_portfolio(portfolio_id, delete_all_data)
        logging.info(f"Deleted portfolio {portfolio_id}: {result}")
        return result

    def add_transaction(
        self,
        symbol: str,
        transaction_type: TransactionType,
        quantity: Decimal,
        price: Decimal,
        timestamp: Optional[datetime] = None,
        currency: Optional[Currency] = None,
        notes: Optional[str] = None,
        isin: Optional[str] = None,
        instrument_type: Optional[str] = None,
    ) -> bool:
        """Add a transaction to the current portfolio."""
        if not self.current_portfolio:
            logging.error("No portfolio loaded")
            return False

        # Enhanced instrument information discovery
        instrument_info_dict = self._discover_instrument_info(
            symbol, isin, currency, notes, instrument_type
        )
        if not instrument_info_dict:
            logging.error("Failed to discover instrument information")
            return False

        instrument = FinancialInstrument(**instrument_info_dict)

        transaction = Transaction(
            id=str(uuid.uuid4()),
            timestamp=timestamp or datetime.now(),
            instrument=instrument,
            transaction_type=transaction_type,
            quantity=quantity,
            price=price,
            currency=currency or instrument.currency,
            notes=notes,
        )

        # Calculate current balance in transaction currency after this transaction
        transaction_currency = currency or instrument.currency
        current_balance = self.current_portfolio.cash_balances.get(
            transaction_currency, Decimal("0")
        )

        # Simulate the transaction effect on cash balance to get the post-transaction balance
        if transaction_type == TransactionType.BUY:
            # Buying decreases cash
            post_transaction_balance = current_balance - transaction.total_value
        elif transaction_type == TransactionType.SELL:
            # Selling increases cash
            post_transaction_balance = current_balance + transaction.total_value
        elif transaction_type == TransactionType.DEPOSIT:
            # Deposit increases cash
            post_transaction_balance = current_balance + transaction.total_value
        elif transaction_type == TransactionType.WITHDRAWAL:
            # Withdrawal decreases cash
            post_transaction_balance = current_balance - transaction.total_value
        elif transaction_type == TransactionType.FEES:
            # Fees decrease cash
            post_transaction_balance = current_balance - transaction.total_value
        elif transaction_type in [TransactionType.DIVIDEND, TransactionType.INTEREST]:
            # Income increases cash
            post_transaction_balance = current_balance + transaction.total_value
        else:
            # For other transaction types, assume no cash impact
            post_transaction_balance = current_balance

        transaction.current_balance = post_transaction_balance

        self.current_portfolio.add_transaction(transaction)
        self.storage.save_portfolio(self.current_portfolio)

        logging.info(
            f"Added {transaction_type} transaction: {quantity} {instrument.symbol} ({instrument.name}) @ {price}"
        )
        return True

    def buy_shares(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        timestamp: Optional[datetime] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Convenience method to buy shares."""
        return self.add_transaction(
            symbol=symbol,
            transaction_type=TransactionType.BUY,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            notes=notes,
        )

    def sell_shares(
        self,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        timestamp: Optional[datetime] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Convenience method to sell shares."""
        return self.add_transaction(
            symbol=symbol,
            transaction_type=TransactionType.SELL,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
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

    def add_fees(
        self,
        amount: Decimal,
        currency: Currency = Currency.USD,
        timestamp: Optional[datetime] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Add a fees transaction."""
        return self.add_transaction(
            symbol="CASH",
            transaction_type=TransactionType.FEES,
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

    def set_position_price(
        self,
        symbol: str,
        price: Decimal,
        target_date: Optional[date] = None,
        update_current: bool = True,
    ) -> bool:
        """Set a custom/manual price for a position on a specific date.

        This is useful when:
        - Price lookup fails and user wants to use purchase price as market price
        - User wants to manually set a custom price for an instrument
        - Correcting historical prices in snapshots

        Args:
            symbol: The instrument symbol
            price: The price to set
            target_date: The date to set the price for (defaults to today)
            update_current: If True and target_date is today, also update current_price
                           in the portfolio's position

        Returns:
            True if successful, False otherwise
        """
        if not self.current_portfolio:
            logging.error("No portfolio loaded")
            return False

        symbol = symbol.upper().strip()
        target_date = target_date or date.today()

        # Validate the symbol exists in portfolio
        if symbol not in self.current_portfolio.positions:
            logging.error(f"Symbol {symbol} not found in portfolio positions")
            return False

        try:
            # Update current portfolio position if target_date is today
            if update_current and target_date == date.today():
                self.current_portfolio.positions[symbol].current_price = price
                self.current_portfolio.positions[symbol].last_updated = datetime.now()
                self.storage.save_portfolio(self.current_portfolio)
                logging.info(f"Updated current price for {symbol} to {price}")

            # Update or create snapshot for the target date
            existing_snapshots = self.storage.load_snapshots(
                self.current_portfolio.id, target_date, target_date
            )

            if existing_snapshots:
                # Update existing snapshot
                snapshot = existing_snapshots[0]
                if symbol in snapshot.positions:
                    snapshot.positions[symbol].current_price = price
                    snapshot.positions[symbol].last_updated = datetime.now()

                    # Recalculate snapshot totals
                    self._recalculate_snapshot_totals(snapshot)

                    # Save updated snapshot
                    self.storage.save_snapshot(self.current_portfolio.id, snapshot)
                    logging.info(
                        f"Updated price for {symbol} to {price} in snapshot for {target_date}"
                    )
                else:
                    logging.warning(
                        f"Symbol {symbol} not found in snapshot for {target_date}"
                    )
                    return False
            else:
                # Create a new snapshot for the target date with the custom price
                # First, temporarily set the price in the current portfolio
                original_price = self.current_portfolio.positions[symbol].current_price
                self.current_portfolio.positions[symbol].current_price = price

                # Create snapshot for the target date
                snapshot = self.analyzer.create_snapshot(
                    self.current_portfolio, target_date, save=True
                )

                # Restore original price if we're not updating current
                if not update_current or target_date != date.today():
                    self.current_portfolio.positions[symbol].current_price = original_price

                logging.info(
                    f"Created snapshot for {target_date} with {symbol} price set to {price}"
                )

            return True

        except Exception as e:
            logging.error(f"Error setting position price: {e}")
            return False

    def _recalculate_snapshot_totals(self, snapshot: PortfolioSnapshot) -> None:
        """Recalculate snapshot totals after position price changes."""
        total_positions_value = Decimal("0")
        total_cost_basis = Decimal("0")
        total_unrealized_pnl = Decimal("0")

        for position in snapshot.positions.values():
            if position.quantity > 0 and position.current_price is not None:
                position_value = position.quantity * position.current_price
                position_cost = position.quantity * position.average_cost

                total_positions_value += position_value
                total_cost_basis += position_cost
                total_unrealized_pnl += position_value - position_cost

        snapshot.positions_value = total_positions_value
        snapshot.total_cost_basis = total_cost_basis
        snapshot.total_unrealized_pnl = total_unrealized_pnl
        snapshot.total_value = snapshot.cash_balance + total_positions_value

        if total_cost_basis > 0:
            snapshot.total_unrealized_pnl_percent = (
                total_unrealized_pnl / total_cost_basis
            ) * Decimal("100")
        else:
            snapshot.total_unrealized_pnl_percent = Decimal("0")

    def get_fallback_prices_from_snapshots(self) -> Dict[str, Decimal]:
        """Get fallback prices from the most recent snapshots for positions without current prices.

        Returns:
            Dict mapping symbol to the most recent price found in snapshots
        """
        if not self.current_portfolio:
            return {}

        fallback_prices = {}

        try:
            # Get the most recent snapshots (last 30 days should be sufficient)
            end_date = date.today()
            start_date = end_date - timedelta(days=30)

            snapshots = self.storage.load_snapshots(
                self.current_portfolio.id, start_date, end_date
            )

            if not snapshots:
                return {}

            # Scan from most recent to oldest to find the latest price for each symbol
            for snapshot in reversed(snapshots):
                for symbol, position in snapshot.positions.items():
                    if symbol not in fallback_prices and position.current_price is not None:
                        fallback_prices[symbol] = position.current_price
                        logging.debug(f"Found fallback price for {symbol}: {position.current_price} from {snapshot.date}")

            logging.info(f"Retrieved fallback prices for {len(fallback_prices)} symbols from snapshots")

        except Exception as e:
            logging.warning(f"Failed to get fallback prices from snapshots: {e}")

        return fallback_prices

    def get_positions_with_fallback_prices(self) -> List[Dict]:
        """Get portfolio positions enhanced with fallback prices from snapshots.

        This method ensures that positions have prices even when current market data
        is not available by using the most recent price from snapshots.

        Returns:
            List of position dictionaries with enhanced price information
        """
        if not self.current_portfolio:
            return []

        # Get fallback prices from snapshots
        fallback_prices = self.get_fallback_prices_from_snapshots()

        positions = []

        for symbol, position in self.current_portfolio.positions.items():
            # Get fallback price for this symbol
            fallback_price = fallback_prices.get(symbol)

            # Calculate values using fallback if needed
            effective_price = position.get_effective_price(fallback_price)
            market_value = position.get_market_value_with_fallback(fallback_price)
            unrealized_pnl = position.get_unrealized_pnl_with_fallback(fallback_price)
            unrealized_pnl_percent = position.get_unrealized_pnl_percent_with_fallback(fallback_price)

            position_data = {
                "symbol": symbol,
                "name": position.instrument.name,
                "isin": position.instrument.isin,
                "instrument_type": position.instrument.instrument_type.value,
                "currency": position.instrument.currency.value,
                "quantity": position.quantity,
                "average_cost": position.average_cost,
                "cost_basis": position.cost_basis,
                "current_price": position.current_price,  # Original price (may be None)
                "effective_price": effective_price,       # Price with fallback
                "fallback_price": fallback_price,        # Price from snapshots
                "market_value": market_value,             # With fallback
                "unrealized_pnl": unrealized_pnl,        # With fallback
                "unrealized_pnl_percent": unrealized_pnl_percent,  # With fallback
                "last_updated": position.last_updated,
                "has_current_price": position.current_price is not None,
                "using_fallback": position.current_price is None and fallback_price is not None,
            }

            positions.append(position_data)

        return positions

    def _get_exchange_rate(
        self, from_currency: Currency, to_currency: Currency
    ) -> Optional[Decimal]:
        """Get exchange rate for currency conversion on-demand."""
        if from_currency == to_currency:
            return Decimal("1")

        rate = self.data_manager.get_exchange_rate(from_currency, to_currency)
        return rate



    def refresh_today_snapshot(self) -> Optional[PortfolioSnapshot]:
        """Refresh current prices and update/create today's snapshot.

        This method fetches current market prices and creates or updates
        today's snapshot, making snapshots the single source of truth
        for all portfolio pricing.

        Returns:
            The updated snapshot, or None if no portfolio is loaded
        """
        if not self.current_portfolio:
            return None
        self.update_current_prices()
        today = date.today()
        snapshot = self.analyzer.create_snapshot(self.current_portfolio, today, save=True)
        return snapshot

    def get_portfolio_value(self, use_snapshot: bool = True) -> Decimal:
        """Get total portfolio value in base currency.

        Args:
            use_snapshot: If True, use the latest snapshot value (single source of truth).
                         If False, fall back to in-memory calculation.

        Returns:
            Total portfolio value in base currency
        """
        if not self.current_portfolio:
            return Decimal("0")

        if use_snapshot:
            latest_snapshot = self.storage.get_latest_snapshot(self.current_portfolio.id)
            if latest_snapshot:
                return latest_snapshot.total_value

        # Fallback to in-memory calculation
        return self.analyzer._calculate_portfolio_value(self.current_portfolio)

    def create_snapshot(
        self, snapshot_date: Optional[date] = None, save: bool = True
    ) -> PortfolioSnapshot:
        """Create a comprehensive portfolio snapshot for the given date.

        If creating a snapshot for today, fetches fresh current prices first.
        For historical dates, uses existing prices on the portfolio.
        """
        if not self.current_portfolio:
            raise ValueError("No portfolio loaded")

        # Determine the target date
        target_date = snapshot_date or date.today()

        # For today's snapshot, fetch fresh current prices
        if target_date == date.today():
            logging.info("Fetching fresh current prices for today's snapshot")
            self.update_current_prices()

        return self.analyzer.create_snapshot(self.current_portfolio, target_date, save)

    def create_snapshots_for_range(
        self, start_date: date, end_date: date, save: bool = True
    ) -> List[PortfolioSnapshot]:
        """Create snapshots for a specific date range."""
        if not self.current_portfolio:
            raise ValueError("No portfolio loaded")

        return self.analyzer.create_snapshots_for_range(
            self.current_portfolio, start_date, end_date, save
        )

    def simulate_snapshots_for_range(
        self,
        start_date: date,
        end_date: date,
        exclude_symbols: Optional[List[str]] = None,
        exclude_transaction_ids: Optional[List[str]] = None,
    ) -> List[PortfolioSnapshot]:
        """Simulate snapshots for a date range while excluding certain transactions or symbols.

        Does not persist snapshots to storage.
        """
        if not self.current_portfolio:
            raise ValueError("No portfolio loaded")

        exclude_symbols = {s.upper() for s in (exclude_symbols or [])}
        exclude_ids = set(exclude_transaction_ids or [])

        original_portfolio = self.current_portfolio

        # Filter transactions
        filtered_txns: List[Transaction] = []
        for t in original_portfolio.transactions:
            if exclude_ids and t.id in exclude_ids:
                continue
            if exclude_symbols and t.instrument.symbol.upper() in exclude_symbols:
                continue
            filtered_txns.append(t)

        # Build a temporary portfolio with filtered transactions
        temp_portfolio = Portfolio(
            id=original_portfolio.id,
            name=original_portfolio.name,
            base_currency=original_portfolio.base_currency,
            created_at=original_portfolio.created_at,
            transactions=filtered_txns,
            positions={},
            cash_balances={},
        )

        # Initialize cash balances from the original portfolio's initial state
        # This ensures the simulation starts with the correct cash position
        for currency, amount in original_portfolio.cash_balances.items():
            temp_portfolio.cash_balances[currency] = amount

        # Create snapshots using the temp portfolio without swapping global state
        simulated = self.create_snapshots_for_range(
            start_date, end_date, save=False, portfolio=temp_portfolio
        )

        return simulated



    def get_position_summary(self) -> List[Dict]:
        """Get summary of current positions."""
        if not self.current_portfolio:
            return []

        summary = []
        for symbol, position in self.current_portfolio.positions.items():
            if position.quantity == 0:
                continue

            current_price = position.current_price
            market_value = (
                position.quantity * current_price if current_price else Decimal("0")
            )

            # Calculate unrealized P&L
            unrealized_pnl = Decimal("0")
            unrealized_pnl_percent = Decimal("0")

            if current_price and position.average_cost > 0:
                cost_basis = position.quantity * position.average_cost
                unrealized_pnl = market_value - cost_basis
                unrealized_pnl_percent = (unrealized_pnl / cost_basis) * 100

            summary.append(
                {
                    "symbol": symbol,
                    "name": position.instrument.name,
                    "quantity": position.quantity,
                    "average_cost": position.average_cost,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_percent": unrealized_pnl_percent,
                    "currency": position.instrument.currency.value,
                    "last_updated": position.last_updated,
                    "has_current_price": current_price is not None,
                }
            )

        return summary

    def get_positions_from_snapshot(self) -> List[Dict]:
        """Get positions with prices from the latest snapshot.

        This method returns position data sourced from the latest snapshot,
        ensuring consistency with analytics and other pages that use snapshot data.
        Falls back to in-memory positions with fallback prices if no snapshot exists.

        Returns:
            List of position dictionaries with snapshot-sourced pricing
        """
        if not self.current_portfolio:
            return []

        latest_snapshot = self.storage.get_latest_snapshot(self.current_portfolio.id)
        if not latest_snapshot:
            # Fall back to in-memory calculation with fallback prices
            return self.get_positions_with_fallback_prices()

        positions = []
        for symbol, position in latest_snapshot.positions.items():
            if position.quantity == 0:
                continue

            # Calculate values from snapshot position
            current_price = position.current_price
            market_value = position.market_value if current_price else Decimal("0")
            cost_basis = position.cost_basis

            # Calculate unrealized P&L
            unrealized_pnl = Decimal("0")
            unrealized_pnl_percent = Decimal("0")

            if current_price and position.average_cost > 0 and cost_basis > 0:
                unrealized_pnl = market_value - cost_basis
                unrealized_pnl_percent = (unrealized_pnl / cost_basis) * Decimal("100")

            positions.append({
                "symbol": symbol,
                "name": position.instrument.name,
                "isin": getattr(position.instrument, 'isin', None),
                "instrument_type": position.instrument.instrument_type.value,
                "quantity": position.quantity,
                "average_cost": position.average_cost,
                "cost_basis": cost_basis,
                "current_price": current_price,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_percent": unrealized_pnl_percent,
                "currency": position.instrument.currency.value,
                "last_updated": position.last_updated,
                "has_current_price": current_price is not None,
                "snapshot_date": latest_snapshot.date,
            })

        return positions

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
        return self.analyzer.get_performance_metrics(self.current_portfolio, days)

    def get_external_cash_flows_by_day(
        self, start_date: date, end_date: date
    ) -> Dict[date, Decimal]:
        """Compute net external cash flows (deposits/withdrawals) per day in base currency."""
        if not self.current_portfolio:
            return {}
        return self.analyzer.get_external_cash_flows_by_day(
            self.current_portfolio, start_date, end_date
        )

    def get_cash_fx_summary(self) -> Dict[Currency, Dict[str, Decimal]]:
        """Compute FX summary for cash balances per currency.

        For each currency, reconstruct a running average base-cost for the remaining
        foreign cash using historical FX on deposit dates and average-cost reduction
        on withdrawals. Returns per-currency metrics including current base value
        and unrealized FX PnL relative to the base-cost of remaining cash.
        """
        if not self.current_portfolio:
            return {}

        base = self.current_portfolio.base_currency

        # Running balances and base-costs by currency
        foreign_balance: Dict[Currency, Decimal] = {}
        base_cost: Dict[Currency, Decimal] = {}

        # Process cash transactions chronologically
        for txn in sorted(
            self.current_portfolio.transactions, key=lambda t: t.timestamp
        ):
            if txn.transaction_type not in [
                TransactionType.DEPOSIT,
                TransactionType.WITHDRAWAL,
            ]:
                continue
            cur = txn.currency
            amt = txn.total_value

            if cur not in foreign_balance:
                foreign_balance[cur] = Decimal("0")
                base_cost[cur] = Decimal("0")

            # Historical FX on transaction date (or 1 if base)
            if cur == base:
                fx = Decimal("1")
            else:
                fx = self.data_manager.get_historical_fx_rate_on(
                    txn.timestamp.date(), cur, base
                ) or Decimal("0")
                if fx == 0:
                    # As a fallback, try current FX
                    fx = self.data_manager.get_exchange_rate(cur, base) or Decimal("1")

            if txn.transaction_type == TransactionType.DEPOSIT:
                foreign_balance[cur] += amt
                base_cost[cur] += amt * fx
            else:
                # Reduce at running average base-cost
                existing_bal = foreign_balance[cur]
                existing_cost = base_cost[cur]
                if existing_bal > 0:
                    avg_rate = existing_cost / existing_bal
                else:
                    avg_rate = fx
                foreign_balance[cur] = existing_bal - amt
                base_cost[cur] = existing_cost - (amt * avg_rate)

                # Clean near-zero noise
                if foreign_balance[cur].copy_abs() < Decimal("0.0000001"):
                    foreign_balance[cur] = Decimal("0")
                if base_cost[cur].copy_abs() < Decimal("0.0000001"):
                    base_cost[cur] = Decimal("0")

        # Build summary using current balances and current FX
        result: Dict[Currency, Dict[str, Decimal]] = {}
        currencies = set(self.current_portfolio.cash_balances.keys()) | set(
            foreign_balance.keys()
        )
        for cur in currencies:
            amt_foreign = self.current_portfolio.cash_balances.get(cur, Decimal("0"))
            cost_base = base_cost.get(cur, Decimal("0"))

            if cur == base:
                rate = Decimal("1")
            else:
                rate = (
                    self.data_manager.get_exchange_rate(cur, base)
                    or self.data_manager.get_historical_fx_rate_on(
                        date.today(), cur, base
                    )
                    or Decimal("1")
                )

            current_value_base = amt_foreign * rate
            avg_cost_rate = (cost_base / amt_foreign) if amt_foreign else Decimal("0")
            fx_unrealized = current_value_base - cost_base

            result[cur] = {
                "foreign_amount": amt_foreign,
                "base_cost": cost_base,
                "current_rate": rate,
                "current_value_base": current_value_base,
                "fx_unrealized_pnl_base": fx_unrealized,
                "avg_cost_rate": avg_cost_rate,
            }

        return result

    def get_cash_ytd_fx_summary(self) -> Dict[Currency, Dict[str, Decimal]]:
        """Compute YTD FX PnL for cash per currency based on transaction timing.

        Method:
        - Determine opening foreign balance as of Jan 1 (pre-YTD flows) and value it at Jan 1 FX.
        - For each YTD cash flow (deposit +, withdrawal -) on date t, add a contribution valued at FX(t).
        - YTD FX PnL = current_balance * FX(today) - [opening_balance * FX(Jan1) + sum(YTD_flows * FX(t))].
        Returns per currency dict with keys: ytd_fx_pnl_base, ytd_fx_percent, base_exposure_cost.
        """
        if not self.current_portfolio:
            return {}

        base = self.current_portfolio.base_currency
        today = date.today()
        jan1 = date(today.year, 1, 1)

        # Collect currencies to evaluate
        currencies: set[Currency] = set(self.current_portfolio.cash_balances.keys())
        for txn in self.current_portfolio.transactions:
            if txn.transaction_type in [
                TransactionType.DEPOSIT,
                TransactionType.WITHDRAWAL,
            ]:
                currencies.add(txn.currency)

        results: Dict[Currency, Dict[str, Decimal]] = {}

        for cur in currencies:
            # FX helpers
            def fx_on(d: date) -> Decimal:
                if cur == base:
                    return Decimal("1")
                r = self.data_manager.get_historical_fx_rate_on(d, cur, base)
                if r is None:
                    r = self.data_manager.get_exchange_rate(cur, base)
                return r or Decimal("1")

            fx_today = fx_on(today)
            fx_jan1 = fx_on(jan1)

            # Opening balance before Jan1
            opening_foreign = Decimal("0")
            for txn in sorted(
                self.current_portfolio.transactions, key=lambda t: t.timestamp
            ):
                if txn.transaction_type not in [
                    TransactionType.DEPOSIT,
                    TransactionType.WITHDRAWAL,
                ]:
                    continue
                if txn.currency != cur:
                    continue
                if txn.timestamp.date() < jan1:
                    amt = (
                        txn.total_value
                        if txn.transaction_type == TransactionType.DEPOSIT
                        else -txn.total_value
                    )
                    opening_foreign += amt

            opening_base_cost = opening_foreign * fx_jan1

            # YTD flows contribution valued at their date FX
            ytd_base_contrib = Decimal("0")
            for txn in sorted(
                self.current_portfolio.transactions, key=lambda t: t.timestamp
            ):
                if txn.transaction_type not in [
                    TransactionType.DEPOSIT,
                    TransactionType.WITHDRAWAL,
                ]:
                    continue
                if txn.currency != cur:
                    continue
                d = txn.timestamp.date()
                if d < jan1:
                    continue
                amt_signed = (
                    txn.total_value
                    if txn.transaction_type == TransactionType.DEPOSIT
                    else -txn.total_value
                )
                ytd_base_contrib += amt_signed * fx_on(d)

            # Current value
            current_foreign = self.current_portfolio.cash_balances.get(
                cur, Decimal("0")
            )
            current_value_base = current_foreign * fx_today

            base_exposure_cost = opening_base_cost + ytd_base_contrib
            ytd_fx_pnl_base = current_value_base - base_exposure_cost
            ytd_fx_percent = (
                (ytd_fx_pnl_base / base_exposure_cost * 100)
                if base_exposure_cost != 0
                else Decimal("0")
            )

            results[cur] = {
                "ytd_fx_pnl_base": ytd_fx_pnl_base,
                "ytd_fx_percent": ytd_fx_percent,
                "base_exposure_cost": base_exposure_cost,
            }

        return results

    def _discover_instrument_info(
        self,
        symbol: Optional[str],
        isin: Optional[str],
        currency: Optional[Currency],
        notes: Optional[str],
        instrument_type: Optional[str] = None,
    ) -> Optional[Dict]:
        """Enhanced instrument information discovery with intelligent fallbacks.

        Note: ISIN is optional. If provided, use it directly. If not provided,
        work with available symbol/name information without searching for ISIN.
        Company names can be automatically converted to symbols.
        """

        # Case 1: Both symbol and ISIN provided
        if symbol and isin:
            return self._handle_symbol_and_isin(symbol, isin, currency, notes, instrument_type)

        # Case 2: Only ISIN provided (symbol will be discovered or created)
        elif isin and not symbol:
            return self._handle_isin_only(isin, currency, notes, instrument_type)

        # Case 3: Only symbol provided (work with what we have, don't search for ISIN)
        elif symbol and not isin:
            return self._handle_symbol_only(symbol, currency, notes, instrument_type)

        # Case 4: Neither provided (invalid)
        else:
            logging.error("Neither symbol nor ISIN provided")
            return None

    def _handle_symbol_and_isin(
        self, symbol: str, isin: str, currency: Optional[Currency], notes: Optional[str], instrument_type: Optional[str] = None
    ) -> Dict:
        """Handle case where both symbol and ISIN are provided."""
        normalized_symbol = symbol.strip().lstrip("$").upper()

        # Try to get comprehensive info from data providers
        instrument_info = self.data_manager.get_instrument_info(normalized_symbol)

        if instrument_info:
            # Use provider data but ensure ISIN matches
            return {
                "symbol": instrument_info.symbol,
                "name": instrument_info.name,
                "instrument_type": self._get_instrument_type(instrument_type, normalized_symbol, isin),
                "currency": currency or instrument_info.currency,
                "exchange": instrument_info.exchange,
                "isin": isin.upper(),  # Use provided ISIN
            }
        else:
            # Fallback: create basic info
            return self._create_basic_instrument_info(
                normalized_symbol, isin, currency, notes, instrument_type
            )

    def _handle_isin_only(
        self, isin: str, currency: Optional[Currency], notes: Optional[str], instrument_type: Optional[str] = None
    ) -> Dict:
        """Handle case where only ISIN is provided (symbol will be discovered or created)."""
        isin = isin.upper().strip()

        # Try to find instrument by ISIN from our known mappings
        instrument_info = self.data_manager.search_by_isin(isin)

        if instrument_info:
            # Found the instrument - use its data
            return {
                "symbol": instrument_info.symbol,
                "name": instrument_info.name,
                "instrument_type": self._get_instrument_type(instrument_type, instrument_info.symbol, isin),
                "currency": currency or instrument_info.currency,
                "exchange": instrument_info.exchange,
                "isin": isin,
            }
        else:
            # Not found in known mappings - create placeholder with intelligent defaults
            return self._create_placeholder_from_isin(isin, currency, notes, instrument_type)

    def _handle_symbol_only(
        self, symbol: str, currency: Optional[Currency], notes: Optional[str], instrument_type: Optional[str] = None
    ) -> Dict:
        """Handle case where only symbol is provided."""
        normalized_symbol = symbol.strip().lstrip("$").upper()

        # Check if this might be a company name rather than a symbol
        if self._is_likely_company_name(normalized_symbol):
            # Try to find the symbol for this company name
            found_symbol = self._find_symbol_from_company_name(normalized_symbol)
            if found_symbol:
                normalized_symbol = found_symbol
                logging.info(
                    f"Converted company name '{symbol}' to symbol '{found_symbol}'"
                )

        # Get instrument info from data providers
        instrument_info = self.data_manager.get_instrument_info(normalized_symbol)

        if instrument_info:
            # Use provider data
            return {
                "symbol": instrument_info.symbol,
                "name": instrument_info.name,
                "instrument_type": self._get_instrument_type(instrument_type, normalized_symbol, None),
                "currency": currency or instrument_info.currency,
                "exchange": instrument_info.exchange,
                "isin": instrument_info.isin,
            }
        else:
            # Create basic instrument info if not found
            return self._create_basic_instrument_info(
                normalized_symbol, None, currency, notes, instrument_type
            )

    def _is_likely_company_name(self, text: str) -> bool:
        """Check if text is likely a company name rather than a stock symbol."""
        # Common patterns that suggest company names
        company_indicators = [
            "inc",
            "corp",
            "corporation",
            "company",
            "co",
            "ltd",
            "limited",
            "plc",
            "ag",
            "sa",
            "nv",
            "holdings",
            "group",
            "technologies",
            "systems",
            "solutions",
            "services",
            "international",
            "global",
        ]

        text_lower = text.lower()

        # Check for company suffixes
        for indicator in company_indicators:
            if indicator in text_lower:
                return True

        # Check if it's all lowercase (likely company name)
        if text.islower() and len(text) > 3:
            return True

        # Check if it contains spaces (likely company name)
        if " " in text:
            return True

        # Check if it's not all uppercase (likely company name)
        if not text.isupper():
            return True

        # Check if it's a known company name that should be converted
        known_companies = [
            "apple",
            "microsoft",
            "google",
            "alphabet",
            "tesla",
            "amazon",
            "meta",
            "facebook",
            "nvidia",
            "netflix",
            "berkshire",
            "hathaway",
            "jpmorgan",
            "chase",
            "johnson",
            "visa",
            "procter",
            "gamble",
            "unitedhealth",
            "home",
            "depot",
            "mastercard",
            "disney",
            "walt",
            "paypal",
            "asml",
            "holding",
            "holdings",
        ]

        if text_lower in known_companies:
            return True

        return False

    def _find_symbol_from_company_name(self, company_name: str) -> Optional[str]:
        """Find stock symbol from company name using known mappings."""
        # Common company name to symbol mappings
        company_to_symbol = {
            "apple": "AAPL",
            "apple inc": "AAPL",
            "apple inc.": "AAPL",
            "apple computer": "AAPL",
            "microsoft": "MSFT",
            "microsoft corporation": "MSFT",
            "microsoft corp": "MSFT",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "alphabet inc": "GOOGL",
            "alphabet inc.": "GOOGL",
            "tesla": "TSLA",
            "tesla inc": "TSLA",
            "tesla inc.": "TSLA",
            "amazon": "AMZN",
            "amazon.com": "AMZN",
            "amazon.com inc": "AMZN",
            "amazon.com inc.": "AMZN",
            "meta": "META",
            "meta platforms": "META",
            "meta platforms inc": "META",
            "meta platforms inc.": "META",
            "facebook": "META",
            "facebook inc": "META",
            "facebook inc.": "META",
            "nvidia": "NVDA",
            "nvidia corporation": "NVDA",
            "nvidia corp": "NVDA",
            "netflix": "NFLX",
            "netflix inc": "NFLX",
            "netflix inc.": "NFLX",
            "berkshire hathaway": "BRK.A",
            "berkshire hathaway inc": "BRK.A",
            "berkshire hathaway inc.": "BRK.A",
            "jpmorgan": "JPM",
            "jpmorgan chase": "JPM",
            "jpmorgan chase & co": "JPM",
            "jpmorgan chase & co.": "JPM",
            "johnson & johnson": "JNJ",
            "johnson and johnson": "JNJ",
            "visa": "V",
            "visa inc": "V",
            "visa inc.": "V",
            "procter & gamble": "PG",
            "procter and gamble": "PG",
            "procter & gamble co": "PG",
            "procter & gamble co.": "PG",
            "unitedhealth": "UNH",
            "unitedhealth group": "UNH",
            "unitedhealth group inc": "UNH",
            "unitedhealth group inc.": "UNH",
            "home depot": "HD",
            "the home depot": "HD",
            "the home depot inc": "HD",
            "the home depot inc.": "HD",
            "mastercard": "MA",
            "mastercard inc": "MA",
            "mastercard inc.": "MA",
            "disney": "DIS",
            "the walt disney company": "DIS",
            "walt disney": "DIS",
            "paypal": "PYPL",
            "paypal holdings": "PYPL",
            "paypal holdings inc": "PYPL",
            "paypal holdings inc.": "PYPL",
            "asml": "ASML",
            "asml holding": "ASML",
            "asml holding nv": "ASML",
            "asml holding n.v.": "ASML",
            "asml holdings": "ASML",
            "asml holdings nv": "ASML",
            "asml holdings n.v.": "ASML",
        }

        # Try exact match first
        company_lower = company_name.lower().strip()
        if company_lower in company_to_symbol:
            return company_to_symbol[company_lower]

        # Try partial matches
        for company, symbol in company_to_symbol.items():
            if company_lower in company or company in company_lower:
                return symbol

        # If no match found, return None (will use company name as symbol)
        return None

    def _create_basic_instrument_info(
        self,
        symbol: str,
        isin: Optional[str],
        currency: Optional[Currency],
        notes: Optional[str],
        instrument_type: Optional[str] = None,
    ) -> Dict:
        """Create basic instrument info when provider data is unavailable."""
        # Use provided instrument type if available, otherwise infer from symbol
        if instrument_type:
            try:
                from src.portfolio.models import InstrumentType
                inferred_type = InstrumentType(instrument_type.lower())
            except ValueError:
                # If invalid instrument type provided, fall back to inference
                inferred_type = self._infer_instrument_type(symbol, isin)
        else:
            inferred_type = self._infer_instrument_type(symbol, isin)

        # Try to find a better name than the symbol
        instrument_name = self._find_instrument_name(symbol, isin, notes)

        return {
            "symbol": symbol.upper(),
            "name": instrument_name,
            "instrument_type": inferred_type,
            "currency": currency or Currency.USD,
            "exchange": None,
            "isin": isin,
        }

    def _create_placeholder_from_isin(
        self, isin: str, currency: Optional[Currency], notes: Optional[str], instrument_type: Optional[str] = None
    ) -> Dict:
        """Create placeholder instrument info when ISIN lookup fails."""
        # Use provided instrument type if available, otherwise infer from ISIN prefix
        if instrument_type:
            try:
                from src.portfolio.models import InstrumentType
                inferred_type = InstrumentType(instrument_type.lower())
            except ValueError:
                # If invalid instrument type provided, fall back to inference
                inferred_type = self._infer_instrument_type_from_isin(isin)
        else:
            inferred_type = self._infer_instrument_type_from_isin(isin)

        # Create a descriptive name
        if notes and len(notes) > 10:
            instrument_name = notes
        else:
            instrument_name = f"Instrument {isin}"

        # Create placeholder symbol
        if isin.upper().startswith("XS"):
            # Bonds - use ISIN prefix
            placeholder_symbol = f"BOND_{isin[:8]}"
        elif isin.upper().startswith("US"):
            # US instruments - use ISIN prefix
            placeholder_symbol = f"US_{isin[:8]}"
        elif isin.upper().startswith("CH"):
            # Swiss instruments - use ISIN prefix
            placeholder_symbol = f"CH_{isin[:8]}"
        else:
            # Generic - use ISIN prefix
            placeholder_symbol = f"ISIN_{isin[:8]}"

        return {
            "symbol": placeholder_symbol,
            "name": instrument_name,
            "instrument_type": inferred_type,
            "currency": currency or Currency.USD,
            "exchange": None,
            "isin": isin,
        }

    def _infer_instrument_type(
        self, symbol: str, isin: Optional[str]
    ) -> InstrumentType:
        """Infer instrument type from symbol and ISIN."""
        if isin:
            return self._infer_instrument_type_from_isin(isin)

        # Infer from symbol
        symbol_upper = symbol.upper()

        # Check for CASH
        if symbol_upper == "CASH":
            return InstrumentType.CASH

        # Common ETF symbols (check ETFs first to avoid conflicts)
        etf_symbols = {
            "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO", "VXUS", "VT",
            "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "DBB", "DJP", "ARKK", "ARKW",
            "ARKF", "ARKG", "ARKQ", "ARKX", "SOXL", "SOXS", "TQQQ", "SQQQ", "LABU",
            "LABD", "FAS", "FAZ", "ERX", "ERY", "DPST", "KRE", "XOP", "XLE", "XLF",
            "XLK", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC"
        }
        if symbol_upper in etf_symbols:
            return InstrumentType.ETF

        # Common bond ETFs and bond-related symbols
        bond_symbols = {
            "TLT", "IEF", "TIP", "LQD", "HYG", "EMB", "SHY", "SHV", "BND", "AGG", "BIL",
            "VCIT", "VCSH", "VGIT", "VGSH", "VTIP", "VGLT", "VCLT", "VWOB", "VWITX",
            "TMF", "TMV", "TBT", "TLH", "TLO", "TENZ", "TAN", "TZA"
        }
        if symbol_upper in bond_symbols:
            return InstrumentType.BOND

        # Crypto symbols
        crypto_symbols = {"BTC", "ETH", "ADA", "DOT", "LINK", "LTC", "BCH", "XRP", "SOL", "AVAX"}
        if symbol_upper in crypto_symbols:
            return InstrumentType.CRYPTO

        # Common stock symbols (well-known companies) - check these before pattern matching
        stock_symbols = {
            "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA", "NFLX",
            "BRK.A", "BRK.B", "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL",
            "ASML", "NVO", "LLY", "PFE", "ABBV", "TMO", "AVGO", "COST", "PEP", "KO",
            "WMT", "MRK", "ABT", "ACN", "CVX", "XOM", "BAC", "WFC", "GS", "MS", "BLK"
        }
        if symbol_upper in stock_symbols:
            return InstrumentType.STOCK

        # Check symbol patterns for better inference (only for unknown symbols)
        if self._is_likely_etf_symbol(symbol_upper):
            return InstrumentType.ETF
        elif self._is_likely_bond_symbol(symbol_upper):
            return InstrumentType.BOND
        elif self._is_likely_crypto_symbol(symbol_upper):
            return InstrumentType.CRYPTO

        # Default to stock for unknown symbols
        return InstrumentType.STOCK

    def _is_likely_etf_symbol(self, symbol: str) -> bool:
        """Check if symbol is likely an ETF based on patterns."""
        # ETFs often have patterns like:
        # - 3-letter symbols ending in common ETF suffixes
        # - Symbols that represent sectors or themes
        # - Common ETF naming conventions

        if len(symbol) >= 3:
            # Common ETF suffixes
            etf_suffixes = ["ETF", "FND", "TR", "FD", "IX", "EX", "AX", "RX", "TX"]
            for suffix in etf_suffixes:
                if symbol.endswith(suffix):
                    return True

            # Sector ETFs
            sector_etfs = ["XLF", "XLK", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC"]
            if symbol in sector_etfs:
                return True

            # Leveraged/inverse ETFs
            leveraged_patterns = ["TQQQ", "SQQQ", "SOXL", "SOXS", "LABU", "LABD", "FAS", "FAZ"]
            if symbol in leveraged_patterns:
                return True

        return False

    def _is_likely_bond_symbol(self, symbol: str) -> bool:
        """Check if symbol is likely a bond based on patterns."""
        # Bonds often have patterns like:
        # - Treasury-related symbols
        # - Corporate bond symbols
        # - Municipal bond symbols

        if len(symbol) >= 3:
            # Treasury-related (exact matches or starts with)
            treasury_patterns = ["T", "TB", "TN", "TL", "TS", "TIPS", "TBILL", "TBOND"]
            for pattern in treasury_patterns:
                if symbol.startswith(pattern) and len(pattern) >= 2:
                    return True

            # Corporate bond patterns (exact matches)
            if symbol in ["CORP", "BOND", "DEBT"]:
                return True

            # Municipal bond patterns (exact matches)
            if symbol in ["MUNI", "MUN"]:
                return True

        return False

    def _is_likely_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is likely a cryptocurrency based on patterns."""
        # Cryptocurrencies often have patterns like:
        # - 3-4 letter symbols
        # - Common crypto naming conventions

        if 3 <= len(symbol) <= 5:
            # Common crypto prefixes/suffixes
            crypto_patterns = ["BTC", "ETH", "ADA", "DOT", "LINK", "LTC", "BCH", "XRP", "SOL", "AVAX"]
            if symbol in crypto_patterns:
                return True

            # Check for common crypto naming patterns (only for short symbols)
            if len(symbol) <= 4 and (symbol.endswith("X") or symbol.endswith("T") or symbol.endswith("N")):
                return True

        return False

    def _infer_instrument_type_from_isin(self, isin: str) -> InstrumentType:
        """Infer instrument type from ISIN prefix with enhanced logic."""
        isin_upper = isin.upper()

        # Bond ISINs
        if isin_upper.startswith("XS"):  # International bonds
            return InstrumentType.BOND
        elif isin_upper.startswith("US") and len(isin_upper) >= 12:
            # US ISINs - check for bond patterns
            if any(pattern in isin_upper for pattern in ["BOND", "DEBT", "TREAS", "CORP"]):
                return InstrumentType.BOND
            # Check if it's a known bond ISIN
            elif isin_upper in ["US4642876555", "US78464A7353", "US78464A7353"]:  # TLT, BIL examples
                return InstrumentType.BOND
            else:
                return InstrumentType.STOCK  # Default for US ISINs

        # European ISINs
        elif isin_upper.startswith("IE"):  # Ireland (ETFs)
            return InstrumentType.ETF
        elif isin_upper.startswith("LU"):  # Luxembourg (ETFs)
            return InstrumentType.ETF
        elif isin_upper.startswith("DE"):  # Germany
            return InstrumentType.STOCK
        elif isin_upper.startswith("FR"):  # France
            return InstrumentType.STOCK
        elif isin_upper.startswith("GB"):  # UK
            return InstrumentType.STOCK

        # Swiss ISINs
        elif isin_upper.startswith("CH"):
            return InstrumentType.STOCK

        # Japanese ISINs
        elif isin_upper.startswith("JP"):
            return InstrumentType.STOCK

        # Canadian ISINs
        elif isin_upper.startswith("CA"):
            return InstrumentType.STOCK

        # Australian ISINs
        elif isin_upper.startswith("AU"):
            return InstrumentType.STOCK

        # Default to stock for unknown ISIN patterns
        return InstrumentType.STOCK

    def _get_instrument_type(
        self,
        user_provided_type: Optional[str],
        symbol: Optional[str],
        isin: Optional[str]
    ) -> InstrumentType:
        """Get instrument type, prioritizing user-provided type over automatic inference."""
        if user_provided_type:
            try:
                from src.portfolio.models import InstrumentType
                return InstrumentType(user_provided_type.lower())
            except ValueError:
                # If invalid instrument type provided, fall back to inference
                pass

        # Fall back to automatic inference
        if symbol:
            return self._infer_instrument_type(symbol, isin)
        elif isin:
            return self._infer_instrument_type_from_isin(isin)
        else:
            return InstrumentType.STOCK

    def _find_instrument_name(
        self, symbol: str, isin: Optional[str], notes: Optional[str]
    ) -> str:
        """Find a better instrument name than just the symbol."""
        symbol_upper = symbol.upper()

        if symbol_upper == "CASH":
            return "Cash"

        # Use notes if provided and meaningful
        if notes and len(notes) > 5 and not notes.upper().startswith(symbol_upper):
            return notes

        # Common stock symbols with known names
        symbol_to_name = {
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corporation",
            "GOOGL": "Alphabet Inc.",
            "TSLA": "Tesla Inc.",
            "AMZN": "Amazon.com Inc.",
            "META": "Meta Platforms Inc.",
            "NVDA": "NVIDIA Corporation",
            "NFLX": "Netflix Inc.",
            "TSLA": "Tesla Inc.",
            "BRK.A": "Berkshire Hathaway Inc.",
            "BRK.B": "Berkshire Hathaway Inc.",
            "JPM": "JPMorgan Chase & Co.",
            "JNJ": "Johnson & Johnson",
            "V": "Visa Inc.",
            "PG": "Procter & Gamble Co.",
            "UNH": "UnitedHealth Group Inc.",
            "HD": "The Home Depot Inc.",
            "MA": "Mastercard Inc.",
            "DIS": "The Walt Disney Company",
            "PYPL": "PayPal Holdings Inc.",
        }

        if symbol_upper in symbol_to_name:
            return symbol_to_name[symbol_upper]

        # Common bond ETFs
        bond_symbols = {
            "TLT": "iShares 20+ Year Treasury Bond ETF",
            "IEF": "iShares 7-10 Year Treasury Bond ETF",
            "TIP": "iShares TIPS Bond ETF",
            "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
            "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
            "EMB": "iShares J.P. Morgan USD Emerging Markets Bond ETF",
            "SHY": "iShares 1-3 Year Treasury Bond ETF",
            "SHV": "iShares Short Treasury Bond ETF",
            "BND": "Vanguard Total Bond Market ETF",
            "AGG": "iShares Core U.S. Aggregate Bond ETF",
            "BIL": "SPDR Bloomberg 1-3 Month T-Bill ETF",
        }

        if symbol_upper in bond_symbols:
            return bond_symbols[symbol_upper]

        # If no better name found, return the symbol
        return symbol_upper


