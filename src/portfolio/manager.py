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

        # Normalize symbol (strip chat-style '$', uppercase)
        normalized_symbol = symbol.strip().lstrip("$").upper()

        # Get instrument info from data providers
        instrument_info = self.data_manager.get_instrument_info(normalized_symbol)
        if not instrument_info:
            # Create basic instrument info if not found
            instrument_info_dict = {
                "symbol": normalized_symbol,
                "name": normalized_symbol,
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
            f"Added {transaction_type} transaction: {quantity} {normalized_symbol} @ {price}"
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

    def _get_exchange_rate(
        self, from_currency: Currency, to_currency: Currency
    ) -> Optional[Decimal]:
        """Get exchange rate for currency conversion on-demand."""
        if from_currency == to_currency:
            return Decimal("1")

        rate = self.data_manager.get_exchange_rate(from_currency, to_currency)
        return rate

    def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value in base currency."""
        if not self.current_portfolio:
            return Decimal("0")

        # Create a function that fetches exchange rates on-demand
        def get_rate(
            from_currency: Currency, to_currency: Currency
        ) -> Optional[Decimal]:
            return self._get_exchange_rate(from_currency, to_currency)

        return self.current_portfolio.get_total_value_with_rate_function(get_rate)

    def create_snapshot(
        self, snapshot_date: Optional[date] = None, save: bool = True
    ) -> PortfolioSnapshot:
        """Create a comprehensive portfolio snapshot for the given date."""
        if not self.current_portfolio:
            raise ValueError("No portfolio loaded")

        snapshot_date = snapshot_date or date.today()

        # Note: Prices are not automatically updated when creating snapshots
        # Users should manually update prices via the UI if needed

        total_value = self.get_portfolio_value()

        # Calculate cash balance in base currency using on-demand rate fetching
        cash_balance = Decimal("0")
        for currency, amount in self.current_portfolio.cash_balances.items():
            if currency == self.current_portfolio.base_currency:
                cash_balance += amount
            else:
                rate = self._get_exchange_rate(
                    currency, self.current_portfolio.base_currency
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
        if save:
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

        # Build a union of symbols that may exist across this period
        all_symbols = set()
        for txn in self.current_portfolio.transactions:
            if (
                txn.instrument.symbol != "CASH"
                and start_date <= txn.timestamp.date() <= end_date
            ):
                all_symbols.add(txn.instrument.symbol)

        price_map = self._build_historical_price_map(
            sorted(all_symbols), start_date, end_date
        )

        snapshots = []
        current_date = start_date

        while current_date <= end_date:
            try:
                # Create a temporary portfolio state for this date
                temp_portfolio = self._get_portfolio_state_for_date(
                    current_date, price_map
                )

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
        self, start_date: date, end_date: date, save: bool = True
    ) -> List[PortfolioSnapshot]:
        """Create snapshots for a specific date range."""
        if not self.current_portfolio:
            raise ValueError("No portfolio loaded")

        if start_date > end_date:
            raise ValueError("Start date must be before or equal to end date")

        # Build a union of symbols that may exist across this period
        all_symbols = set()
        for txn in self.current_portfolio.transactions:
            if txn.instrument.symbol != "CASH" and txn.timestamp.date() <= end_date:
                all_symbols.add(txn.instrument.symbol)

        price_map = self._build_historical_price_map(
            sorted(all_symbols), start_date, end_date
        )

        snapshots = []
        current_date = start_date

        while current_date <= end_date:
            try:
                # Create a temporary portfolio state for this date
                temp_portfolio = self._get_portfolio_state_for_date(
                    current_date, price_map
                )

                # Temporarily set current portfolio to the historical state
                original_portfolio = self.current_portfolio
                self.current_portfolio = temp_portfolio

                # Create snapshot for this date
                snapshot = self.create_snapshot(current_date, save=save)
                snapshots.append(snapshot)

                # Restore original portfolio
                self.current_portfolio = original_portfolio

            except Exception as e:
                logging.warning(f"Failed to create snapshot for {current_date}: {e}")

            current_date += timedelta(days=1)

        return snapshots

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

        # Swap in temp, create snapshots without saving, restore original
        self.current_portfolio = temp_portfolio
        try:
            simulated = self.create_snapshots_for_range(
                start_date, end_date, save=False
            )
        finally:
            self.current_portfolio = original_portfolio

        return simulated

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

    def _get_portfolio_state_for_date(
        self,
        target_date: date,
        price_map: Optional[Dict[str, Dict[date, Decimal]]] = None,
    ) -> Portfolio:
        """Get portfolio state as it was on a specific date.

        If price_map is provided, use it to set position prices deterministically for the target date.
        """
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

        # Update prices for the target date
        if price_map is not None:
            self._apply_price_map_for_date(portfolio_copy, price_map, target_date)
        else:
            if target_date == date.today():
                self._update_portfolio_prices(portfolio_copy)
            else:
                self._update_portfolio_prices_for_date(portfolio_copy, target_date)

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

    def _build_historical_price_map(
        self, symbols: List[str], start_date: date, end_date: date
    ) -> Dict[str, Dict[date, Decimal]]:
        """Build a symbol -> {date: price} map for a date range with forward-fill for missing non-trading days."""
        price_map: Dict[str, Dict[date, Decimal]] = {}
        for symbol in symbols:
            try:
                series = self.data_manager.get_historical_prices(
                    symbol, start_date, end_date
                )
                daily: Dict[date, Decimal] = {}
                for pd_item in series:
                    px = (
                        pd_item.close_price
                        or pd_item.open_price
                        or pd_item.high_price
                        or pd_item.low_price
                    )
                    if px is not None:
                        daily[pd_item.date] = px

                # Forward-fill through the calendar window
                current = start_date
                last_px: Optional[Decimal] = None
                while current <= end_date:
                    if current in daily:
                        last_px = daily[current]
                    elif last_px is not None:
                        daily[current] = last_px
                    current += timedelta(days=1)

                if daily:
                    price_map[symbol] = daily
            except Exception as e:
                logging.warning(f"Failed building price map for {symbol}: {e}")

        return price_map

    def _apply_price_map_for_date(
        self,
        portfolio: Portfolio,
        price_map: Dict[str, Dict[date, Decimal]],
        target_date: date,
    ) -> None:
        """Apply precomputed historical prices to a portfolio for a given date."""
        for position in portfolio.positions.values():
            symbol = position.instrument.symbol
            if symbol in price_map and target_date in price_map[symbol]:
                position.current_price = price_map[symbol][target_date]
                position.last_updated = datetime.combine(
                    target_date, datetime.min.time()
                )
            else:
                # Fallback to on-demand per-date logic
                self._update_portfolio_prices_for_date(portfolio, target_date)

    def _update_portfolio_prices_for_date(
        self, portfolio: Portfolio, target_date: date
    ) -> None:
        """Update prices for all positions using historical data for a specific date.

        Uses the close price for target_date when available. Falls back to the most
        recent available price in the range [target_date-3d, target_date] if the
        specific date is missing (weekends/holidays). As a last resort, attempts
        to get the current price.
        """
        for position in portfolio.positions.values():
            symbol = position.instrument.symbol
            try:
                # Try exact date first
                prices = self.data_manager.get_historical_prices(
                    symbol, target_date, target_date
                )
                selected_price: Optional[Decimal] = None

                if prices:
                    # Take the first/only entry for that date
                    pd0 = prices[0]
                    selected_price = (
                        pd0.close_price
                        or pd0.open_price
                        or pd0.high_price
                        or pd0.low_price
                    )

                # If nothing for that exact date (e.g., weekend), look back a few days
                if selected_price is None:
                    lookback_start = target_date - timedelta(days=3)
                    prices = self.data_manager.get_historical_prices(
                        symbol, lookback_start, target_date
                    )
                    # choose the last available (closest to target_date)
                    if prices:
                        last = prices[-1]
                        selected_price = (
                            last.close_price
                            or last.open_price
                            or last.high_price
                            or last.low_price
                        )

                # Final fallback to current price
                if selected_price is None:
                    selected_price = self.data_manager.get_current_price(symbol)

                if selected_price is not None:
                    position.current_price = selected_price
                    # stamp last_updated as target_date at end of day for clarity
                    position.last_updated = datetime.combine(
                        target_date, datetime.min.time()
                    )
            except Exception as e:
                logging.warning(
                    f"Failed to set historical price for {symbol} on {target_date}: {e}"
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

    def get_external_cash_flows_by_day(
        self, start_date: date, end_date: date
    ) -> Dict[date, Decimal]:
        """Compute net external cash flows (deposits/withdrawals) per day in base currency.

        Deposits are positive contributions; withdrawals are negative. Other transaction
        types (buy/sell/dividend/interest) are ignored as they are internal.
        Uses historical FX on the transaction date when available.
        """
        if not self.current_portfolio:
            return {}

        flows: Dict[date, Decimal] = {}
        base = self.current_portfolio.base_currency

        for txn in self.current_portfolio.transactions:
            txn_date = txn.timestamp.date()
            if txn_date < start_date or txn_date > end_date:
                continue
            if txn.transaction_type not in [
                TransactionType.DEPOSIT,
                TransactionType.WITHDRAWAL,
            ]:
                continue

            # Amount in txn currency
            amount = txn.total_value

            # Convert to base using historical FX
            if txn.currency == base:
                amount_base = amount
            else:
                rate = self.data_manager.get_historical_fx_rate_on(
                    txn_date, txn.currency, base
                )
                amount_base = amount * rate if rate else amount

            if txn.transaction_type == TransactionType.WITHDRAWAL:
                amount_base = -amount_base

            flows[txn_date] = flows.get(txn_date, Decimal("0")) + amount_base

        return flows
