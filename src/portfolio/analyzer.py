"""
Portfolio analyzer for calculating metrics and creating snapshots.

SIMPLE RULE: This class only does CALCULATIONS - it NEVER fetches data.
The manager is responsible for fetching prices before calling analyzer methods.
"""

import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List

from ..data_providers.manager import DataProviderManager
from .models import Currency, Portfolio, PortfolioSnapshot, Position, TransactionType
from .storage import FileBasedStorage


class PortfolioAnalyzer:
    """Pure calculation engine - no data fetching, only calculations."""

    def __init__(self, data_manager: DataProviderManager, storage: FileBasedStorage):
        self.data_manager = data_manager
        self.storage = storage

    def create_snapshot(
        self, portfolio: Portfolio, snapshot_date: date, save: bool = True
    ) -> PortfolioSnapshot:
        """Create snapshot from portfolio state. Prices must already be set!

        This is a PURE calculation - it does NOT fetch any prices.
        """
        # Calculate metrics from current state
        total_value = self._calculate_portfolio_value(portfolio)
        cash_balance = self._calculate_cash_balance(portfolio)
        positions_value = total_value - cash_balance

        total_cost_basis = sum(pos.cost_basis for pos in portfolio.positions.values())
        total_unrealized_pnl = sum(
            pos.unrealized_pnl for pos in portfolio.positions.values()
            if pos.unrealized_pnl
        )
        total_unrealized_pnl_percent = (
            (total_unrealized_pnl / total_cost_basis * 100)
            if total_cost_basis > 0
            else Decimal("0")
        )

        # Deep copy positions
        snapshot_positions = {
            symbol: Position(
                instrument=pos.instrument,
                quantity=pos.quantity,
                average_cost=pos.average_cost,
                current_price=pos.current_price,
                last_updated=pos.last_updated,
            )
            for symbol, pos in portfolio.positions.items()
        }

        # Deep copy cash balances
        snapshot_cash_balances = dict(portfolio.cash_balances)

        snapshot = PortfolioSnapshot(
            date=snapshot_date,
            total_value=total_value,
            cash_balance=cash_balance,
            positions_value=positions_value,
            base_currency=portfolio.base_currency,
            positions=snapshot_positions,
            cash_balances=snapshot_cash_balances,
            total_cost_basis=total_cost_basis,
            total_unrealized_pnl=total_unrealized_pnl,
            total_unrealized_pnl_percent=total_unrealized_pnl_percent,
        )

        if save:
            self.storage.save_snapshot(portfolio.id, snapshot)

        return snapshot

    def create_snapshots_for_range(
        self, portfolio: Portfolio, start_date: date, end_date: date, save: bool = True
    ) -> List[PortfolioSnapshot]:
        """Create historical snapshots by replaying transactions with historical prices."""
        if start_date > end_date:
            raise ValueError("Start date must be before end date")

        # Get all symbols
        symbols = {
            txn.instrument.symbol
            for txn in portfolio.transactions
            if txn.instrument.symbol != "CASH" and txn.timestamp.date() <= end_date
        }

        # Fetch ALL historical prices once
        logging.info(f"Fetching historical prices for {len(symbols)} symbols from {start_date} to {end_date}")
        price_map = self._fetch_historical_prices(symbols, start_date, end_date)

        snapshots = []
        current_date = start_date

        while current_date <= end_date:
            try:
                # 1. Rebuild portfolio for this date
                temp_portfolio = self._rebuild_portfolio_for_date(portfolio, current_date)

                # 2. Set historical prices
                self._apply_historical_prices(temp_portfolio, price_map, current_date)

                # 3. Create snapshot (pure calculation)
                snapshot = self.create_snapshot(temp_portfolio, current_date, save=False)
                snapshots.append(snapshot)

                logging.debug(f"Created snapshot for {current_date}")

            except Exception as e:
                logging.error(f"Failed snapshot for {current_date}: {e}")

            current_date += timedelta(days=1)

        # Batch save
        if save and snapshots:
            self.storage.save_snapshots_batch(portfolio.id, snapshots)
            logging.info(f"Saved {len(snapshots)} snapshots")

        return snapshots

    def get_performance_metrics(self, portfolio: Portfolio, days: int = 365) -> Dict:
        """Calculate performance metrics from historical snapshots."""
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        snapshots = self.storage.load_snapshots(portfolio.id, start_date, end_date)

        if len(snapshots) < 2:
            return {"error": "Insufficient data"}

        first_value = snapshots[0].total_value
        last_value = snapshots[-1].total_value

        total_return = (
            ((last_value - first_value) / first_value * 100)
            if first_value > 0
            else Decimal("0")
        )

        # Volatility
        daily_returns = []
        for i in range(1, len(snapshots)):
            prev = snapshots[i - 1].total_value
            curr = snapshots[i].total_value
            if prev > 0:
                daily_returns.append(float((curr - prev) / prev))

        volatility = Decimal("0")
        if len(daily_returns) > 1:
            import statistics
            volatility = Decimal(str(statistics.stdev(daily_returns) * (252**0.5) * 100))

        return {
            "total_return_percent": total_return,
            "annualized_volatility_percent": volatility,
            "current_value": last_value,
            "period_start_value": first_value,
            "days_analyzed": len(snapshots),
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

    def _rebuild_portfolio_for_date(
        self, original: Portfolio, target_date: date
    ) -> Portfolio:
        """Replay transactions up to target_date to get portfolio state."""
        temp = Portfolio(
            id=original.id,
            name=original.name,
            base_currency=original.base_currency,
            created_at=original.created_at,
            transactions=[],
            positions={},
            cash_balances={},
        )

        # Replay transactions
        for txn in sorted(original.transactions, key=lambda t: t.timestamp):
            if txn.timestamp.date() <= target_date:
                temp.add_transaction(txn)

        return temp

    def _fetch_historical_prices(
        self, symbols: set, start_date: date, end_date: date
    ) -> Dict[str, Dict[date, Decimal]]:
        """Fetch prices for all symbols, forward-fill missing dates."""
        price_map: Dict[str, Dict[date, Decimal]] = {}

        for symbol in symbols:
            try:
                series = self.data_manager.get_historical_prices(symbol, start_date, end_date)

                daily = {}
                for item in series:
                    px = item.close_price or item.open_price or item.high_price or item.low_price
                    if px:
                        daily[item.date] = px

                # Forward-fill
                filled = {}
                current = start_date
                last_price = None

                while current <= end_date:
                    if current in daily:
                        last_price = daily[current]
                        filled[current] = last_price
                    elif last_price:
                        filled[current] = last_price
                    current += timedelta(days=1)

                if filled:
                    price_map[symbol] = filled
                    logging.debug(f"Loaded {len(filled)} prices for {symbol}")

            except Exception as e:
                logging.error(f"Failed to fetch prices for {symbol}: {e}")

        return price_map

    def _apply_historical_prices(
        self, portfolio: Portfolio, price_map: Dict[str, Dict[date, Decimal]], target_date: date
    ) -> None:
        """Apply prices from price_map to positions."""
        for pos in portfolio.positions.values():
            symbol = pos.instrument.symbol

            # Get from map
            if symbol in price_map and target_date in price_map[symbol]:
                pos.current_price = price_map[symbol][target_date]
                pos.last_updated = datetime.combine(target_date, datetime.min.time())
            else:
                # Fallback: fetch current price only for today
                if target_date == date.today():
                    try:
                        price = self.data_manager.get_current_price(symbol)
                        if price:
                            pos.current_price = price
                            pos.last_updated = datetime.now()
                        else:
                            logging.warning(f"No price for {symbol} on {target_date}")
                    except Exception as e:
                        logging.error(f"Failed to get price for {symbol}: {e}")
                else:
                    logging.warning(f"No historical price for {symbol} on {target_date}")
