"""
Portfolio analyzer for calculating metrics and creating snapshots.

SIMPLE RULE: This class only does CALCULATIONS - it NEVER fetches data.
The manager is responsible for fetching prices before calling analyzer methods.
"""

import logging
import statistics
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import pandas as pd

from ..data_providers.manager import DataProviderManager
from .models import Currency, Portfolio, PortfolioSnapshot, Position, TransactionType
from .storage import FileBasedStorage

if TYPE_CHECKING:
    from ..services.market_data_service import MarketDataService


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

        # Try DataFrame approach first (more efficient)
        df = self.storage.load_snapshots_df(portfolio.id, start_date, end_date)

        if df.empty or len(df) < 2:
            return {"error": "Insufficient data"}

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

    def get_performance_metrics_df(
        self, portfolio: Portfolio, days: int = 365
    ) -> pd.DataFrame:
        """Calculate performance metrics and return as DataFrame.

        Returns a DataFrame with daily values and returns.
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        df = self.storage.load_snapshots_df(portfolio.id, start_date, end_date)

        if df.empty:
            return pd.DataFrame()

        # Add calculated columns
        df["daily_return"] = df["total_value"].pct_change()
        df["cumulative_return"] = (1 + df["daily_return"]).cumprod() - 1

        return df

    def calculate_volatility(
        self, portfolio: Portfolio, days: int = 365
    ) -> Optional[Decimal]:
        """Calculate annualized volatility using daily returns."""
        df = self.storage.load_snapshots_df(
            portfolio.id,
            date.today() - timedelta(days=days),
            date.today(),
        )

        if df.empty or len(df) < 2:
            return None

        daily_returns = df["total_value"].pct_change().dropna()
        if len(daily_returns) < 2:
            return None

        vol = daily_returns.std() * (252**0.5) * 100
        return Decimal(str(vol))

    def calculate_max_drawdown(
        self, portfolio: Portfolio, days: int = 365
    ) -> Optional[Decimal]:
        """Calculate maximum drawdown over the period."""
        df = self.storage.load_snapshots_df(
            portfolio.id,
            date.today() - timedelta(days=days),
            date.today(),
        )

        if df.empty or len(df) < 2:
            return None

        # Calculate running maximum and drawdown
        values = df["total_value"]
        running_max = values.expanding().max()
        drawdown = (values - running_max) / running_max * 100

        max_dd = drawdown.min()
        return Decimal(str(max_dd)) if pd.notna(max_dd) else None

    def calculate_sharpe_ratio(
        self,
        portfolio: Portfolio,
        days: int = 365,
        risk_free_rate: float = 0.04,
    ) -> Optional[Decimal]:
        """Calculate Sharpe ratio.

        Args:
            portfolio: The portfolio
            days: Number of days to analyze
            risk_free_rate: Annual risk-free rate (default 4%)

        Returns:
            Sharpe ratio or None if insufficient data
        """
        df = self.storage.load_snapshots_df(
            portfolio.id,
            date.today() - timedelta(days=days),
            date.today(),
        )

        if df.empty or len(df) < 2:
            return None

        daily_returns = df["total_value"].pct_change().dropna()
        if len(daily_returns) < 2:
            return None

        # Annualize returns
        annual_return = daily_returns.mean() * 252
        annual_vol = daily_returns.std() * (252**0.5)

        if annual_vol == 0:
            return None

        sharpe = (annual_return - risk_free_rate) / annual_vol
        return Decimal(str(sharpe))

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
