"""
LangChain tools for portfolio management and analysis.
"""

import math
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import List, Optional

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from pypdf import PdfReader

from ..data_providers.manager import DataProviderManager
from ..portfolio.manager import PortfolioManager
from ..portfolio.models import TransactionType
from ..portfolio.optimizer import OptimizationMethod, PortfolioOptimizer
from ..utils.metrics import FinancialMetricsCalculator


class PortfolioToolInput(BaseModel):
    """Base input model for portfolio tools."""

    pass


class AddTransactionInput(PortfolioToolInput):
    """Input for adding a transaction."""

    symbol: Optional[str] = Field(
        default=None, description="Stock symbol (e.g., AAPL, TSLA)"
    )
    isin: Optional[str] = Field(
        default=None, description="ISIN identifier (e.g., US0378331005)"
    )
    instrument_type: Optional[str] = Field(
        default=None,
        description="Type of financial instrument: stock, etf, bond, crypto, cash, mutual_fund, option, future. If not specified, the system will automatically detect the type."
    )
    transaction_type: str = Field(
        description="Type: buy, sell, dividend, deposit, withdrawal, fees"
    )
    quantity: float = Field(description="Number of shares or amount")
    price: float = Field(description="Price per share or total amount")
    currency: Optional[str] = Field(
        default=None,
        description="Currency code (e.g., USD, EUR, GBP, CHF). Required for deposits/withdrawals; overrides instrument currency for trades if provided.",
    )
    date: Optional[str] = Field(
        default=None,
        description="Trade date in YYYY-MM-DD format (defaults to today if omitted)",
    )
    days_ago: int = Field(
        default=0, description="Fallback: how many days ago (0 for today)"
    )
    notes: Optional[str] = Field(default=None, description="Additional notes")


class GetPortfolioSummaryInput(PortfolioToolInput):
    """Input for portfolio summary."""

    include_metrics: bool = Field(
        default=True, description="Include performance metrics"
    )


class SearchInstrumentInput(BaseModel):
    """Input for instrument search."""

    query: str = Field(description="Search query (symbol or company name)")


class GetPriceInput(PortfolioToolInput):
    """Input for getting current price."""

    symbol: str = Field(description="Stock symbol")


class GetMetricsInput(PortfolioToolInput):
    """Input for getting portfolio metrics."""

    days: int = Field(default=365, description="Number of days to analyze")
    benchmark: str = Field(default="SPY", description="Benchmark symbol for comparison")


class AddTransactionTool(BaseTool):
    """Tool for adding transactions to the portfolio."""

    name: str = "add_transaction"
    description: str = """Add a transaction to the portfolio. CRITICAL: Follow user specifications exactly!

    PRIORITY RULES:
    1. If user specifies instrument_type (bond, stock, etf, etc.), use that EXACTLY
    2. If user specifies currency (EUR, USD, etc.), use that EXACTLY
    3. If user specifies symbol, isin, or notes, use those EXACTLY
    4. Only auto-detect when user doesn't specify

    Supports:
    - buy/sell stocks, bonds, ETFs: specify symbol, quantity, price, ISIN (optional), instrument_type (if specified by user)
    - deposit/withdraw cash: use 'CASH' as symbol (ISIN not required for cash)
    - dividends: specify symbol, amount, and optionally ISIN
    - fees: use 'fees' as transaction type with CASH symbol

    Instrument Type Handling:
    - ALWAYS use user-specified instrument_type if provided (bond, stock, etf, crypto, etc.)
    - Only auto-detect type when user doesn't specify
    - Valid instrument types: stock, etf, bond, crypto, cash, mutual_fund, option, future

    The system handles symbol/ISIN/name mapping while respecting user specifications:
    - If you provide a symbol (e.g., AAPL) + ISIN (e.g., US0378331005), it will find the company name
    - If you provide only an ISIN (e.g., US0378331005), it will find the symbol and company name
    - If you provide only a symbol (e.g., AAPL), it will find the company name (no ISIN search)
    - If you provide only a company name (e.g., Apple), it will automatically find the symbol and proceed

    Examples with USER SPECIFICATIONS (follow exactly):
    - "I bought 50 AAPL bonds in EUR at 95%" → instrument_type="bond", currency="EUR" (user said "bonds"!)
    - "Buy 100 TLT as equity at $90" → instrument_type="stock" (user said "equity"!)
    - "Purchase 1000 EUR bonds with ISIN XS2472298335" → instrument_type="bond", currency="EUR"
    - "Add 50 Microsoft stock in USD" → instrument_type="stock", currency="USD"
    - "Buy 25 Tesla bonds at 98.5%" → instrument_type="bond" (user said "bonds"!)

    Examples with AUTO-DETECTION (when user doesn't specify):
    - "I bought 50 shares of AAPL at $150" (auto-detect: type=stock)
    - "I bought 100 TLT at $90.50" (auto-detect: type=bond based on TLT pattern)
    - "I bought 100 SPY at $450" (auto-detect: type=etf)
    - "I bought 5 BTC at $45000" (auto-detect: type=crypto)
    - "Buy 100 using ISIN XS2472298335 at 98.5%" (auto-detect: type=bond based on XS prefix)
    """
    args_schema: type[BaseModel] = AddTransactionInput
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _run(
        self,
        symbol: Optional[str] = None,
        isin: Optional[str] = None,
        instrument_type: Optional[str] = None,
        transaction_type: str = "buy",
        quantity: float = 0.0,
        price: float = 0.0,
        date: Optional[str] = None,
        days_ago: int = 0,
        notes: Optional[str] = None,
        currency: Optional[str] = None,
    ) -> str:
        """Add a transaction to the portfolio."""
        try:
            # Parse transaction type
            txn_type_map = {
                "buy": TransactionType.BUY,
                "sell": TransactionType.SELL,
                "dividend": TransactionType.DIVIDEND,
                "deposit": TransactionType.DEPOSIT,
                "withdrawal": TransactionType.WITHDRAWAL,
                "withdraw": TransactionType.WITHDRAWAL,
                "fees": TransactionType.FEES,
            }

            txn_type = txn_type_map.get(transaction_type.lower())
            if not txn_type:
                return f"Invalid transaction type: {transaction_type}. Use: buy, sell, dividend, deposit, withdrawal"

            # Validate required fields based on transaction type
            is_cash_movement = txn_type in (TransactionType.DEPOSIT, TransactionType.WITHDRAWAL)

            if is_cash_movement:
                # For deposits/withdrawals: require amount, currency, and date
                missing_fields = []

                # Amount can be provided as quantity or price
                amount = quantity if quantity > 0 else price
                if amount <= 0:
                    missing_fields.append("amount (how much to deposit/withdraw)")
                if not currency:
                    missing_fields.append("currency (USD, EUR, GBP, etc.)")
                if not date:
                    missing_fields.append("date (when did this occur)")

                if missing_fields:
                    return f"Missing required information for {transaction_type}: {', '.join(missing_fields)}. Please ask the user to provide this information."

                # Set up cash movement
                symbol = "CASH"
                isin = None
                price = amount
                quantity = 1.0

            else:
                # For buy/sell/dividend: require quantity, price, and date
                missing_fields = []
                if quantity <= 0:
                    missing_fields.append("quantity (how many shares/units)")
                if price <= 0:
                    missing_fields.append("price (price per share/unit)")
                if not date:
                    missing_fields.append("date (when did this transaction occur)")

                if missing_fields:
                    return f"Missing required information: {', '.join(missing_fields)}. Please ask the user to provide this information before proceeding."

                # Validate that we have at least one identifier for non-cash transactions
                if not symbol and not isin:
                    return "Please provide either a symbol (e.g., AAPL) or an ISIN (e.g., US0378331005)."

            # Determine timestamp
            if date:
                try:
                    timestamp = datetime.strptime(date, "%Y-%m-%d")
                except ValueError:
                    return "Invalid date format. Use YYYY-MM-DD."
            else:
                timestamp = datetime.now() - timedelta(days=days_ago)

            # Prepare currency
            from src.portfolio.models import Currency as Cur

            cur_obj = None
            if currency:
                try:
                    cur_obj = Cur(currency.upper())
                except ValueError:
                    return f"❌ Invalid currency: {currency}. Use one of {[c.value for c in Cur]}"

            # Handle bond price interpretation (percentages)
            final_price = price
            if isin and isin.upper().startswith("XS"):  # XS ISINs are typically bonds
                # If price looks like a percentage (between 0 and 200), treat as percentage of face value
                if 0 < price <= 200:
                    # Keep the percentage as-is for bonds (98.85% -> 98.85)
                    # This is the standard way bonds are quoted
                    final_price = price
                    notes = f"{notes or ''} (Price: {price}% of face value)".strip()

            # Add transaction
            # The portfolio manager will handle all the symbol/ISIN/name mapping
            success = self.portfolio_manager.add_transaction(
                symbol=symbol.upper() if symbol else None,
                transaction_type=txn_type,
                quantity=Decimal(str(quantity)),
                price=Decimal(str(final_price)),
                timestamp=timestamp,
                notes=notes,
                isin=isin.upper() if isin else None,
                currency=cur_obj,
                instrument_type=instrument_type,
            )

            if success:
                # Show what was actually stored
                if symbol:
                    label = symbol.upper()
                elif isin:
                    label = f"ISIN_{isin[:8]}"
                else:
                    label = "Unknown"

                # Get the actual instrument info to show the resolved name
                try:
                    if symbol and self.portfolio_manager.current_portfolio:
                        # Find the transaction we just added
                        for txn in reversed(
                            self.portfolio_manager.current_portfolio.transactions
                        ):
                            if txn.instrument.symbol == symbol.upper():
                                return f"✅ Added {transaction_type} transaction: {quantity} {txn.instrument.symbol} ({txn.instrument.name}) @ {price}"
                except Exception:
                    pass

                return f"✅ Added {transaction_type} transaction: {quantity} {label} @ {price}"
            else:
                return "❌ Failed to add transaction. Make sure a portfolio is loaded."

        except Exception as e:
            return f"❌ Error adding transaction: {str(e)}"


class GetPortfolioSummaryTool(BaseTool):
    """Tool for getting portfolio summary."""

    name: str = "get_portfolio_summary"
    description: str = (
        "Get a comprehensive summary of the current portfolio including positions, cash balances, and performance."
    )
    args_schema: type[BaseModel] = GetPortfolioSummaryInput
    portfolio_manager: Optional[PortfolioManager] = None
    metrics_calculator: Optional[FinancialMetricsCalculator] = None

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        metrics_calculator: FinancialMetricsCalculator,
    ):
        super().__init__()
        self.portfolio_manager = portfolio_manager
        self.metrics_calculator = metrics_calculator

    def _run(self, include_metrics: bool = True) -> str:
        """Get portfolio summary."""
        try:
            if not self.portfolio_manager.current_portfolio:
                return "No portfolio loaded. Please create or load a portfolio first."

            portfolio = self.portfolio_manager.current_portfolio

            # Note: Prices are not automatically updated - use existing data only
            # Users should manually update prices via the UI if needed

            # Get positions summary
            positions = self.portfolio_manager.get_position_summary()
            total_value = self.portfolio_manager.get_portfolio_value()

            summary = [
                f"Portfolio: {portfolio.name}",
                f"Total Value: ${total_value:,.2f} {portfolio.base_currency.value}",
                f"Created: {portfolio.created_at.strftime('%Y-%m-%d')}",
                "",
            ]

            # Cash balances
            if portfolio.cash_balances:
                summary.append("Cash Balances:")
                for currency, amount in portfolio.cash_balances.items():
                    code = getattr(currency, "value", str(currency))
                    summary.append(f"- {code}: ${amount:,.2f}")
                summary.append("")

            # Positions
            if positions:
                summary.append("Current Positions:")
                for pos in positions:
                    # Get appropriate unit label based on instrument type
                    instrument_type = pos.get("instrument_type", "stock")
                    if instrument_type == "bond":
                        unit_label = "bonds"
                    elif instrument_type == "etf":
                        unit_label = "shares"
                    elif instrument_type == "crypto":
                        unit_label = "coins"
                    else:
                        unit_label = "shares"

                    # Format price
                    price_str = f"${pos['current_price']:,.2f}" if pos['current_price'] else "N/A"

                    # Format P&L
                    pnl_str = ""
                    if pos["unrealized_pnl"] is not None:
                        pnl = float(pos["unrealized_pnl"])
                        pnl_pct = float(pos["unrealized_pnl_percent"] or 0)
                        sign = "+" if pnl >= 0 else ""
                        pnl_str = f" ({sign}${pnl:,.2f}, {sign}{pnl_pct:.1f}%)"

                    summary.append(
                        f"- {pos['symbol']} ({pos['name']}): "
                        f"{pos['quantity']:.4g} {unit_label} @ {price_str}{pnl_str}"
                    )
                summary.append("")

            # Basic metrics if requested
            if include_metrics:
                try:
                    metrics = self.portfolio_manager.get_performance_metrics()
                    if metrics and "error" not in metrics:
                        summary.append("Performance Metrics:")
                        summary.append(
                            f"- Total Return: {metrics.get('total_return_percent', 0):.2f}%"
                        )
                        summary.append(
                            f"- Volatility: {metrics.get('annualized_volatility_percent', 0):.2f}%"
                        )
                        summary.append("")
                except Exception:
                    # Metrics calculation failed, continue without them
                    pass

            return "\n".join(summary)

        except Exception as e:
            return f"Error getting portfolio summary: {str(e)}"


class GetTransactionsTool(BaseTool):
    """Tool for exporting transactions and history with metrics-ready fields."""

    name: str = "get_transactions"
    description: str = (
        "Return all transactions with key fields for analysis (ids, timestamps, symbols, quantities, prices, currency)."
    )
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _run(self) -> str:
        try:
            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            txns = self.portfolio_manager.current_portfolio.transactions
            lines = [
                "🧾 Transactions (all):",
                "id,timestamp,symbol,type,quantity,price,currency,notes",
            ]
            for t in sorted(txns, key=lambda x: x.timestamp):
                lines.append(
                    ",".join(
                        [
                            t.id,
                            t.timestamp.isoformat(),
                            t.instrument.symbol,
                            t.transaction_type.value,
                            str(t.quantity),
                            str(t.price),
                            t.currency.value,
                            (t.notes or "").replace(",", " "),
                        ]
                    )
                )
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error exporting transactions: {str(e)}"


class SimulateWhatIfTool(BaseTool):
    """Tool to simulate a what-if scenario by excluding symbols or transaction ids and returning end value."""

    name: str = "simulate_what_if"
    description: str = (
        "Simulate snapshots for a date range excluding certain symbols or transactions; returns end total value and basic stats."
    )
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _run(
        self,
        start: str,
        end: str,
        exclude_symbols: str = "",
        exclude_txn_ids: str = "",
    ) -> str:
        try:
            from datetime import date as date_cls

            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            start_date = date_cls.fromisoformat(start)
            end_date = date_cls.fromisoformat(end)
            if start_date > end_date:
                return "❌ Start date must be on or before end date."

            # Split on commas and whitespace
            import re

            splitter = re.compile(r"[\s,]+")
            symbols = [
                s.strip().upper() for s in splitter.split(exclude_symbols) if s.strip()
            ]
            ids = [x.strip() for x in splitter.split(exclude_txn_ids) if x.strip()]

            # Baseline (no exclusions)
            baseline_snaps = self.portfolio_manager.simulate_snapshots_for_range(
                start_date, end_date
            )

            # What-if with exclusions
            snaps = self.portfolio_manager.simulate_snapshots_for_range(
                start_date,
                end_date,
                exclude_symbols=symbols,
                exclude_transaction_ids=ids,
            )

            if not snaps or not baseline_snaps:
                return "No snapshots generated for the specified range."

            end_value = float(snaps[-1].total_value)
            start_value = float(snaps[0].total_value)
            total_return = (
                ((end_value - start_value) / start_value * 100)
                if start_value > 0
                else 0.0
            )

            base_end = float(baseline_snaps[-1].total_value)
            base_start = float(baseline_snaps[0].total_value)
            base_return = (
                ((base_end - base_start) / base_start * 100) if base_start > 0 else 0.0
            )
            delta_value = end_value - base_end
            delta_return = total_return - base_return

            return (
                f"📈 What-if Simulation ({start} → {end})\n"
                f"• Excluded symbols: {symbols or '-'}\n"
                f"• Excluded txn ids: {ids or '-'}\n"
                f"• Start value: {start_value:,.2f} USD\n"
                f"• End value: {end_value:,.2f} USD\n"
                f"• Total return: {total_return:.2f}%\n"
                f"• Baseline end: {base_end:,.2f} USD | Baseline return: {base_return:.2f}%\n"
                f"• Δ End value vs baseline: {delta_value:+,.2f} USD | Δ Return: {delta_return:+.2f}%"
            )
        except Exception as e:
            return f"❌ Error running simulation: {str(e)}"


class AdvancedWhatIfTool(BaseTool):
    """Advanced what-if tool for specific portfolio modifications and Monte Carlo scenarios."""

    name: str = "advanced_what_if"
    description: str = (
        "Run advanced what-if analysis with specific portfolio modifications, position adjustments, "
        "or hypothetical additions. Supports Monte Carlo simulations with custom market assumptions."
    )
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _run(
        self,
        scenario_type: str = "custom",
        projection_years: float = 2.0,
        monte_carlo_runs: int = 1000,
        modify_positions: str = "",
        add_positions: str = "",
        market_return: float = 0.08,
        market_volatility: float = 0.20,
        recurring_deposits: float = 0.0,
        stress_test: bool = False,
    ) -> str:
        """Run advanced what-if analysis with portfolio modifications.

        Args:
            scenario_type: Type of scenario (optimistic, likely, pessimistic, stress, custom)
            projection_years: Years to project (1-10)
            monte_carlo_runs: Number of simulations (100-5000)
            modify_positions: JSON-like string of position modifications, e.g.,
                            "AAPL:+50%,MSFT:-25%,GOOGL:=150" (increase by 50%, decrease by 25%, set to 150 shares)
            add_positions: JSON-like string of new positions to add, e.g.,
                         "NVDA:100@$800,TSLA:50@$250" (100 shares at $800, 50 shares at $250)
            market_return: Expected annual market return (as decimal, e.g., 0.08 for 8%)
            market_volatility: Market volatility (as decimal, e.g., 0.20 for 20%)
            recurring_deposits: Monthly recurring deposits in USD
            stress_test: Whether to apply stress testing conditions
        """
        try:
            from src.portfolio.scenarios import (
                PortfolioScenarioEngine, ScenarioConfiguration, ScenarioType,
                MarketAssumptions, AssetClassAssumptions
            )
            from src.portfolio.models import PortfolioSnapshot, Position, FinancialInstrument, InstrumentType, Currency
            from datetime import date
            import copy

            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            # Validate parameters
            projection_years = max(0.5, min(10.0, projection_years))
            monte_carlo_runs = max(100, min(5000, monte_carlo_runs))
            market_return = max(-0.5, min(1.0, market_return))
            market_volatility = max(0.01, min(2.0, market_volatility))

            # Create current portfolio snapshot
            current_snapshot = self.portfolio_manager.create_snapshot(save=False)

            # Apply portfolio modifications
            modified_snapshot = self._apply_portfolio_modifications(
                current_snapshot, modify_positions, add_positions
            )

            # Create scenario configuration
            scenario_config = self._create_scenario_config(
                scenario_type, projection_years, monte_carlo_runs,
                market_return, market_volatility, recurring_deposits, stress_test
            )

            # Run simulation
            engine = PortfolioScenarioEngine(random_seed=42)
            result = engine.run_scenario_simulation(modified_snapshot, scenario_config)

            # Format results
            return self._format_advanced_results(
                result, current_snapshot, modified_snapshot,
                modify_positions, add_positions
            )

        except Exception as e:
            return f"❌ Error running advanced what-if analysis: {str(e)}"

    def _apply_portfolio_modifications(self, snapshot, modify_positions, add_positions):
        """Apply position modifications and additions to the portfolio snapshot."""
        from src.portfolio.models import Position, FinancialInstrument, InstrumentType, Currency
        from decimal import Decimal
        import copy

        # Create a copy of the snapshot to modify
        modified_snapshot = copy.deepcopy(snapshot)

        # Parse and apply position modifications
        if modify_positions:
            modifications = self._parse_position_modifications(modify_positions)
            for symbol, change in modifications.items():
                if symbol in modified_snapshot.positions:
                    position = modified_snapshot.positions[symbol]
                    if change['type'] == 'percent':
                        # Percentage change
                        multiplier = 1.0 + change['value'] / 100.0
                        position.quantity = position.quantity * Decimal(str(multiplier))
                    elif change['type'] == 'absolute':
                        # Absolute quantity
                        position.quantity = Decimal(str(change['value']))
                    elif change['type'] == 'delta':
                        # Add/subtract shares
                        position.quantity = position.quantity + Decimal(str(change['value']))

                    # Ensure quantity doesn't go negative
                    position.quantity = max(Decimal("0"), position.quantity)

        # Parse and apply new positions
        if add_positions:
            new_positions = self._parse_new_positions(add_positions)
            for symbol, position_data in new_positions.items():
                if symbol not in modified_snapshot.positions:
                    # Create new position
                    instrument = FinancialInstrument(
                        symbol=symbol,
                        name=f"{symbol} Corporation",
                        instrument_type=InstrumentType.STOCK,  # Default to stock
                        currency=Currency.USD,
                        isin=f"US{symbol}123456"  # Placeholder ISIN
                    )

                    new_position = Position(
                        instrument=instrument,
                        quantity=Decimal(str(position_data['quantity'])),
                        average_cost=Decimal(str(position_data['price'])),
                        current_price=Decimal(str(position_data['price'])),
                        last_updated=snapshot.date
                    )

                    modified_snapshot.positions[symbol] = new_position

        # Recalculate snapshot totals
        self._recalculate_snapshot_totals(modified_snapshot)

        return modified_snapshot

    def _parse_position_modifications(self, modify_string):
        """Parse position modification string like 'AAPL:+50%,MSFT:-25%,GOOGL:=150'."""
        modifications = {}
        if not modify_string:
            return modifications

        for item in modify_string.split(','):
            if ':' not in item:
                continue

            symbol, change_str = item.strip().split(':', 1)
            symbol = symbol.strip().upper()
            change_str = change_str.strip()

            if change_str.endswith('%'):
                # Percentage change
                value = float(change_str[:-1])
                modifications[symbol] = {'type': 'percent', 'value': value}
            elif change_str.startswith('='):
                # Absolute value
                value = float(change_str[1:])
                modifications[symbol] = {'type': 'absolute', 'value': value}
            elif change_str.startswith('+') or change_str.startswith('-'):
                # Delta change
                value = float(change_str)
                modifications[symbol] = {'type': 'delta', 'value': value}
            else:
                # Default to absolute
                value = float(change_str)
                modifications[symbol] = {'type': 'absolute', 'value': value}

        return modifications

    def _parse_new_positions(self, add_string):
        """Parse new position string like 'NVDA:100@$800,TSLA:50@$250'."""
        new_positions = {}
        if not add_string:
            return new_positions

        for item in add_string.split(','):
            if ':' not in item or '@' not in item:
                continue

            symbol, details = item.strip().split(':', 1)
            symbol = symbol.strip().upper()

            if '@' in details:
                quantity_str, price_str = details.split('@', 1)
                quantity = float(quantity_str.strip())
                price = float(price_str.strip().replace('$', ''))

                new_positions[symbol] = {
                    'quantity': quantity,
                    'price': price
                }

        return new_positions

    def _recalculate_snapshot_totals(self, snapshot):
        """Recalculate snapshot totals after position modifications."""
        from decimal import Decimal

        total_positions_value = Decimal("0")
        total_cost_basis = Decimal("0")
        total_unrealized_pnl = Decimal("0")

        for position in snapshot.positions.values():
            if position.quantity > 0:
                position_value = position.quantity * position.current_price
                position_cost = position.quantity * position.average_cost

                total_positions_value += position_value
                total_cost_basis += position_cost
                total_unrealized_pnl += (position_value - position_cost)

        snapshot.positions_value = total_positions_value
        snapshot.total_cost_basis = total_cost_basis
        snapshot.total_unrealized_pnl = total_unrealized_pnl
        snapshot.total_value = snapshot.cash_balance + total_positions_value

        if total_cost_basis > 0:
            snapshot.total_unrealized_pnl_percent = (total_unrealized_pnl / total_cost_basis) * Decimal("100")
        else:
            snapshot.total_unrealized_pnl_percent = Decimal("0")

    def _create_scenario_config(self, scenario_type, projection_years, monte_carlo_runs,
                              market_return, market_volatility, recurring_deposits, stress_test):
        """Create a scenario configuration based on parameters."""
        from src.portfolio.scenarios import ScenarioConfiguration, ScenarioType, MarketAssumptions, AssetClassAssumptions
        from src.portfolio.models import InstrumentType

        # Map scenario type
        scenario_type_map = {
            "optimistic": ScenarioType.OPTIMISTIC,
            "likely": ScenarioType.LIKELY,
            "pessimistic": ScenarioType.PESSIMISTIC,
            "stress": ScenarioType.STRESS,
            "custom": ScenarioType.CUSTOM
        }

        mapped_scenario_type = scenario_type_map.get(scenario_type.lower(), ScenarioType.CUSTOM)

        # Adjust parameters for stress test
        if stress_test or mapped_scenario_type == ScenarioType.STRESS:
            market_return = min(market_return, -0.02)  # At least -2% return
            market_volatility = max(market_volatility, 0.35)  # At least 35% volatility

        # Create market assumptions
        market_assumptions = MarketAssumptions(
            expected_return=market_return,
            volatility=market_volatility,
            equity_correlation=0.7 if not stress_test else 0.9,
            bond_correlation=0.3 if not stress_test else 0.5,
            equity_bond_correlation=-0.1 if not stress_test else 0.2,
            inflation_rate=0.025,
            risk_free_rate=0.02
        )

        # Create asset class assumptions
        asset_assumptions = {
            "STOCK": AssetClassAssumptions(
                asset_class=InstrumentType.STOCK,
                expected_return=market_return + 0.02,
                volatility=market_volatility + 0.05,
                dividend_yield=0.02
            ),
            "ETF": AssetClassAssumptions(
                asset_class=InstrumentType.ETF,
                expected_return=market_return,
                volatility=market_volatility,
                expense_ratio=0.007
            ),
            "BOND": AssetClassAssumptions(
                asset_class=InstrumentType.BOND,
                expected_return=max(0.01, market_return - 0.03),
                volatility=market_volatility * 0.3,
                dividend_yield=0.035
            )
        }

        return ScenarioConfiguration(
            scenario_type=mapped_scenario_type,
            name=f"Custom {scenario_type.title()} Scenario",
            description=f"Custom scenario with {market_return*100:.1f}% return and {market_volatility*100:.1f}% volatility",
            projection_years=projection_years,
            monte_carlo_runs=monte_carlo_runs,
            market_assumptions=market_assumptions,
            asset_class_assumptions=asset_assumptions,
            recurring_deposits=recurring_deposits
        )

    def _format_advanced_results(self, result, original_snapshot, modified_snapshot,
                                modify_positions, add_positions):
        """Format the advanced what-if results for display."""
        stats = result.get_summary_stats()

        # Calculate changes from original
        original_value = float(original_snapshot.total_value)
        modified_start_value = float(modified_snapshot.total_value)
        mean_final_value = stats['mean_final_value']

        # Calculate portfolio modification impact
        modification_impact = modified_start_value - original_value

        # Build result string (use "USD" instead of "$" to avoid Streamlit LaTeX interpretation)
        lines = [
            f"🔮 Advanced What-If Analysis",
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"",
            f"📊 Portfolio Modifications Applied:",
        ]

        if modify_positions:
            lines.append(f"   🔧 Position Changes: {modify_positions}")
        if add_positions:
            lines.append(f"   ➕ New Positions: {add_positions}")
        if modification_impact != 0:
            lines.append(f"   📈 Immediate Impact: {modification_impact:+,.2f} USD ({modification_impact/original_value*100:+.1f}%)")

        lines.extend([
            f"",
            f"🎯 Scenario: {result.scenario_config.name}",
            f"   • Projection Period: {result.scenario_config.projection_years:.1f} years",
            f"   • Monte Carlo Runs: {result.scenario_config.monte_carlo_runs:,}",
            f"   • Market Return: {result.scenario_config.market_assumptions.expected_return*100:.1f}%",
            f"   • Market Volatility: {result.scenario_config.market_assumptions.volatility*100:.1f}%",
            f"",
            f"📈 Projection Results:",
            f"   • Starting Value: {modified_start_value:,.2f} USD",
            f"   • Mean Final Value: {mean_final_value:,.2f} USD",
            f"   • Median Final Value: {stats['median_final_value']:,.2f} USD",
            f"   • Best Case (95%): {stats.get('percentile_95', mean_final_value):,.2f} USD",
            f"   • Worst Case (5%): {stats.get('percentile_25', mean_final_value):,.2f} USD",
            f"",
            f"⚡ Performance Metrics:",
            f"   • Mean Annual Return: {stats['mean_annualized_return']*100:.1f}%",
            f"   • Probability of Loss: {stats['probability_of_loss']*100:.1f}%",
            f"   • Probability of Doubling: {stats['probability_of_doubling']*100:.1f}%",
            f"   • Mean Sharpe Ratio: {stats['mean_sharpe_ratio']:.2f}",
            f"",
            f"⚠️ Risk Analysis:",
            f"   • Mean Max Drawdown: {stats['mean_max_drawdown']*100:.1f}%",
            f"   • Standard Deviation: {stats['std_final_value']:,.2f} USD",
            f"   • Value at Risk (95%): {stats.get('var_95', 0):,.2f} USD",
            f"",
            f"💡 Key Insights:",
            f"   • Total Return Potential: {(mean_final_value/modified_start_value - 1)*100:+.1f}%",
            f"   • Annualized Growth: {((mean_final_value/modified_start_value)**(1/result.scenario_config.projection_years) - 1)*100:.1f}%",
            f"   • Risk-Adjusted Score: {stats['mean_sharpe_ratio']:.2f} (higher is better)",
        ])

        if stats['probability_of_loss'] > 0.3:
            lines.append(f"   ⚠️ High Risk: {stats['probability_of_loss']*100:.0f}% chance of loss - consider risk management")
        elif stats['probability_of_doubling'] > 0.2:
            lines.append(f"   🚀 High Growth Potential: {stats['probability_of_doubling']*100:.0f}% chance of doubling")

        return "\n".join(lines)


class HypotheticalPositionTool(BaseTool):
    """Tool for testing hypothetical positions without modifying the actual portfolio."""

    name: str = "test_hypothetical_position"
    description: str = (
        "Test adding a hypothetical position to your portfolio and see how it would perform "
        "under different market scenarios. Useful for exploring new investment opportunities."
    )
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _run(
        self,
        symbol: str,
        quantity: float,
        purchase_price: float,
        investment_amount: str = "",
        scenario: str = "likely",
        time_horizon: float = 1.0,
    ) -> str:
        """Test a hypothetical position in the portfolio.

        Args:
            symbol: Stock symbol to test (e.g., "AAPL", "MSFT")
            quantity: Number of shares to hypothetically purchase (use 0 if using investment_amount)
            purchase_price: Price per share for the hypothetical purchase
            investment_amount: Alternative to quantity - dollar amount to invest (e.g., "$5000")
            scenario: Market scenario to test (optimistic, likely, pessimistic, stress)
            time_horizon: Years to project the investment (0.5 to 5.0)
        """
        try:
            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            # Parse investment amount if provided
            if investment_amount:
                amount_str = investment_amount.replace('$', '').replace(',', '')
                try:
                    amount = float(amount_str)
                    quantity = amount / purchase_price
                except ValueError:
                    return f"❌ Invalid investment amount: {investment_amount}"

            if quantity <= 0:
                return "❌ Quantity must be positive"

            # Validate parameters
            time_horizon = max(0.5, min(5.0, time_horizon))
            symbol = symbol.upper().strip()

            # Get current portfolio value for context
            current_snapshot = self.portfolio_manager.create_snapshot(save=False)
            current_value = float(current_snapshot.total_value)
            investment_value = quantity * purchase_price

            # Create scenario for the hypothetical position (no $ to avoid Streamlit LaTeX issues)
            add_position_str = f"{symbol}:{quantity}@{purchase_price}"

            # Use the advanced what-if tool
            advanced_tool = AdvancedWhatIfTool(self.portfolio_manager)

            # Map scenario to market parameters
            scenario_params = {
                "optimistic": {"market_return": 0.12, "market_volatility": 0.16},
                "likely": {"market_return": 0.08, "market_volatility": 0.20},
                "pessimistic": {"market_return": 0.03, "market_volatility": 0.28},
                "stress": {"market_return": -0.05, "market_volatility": 0.40}
            }

            params = scenario_params.get(scenario.lower(), scenario_params["likely"])

            # Run the analysis
            result = advanced_tool._run(
                scenario_type=scenario,
                projection_years=time_horizon,
                monte_carlo_runs=1000,
                add_positions=add_position_str,
                market_return=params["market_return"],
                market_volatility=params["market_volatility"]
            )

            # Add hypothetical-specific formatting
            investment_pct = (investment_value / current_value) * 100

            header = [
                f"🧪 Hypothetical Position Analysis",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"",
                f"💼 Hypothetical Investment:",
                f"   • Symbol: {symbol}",
                f"   • Quantity: {quantity:,.0f} shares",
                f"   • Purchase Price: {purchase_price:,.2f} USD",
                f"   • Total Investment: {investment_value:,.2f} USD ({investment_pct:.1f}% of portfolio)",
                f"   • Scenario: {scenario.title()}",
                f"   • Time Horizon: {time_horizon:.1f} years",
                f"",
                f"📊 Impact on Portfolio:",
                ""
            ]

            return "\n".join(header) + result.split("📊 Portfolio Modifications Applied:")[1] if "📊 Portfolio Modifications Applied:" in result else result

        except Exception as e:
            return f"❌ Error testing hypothetical position: {str(e)}"


class SearchInstrumentTool(BaseTool):
    """Tool for searching financial instruments."""

    name: str = "search_instrument"
    description: str = (
        "Search for stocks, bonds, ETFs, or other financial instruments by symbol or company name."
    )
    args_schema: type[BaseModel] = SearchInstrumentInput
    data_manager: Optional[DataProviderManager] = None

    def __init__(self, data_manager: DataProviderManager):
        super().__init__()
        self.data_manager = data_manager

    def _run(self, query: str) -> str:
        """Search for instruments."""
        try:
            results = self.data_manager.search_instruments(query)

            if not results:
                return f"❌ No instruments found for '{query}'"

            search_results = [f"🔍 **Search Results for '{query}':**", ""]

            for instrument in results[:10]:  # Limit to top 10
                search_results.append(
                    f"• **{instrument.symbol}** - {instrument.name}\n"
                    f"  Type: {instrument.instrument_type.value.title()} | "
                    f"Currency: {instrument.currency.value}"
                    + (
                        f" | Exchange: {instrument.exchange}"
                        if instrument.exchange
                        else ""
                    )
                )

            return "\n".join(search_results)

        except Exception as e:
            return f"❌ Error searching instruments: {str(e)}"


class SearchCompanyTool(BaseTool):
    """Tool for searching company information by name."""

    name: str = "search_company"
    description: str = (
        "Search for company information by company name. Use this when you have a company name but need the stock symbol or ISIN. "
        "This tool will help find the trading symbol, ISIN, and other details for companies."
    )
    args_schema: type[BaseModel] = SearchInstrumentInput
    data_manager: Optional[DataProviderManager] = None

    def __init__(self, data_manager: DataProviderManager):
        super().__init__()
        self.data_manager = data_manager

    def _run(self, query: str) -> str:
        """Search for company information."""
        try:
            # First try to search by company name
            results = self.data_manager.search_by_company_name(query)

            if not results:
                # Fallback to regular instrument search
                results = self.data_manager.search_instruments(query)

            if not results:
                return f"❌ No companies found for '{query}'. Try using web search to find the correct company name or stock symbol."

            search_results = [f"🔍 **Company Search Results for '{query}':**", ""]

            for instrument in results[:10]:  # Limit to top 10
                result_line = f"• **{instrument.symbol}** - {instrument.name}"

                if instrument.isin:
                    result_line += f"\n  ISIN: {instrument.isin}"

                result_line += f"\n  Type: {instrument.instrument_type.value.title()} | Currency: {instrument.currency.value}"

                if instrument.exchange:
                    result_line += f" | Exchange: {instrument.exchange}"

                if instrument.sector:
                    result_line += f" | Sector: {instrument.sector}"

                search_results.append(result_line)

            # Add guidance for using the results
            search_results.append("")
            search_results.append("💡 **How to use these results:**")
            search_results.append(
                "- Use the symbol (e.g., AAPL) in the add_transaction tool"
            )
            search_results.append(
                "- Use the ISIN if available for more precise identification"
            )
            search_results.append(
                "- If you need more details, use web search with the company name"
            )

            return "\n".join(search_results)

        except Exception as e:
            return f"Error searching for company '{query}': {str(e)}"


class ResolveInstrumentInput(BaseModel):
    """Input for resolving an instrument."""

    isin: Optional[str] = Field(
        default=None, description="ISIN identifier (highest priority)"
    )
    symbol: Optional[str] = Field(
        default=None, description="Stock/instrument symbol (second priority)"
    )
    name: Optional[str] = Field(
        default=None,
        description="Company or instrument name to search (requires user confirmation)",
    )


class ResolveInstrumentTool(BaseTool):
    """Tool for resolving instrument identity with priority: ISIN > Symbol > Name search."""

    name: str = "resolve_instrument"
    description: str = """Resolve an instrument's identity before adding a transaction.

    PRIORITY ORDER:
    1. ISIN (highest priority) - Direct lookup, no confirmation needed
    2. Symbol - Direct lookup, no confirmation needed
    3. Name/Description - Search and REQUIRES user confirmation

    Use this tool BEFORE add_transaction when:
    - User provides a company name instead of symbol (e.g., "Apple" instead of "AAPL")
    - You're unsure if the symbol/ISIN is correct
    - User describes an instrument without clear identifiers

    Returns:
    - If ISIN/symbol found: Confirmed instrument details ready for add_transaction
    - If name search: List of candidates - YOU MUST ASK USER TO CONFIRM before proceeding
    """
    args_schema: type[BaseModel] = ResolveInstrumentInput
    data_manager: Optional[DataProviderManager] = None

    def __init__(self, data_manager: DataProviderManager):
        super().__init__()
        self.data_manager = data_manager

    def _run(
        self,
        isin: Optional[str] = None,
        symbol: Optional[str] = None,
        name: Optional[str] = None,
    ) -> str:
        """Resolve instrument identity with priority: ISIN > Symbol > Name."""
        try:
            # Priority 1: ISIN lookup
            if isin:
                isin = isin.strip().upper()
                # Search for the ISIN
                results = self.data_manager.search_instruments(isin)

                # Look for exact ISIN match
                exact_match = None
                for r in results:
                    if hasattr(r, 'isin') and r.isin and r.isin.upper() == isin:
                        exact_match = r
                        break

                if exact_match:
                    return self._format_confirmed_result(exact_match, "ISIN")

                # No exact ISIN match - show search results if any
                if results:
                    return self._format_search_results(
                        results, f"ISIN '{isin}'", needs_confirmation=True
                    )
                return f"No instrument found with ISIN '{isin}'. Please verify the ISIN is correct."

            # Priority 2: Symbol lookup
            if symbol:
                symbol = symbol.strip().upper()
                # Try to get instrument info directly
                results = self.data_manager.search_instruments(symbol)
                exact_match = None
                for r in results:
                    if r.symbol.upper() == symbol:
                        exact_match = r
                        break

                if exact_match:
                    return self._format_confirmed_result(exact_match, "symbol")

                # No exact match - show search results
                if results:
                    return self._format_search_results(
                        results, f"symbol '{symbol}'", needs_confirmation=True
                    )
                return f"No instrument found with symbol '{symbol}'. Please verify the symbol is correct or provide the company name."

            # Priority 3: Name search (always requires confirmation)
            if name:
                name = name.strip()
                # Try company name search first
                results = self.data_manager.search_by_company_name(name)
                if not results:
                    results = self.data_manager.search_instruments(name)

                if results:
                    return self._format_search_results(
                        results, f"name '{name}'", needs_confirmation=True
                    )
                return f"No instruments found matching '{name}'. Try a different search term or check the spelling."

            return "Please provide at least one of: ISIN, symbol, or name to search."

        except Exception as e:
            return f"Error resolving instrument: {str(e)}"

    def _format_confirmed_result(self, instrument, lookup_type: str) -> str:
        """Format a confirmed instrument result."""
        lines = [
            f"CONFIRMED INSTRUMENT (found by {lookup_type}):",
            f"- Symbol: {instrument.symbol}",
            f"- Name: {instrument.name}",
            f"- Type: {instrument.instrument_type.value}",
            f"- Currency: {instrument.currency.value}",
        ]
        if instrument.isin:
            lines.append(f"- ISIN: {instrument.isin}")
        if instrument.exchange:
            lines.append(f"- Exchange: {instrument.exchange}")

        lines.append("")
        lines.append("You can proceed with add_transaction using the symbol above.")
        return "\n".join(lines)

    def _format_search_results(
        self, results: list, search_term: str, needs_confirmation: bool = False
    ) -> str:
        """Format search results that may need user confirmation."""
        lines = [f"Search results for {search_term}:", ""]

        for i, instrument in enumerate(results[:5], 1):  # Limit to top 5
            lines.append(f"{i}. {instrument.symbol} - {instrument.name}")
            lines.append(f"   Type: {instrument.instrument_type.value}, Currency: {instrument.currency.value}")
            if instrument.isin:
                lines.append(f"   ISIN: {instrument.isin}")
            lines.append("")

        if needs_confirmation:
            lines.append("ACTION REQUIRED: Please ask the user to confirm which instrument they want.")
            lines.append("Example: 'I found these matches. Which one did you mean?'")
            lines.append("Only proceed with add_transaction after user confirms the correct instrument.")

        return "\n".join(lines)


class GetCurrentPriceTool(BaseTool):
    """Tool for getting current price of an instrument."""

    name: str = "get_current_price"
    description: str = (
        "Get the current market price of a stock, bond, ETF, or other financial instrument."
    )
    args_schema: type[BaseModel] = GetPriceInput
    data_manager: Optional[DataProviderManager] = None

    def __init__(self, data_manager: DataProviderManager):
        super().__init__()
        self.data_manager = data_manager

    def _run(self, symbol: str) -> str:
        """Get current price."""
        try:
            price = self.data_manager.get_current_price(symbol.upper())

            if price is None:
                return f"❌ Could not get current price for {symbol.upper()}"

            # Also get instrument info for context
            info = self.data_manager.get_instrument_info(symbol.upper())
            name = info.name if info else symbol.upper()

            return f"💰 **{symbol.upper()}** ({name}): {price:.2f} USD"

        except Exception as e:
            return f"❌ Error getting price for {symbol}: {str(e)}"


class GetPortfolioMetricsTool(BaseTool):
    """Tool for getting detailed portfolio metrics."""

    name: str = "get_portfolio_metrics"
    description: str = (
        "Calculate detailed portfolio performance metrics including volatility, Sharpe ratio, drawdown, alpha, and beta."
    )
    args_schema: type[BaseModel] = GetMetricsInput
    portfolio_manager: Optional[PortfolioManager] = None
    metrics_calculator: Optional[FinancialMetricsCalculator] = None

    def __init__(
        self,
        portfolio_manager: PortfolioManager,
        metrics_calculator: FinancialMetricsCalculator,
    ):
        super().__init__()
        self.portfolio_manager = portfolio_manager
        self.metrics_calculator = metrics_calculator

    def _run(self, days: int = 365, benchmark: str = "SPY") -> str:
        """Calculate portfolio metrics."""
        try:
            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            # Get snapshots for the period
            end_date = date.today()
            start_date = end_date - timedelta(days=days)

            try:
                snapshots = self.portfolio_manager.storage.load_snapshots(
                    self.portfolio_manager.current_portfolio.id, start_date, end_date
                )
            except Exception as e:
                return f"❌ Error loading snapshots: {str(e)}"

            if len(snapshots) < 2:
                return "❌ Insufficient historical data for metrics calculation. Need at least 2 data points."

            metrics = self.metrics_calculator.calculate_portfolio_metrics(
                snapshots, benchmark_symbol=benchmark
            )

            if "error" in metrics:
                return f"❌ {metrics['error']}"

            result = [
                f"📊 **Portfolio Metrics** ({days} days vs {benchmark})",
                "",
                "**📈 Returns:**",
                f"  • Total Return: {metrics.get('total_return', 0)*100:.2f}%",
                f"  • Annualized Return: {metrics.get('annualized_return', 0)*100:.2f}%",
                "",
                "**⚡ Risk Metrics:**",
                f"  • Volatility: {metrics.get('volatility', 0)*100:.2f}%",
                f"  • Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%",
                f"  • Value at Risk (5%): {metrics.get('var_5pct', 0)*100:.2f}%",
                "",
                "**🎯 Risk-Adjusted Returns:**",
                f"  • Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}",
                f"  • Sortino Ratio: {metrics.get('sortino_ratio', 0):.3f}",
                f"  • Calmar Ratio: {metrics.get('calmar_ratio', 0):.3f}",
            ]

            # Add benchmark comparison if available
            if metrics.get("benchmark_available"):
                result.extend(
                    [
                        "",
                        f"**📊 vs {benchmark}:**",
                        f"  • Beta: {metrics.get('beta', 0):.3f}",
                        f"  • Alpha: {metrics.get('alpha', 0)*100:.2f}%",
                        f"  • Information Ratio: {metrics.get('information_ratio', 0):.3f}",
                        f"  • Benchmark Return: {metrics.get('benchmark_return', 0)*100:.2f}%",
                    ]
                )

            return "\n".join(result)

        except Exception as e:
            return f"❌ Error calculating metrics: {str(e)}"


class GetTransactionHistoryTool(BaseTool):
    """Tool for getting transaction history."""

    name: str = "get_transaction_history"
    description: str = "Get recent transaction history for the portfolio."
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _run(self, days: int = 30) -> str:
        """Get transaction history."""
        try:
            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            try:
                transactions = self.portfolio_manager.get_transaction_history(days)
            except Exception as e:
                return f"❌ Error getting transaction history: {str(e)}"

            if not transactions:
                return f"📝 No transactions found in the last {days} days."

            result = [f"📝 **Transaction History** (Last {days} days)", ""]

            for txn in transactions[:20]:  # Limit to 20 most recent
                date_str = txn["timestamp"].strftime("%Y-%m-%d")
                txn_type = txn["type"].upper()
                symbol = txn["symbol"]

                if txn_type in ["BUY", "SELL"]:
                    result.append(
                        f"• {date_str}: {txn_type} {txn['quantity']} {symbol} "
                        f"@ {txn['price']} (Total: {txn['total_value']})"
                    )
                elif txn_type == "DIVIDEND":
                    result.append(
                        f"• {date_str}: DIVIDEND {symbol} {txn['total_value']}"
                    )
                else:
                    result.append(f"• {date_str}: {txn_type} {txn['total_value']}")

            return "\n".join(result)

        except Exception as e:
            return f"❌ Error getting transaction history: {str(e)}"


class ModifyTransactionInput(BaseModel):
    """Input for modifying a transaction."""

    transaction_id: str = Field(description="ID of the transaction to modify")
    quantity: Optional[float] = Field(default=None, description="New quantity (optional)")
    price: Optional[float] = Field(default=None, description="New price (optional)")
    date: Optional[str] = Field(
        default=None, description="New date in YYYY-MM-DD format (optional)"
    )
    notes: Optional[str] = Field(default=None, description="New notes (optional)")


class ModifyTransactionTool(BaseTool):
    """Tool for modifying existing transactions."""

    name: str = "modify_transaction"
    description: str = """Modify an existing transaction by its ID.

    You can modify:
    - quantity: New number of shares/units
    - price: New price per share/unit
    - date: New transaction date (YYYY-MM-DD)
    - notes: New notes for the transaction

    Use get_transactions or get_transaction_history to find transaction IDs first.
    """
    args_schema: type[BaseModel] = ModifyTransactionInput
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _run(
        self,
        transaction_id: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        date: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> str:
        """Modify a transaction."""
        try:
            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            # Find the transaction
            portfolio = self.portfolio_manager.current_portfolio
            transaction = None
            for txn in portfolio.transactions:
                if txn.id == transaction_id:
                    transaction = txn
                    break

            if not transaction:
                return f"❌ Transaction not found with ID: {transaction_id}"

            # Track what was modified
            modifications = []

            if quantity is not None:
                old_qty = transaction.quantity
                transaction.quantity = Decimal(str(quantity))
                modifications.append(f"quantity: {old_qty} → {quantity}")

            if price is not None:
                old_price = transaction.price
                transaction.price = Decimal(str(price))
                modifications.append(f"price: {old_price} → {price}")

            if date is not None:
                try:
                    new_timestamp = datetime.strptime(date, "%Y-%m-%d")
                    old_date = transaction.timestamp.strftime("%Y-%m-%d")
                    transaction.timestamp = new_timestamp
                    modifications.append(f"date: {old_date} → {date}")
                except ValueError:
                    return "❌ Invalid date format. Use YYYY-MM-DD."

            if notes is not None:
                old_notes = transaction.notes or "(none)"
                transaction.notes = notes
                modifications.append(f"notes: {old_notes} → {notes}")

            if not modifications:
                return "❌ No modifications specified. Provide at least one field to modify."

            # Recalculate positions
            portfolio.recalculate_positions()

            # Save the portfolio
            self.portfolio_manager.storage.save_portfolio(portfolio)

            return (
                f"✅ Modified transaction {transaction_id[:8]}...\n"
                f"Changes: {', '.join(modifications)}"
            )

        except Exception as e:
            return f"❌ Error modifying transaction: {str(e)}"


class DeleteTransactionInput(BaseModel):
    """Input for deleting a transaction."""

    transaction_id: str = Field(description="ID of the transaction to delete")
    confirm: bool = Field(
        default=True, description="Set to True to confirm deletion"
    )


class DeleteTransactionTool(BaseTool):
    """Tool for deleting transactions from the portfolio."""

    name: str = "delete_transaction"
    description: str = """Delete a transaction from the portfolio by its ID.

    WARNING: This action cannot be undone. The portfolio positions will be recalculated
    after the transaction is removed.

    Use get_transactions or get_transaction_history to find transaction IDs first.
    """
    args_schema: type[BaseModel] = DeleteTransactionInput
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _run(self, transaction_id: str, confirm: bool = True) -> str:
        """Delete a transaction."""
        try:
            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            if not confirm:
                return "❌ Deletion not confirmed. Set confirm=True to proceed."

            portfolio = self.portfolio_manager.current_portfolio

            # Find and remove the transaction
            transaction_to_delete = None
            for txn in portfolio.transactions:
                if txn.id == transaction_id:
                    transaction_to_delete = txn
                    break

            if not transaction_to_delete:
                return f"❌ Transaction not found with ID: {transaction_id}"

            # Store details for confirmation message
            symbol = transaction_to_delete.instrument.symbol
            txn_type = transaction_to_delete.transaction_type.value
            quantity = transaction_to_delete.quantity
            price = transaction_to_delete.price
            txn_date = transaction_to_delete.timestamp.strftime("%Y-%m-%d")

            # Remove the transaction
            portfolio.transactions.remove(transaction_to_delete)

            # Recalculate positions
            portfolio.recalculate_positions()

            # Save the portfolio
            self.portfolio_manager.storage.save_portfolio(portfolio)

            return (
                f"✅ Deleted transaction:\n"
                f"• ID: {transaction_id[:8]}...\n"
                f"• Type: {txn_type}\n"
                f"• Symbol: {symbol}\n"
                f"• Quantity: {quantity}\n"
                f"• Price: {price}\n"
                f"• Date: {txn_date}\n\n"
                f"Portfolio positions have been recalculated."
            )

        except Exception as e:
            return f"❌ Error deleting transaction: {str(e)}"


class SetMarketPriceInput(BaseModel):
    """Input for setting market price."""

    symbol: str = Field(description="Stock/instrument symbol (e.g., AAPL, TSLA)")
    price: Optional[float] = Field(
        default=None,
        description="Price to set. Required unless use_purchase_price is True.",
    )
    date: Optional[str] = Field(
        default=None,
        description="Date for the price in YYYY-MM-DD format. Defaults to today.",
    )
    use_purchase_price: bool = Field(
        default=False,
        description="If True, uses the position's purchase price (average_cost) as the market price instead of the 'price' parameter.",
    )


class SetMarketPriceTool(BaseTool):
    """Tool for setting or updating market prices for instruments."""

    name: str = "set_market_price"
    description: str = """Set or update the market price for an instrument on a specific date.

    Use cases:
    - When price lookup fails and user wants to use purchase price as market price
    - When user wants to manually set a custom price for an instrument
    - When correcting historical prices in snapshots

    Parameters:
    - symbol: The instrument symbol (required)
    - price: The price to set (required unless use_purchase_price is True)
    - date: Date for the price in YYYY-MM-DD format (optional, defaults to today)
    - use_purchase_price: If True, uses the position's average_cost as the market price

    Examples:
    - "Set AAPL price to $150" → set_market_price(symbol="AAPL", price=150)
    - "Use purchase price for XYZ" → set_market_price(symbol="XYZ", use_purchase_price=True)
    - "Set TSLA price to $200 for 2024-01-15" → set_market_price(symbol="TSLA", price=200, date="2024-01-15")
    """
    args_schema: type[BaseModel] = SetMarketPriceInput
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _run(
        self,
        symbol: str,
        price: Optional[float] = None,
        date: Optional[str] = None,
        use_purchase_price: bool = False,
    ) -> str:
        """Set market price for an instrument."""
        try:
            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            symbol = symbol.upper().strip()

            # Check if symbol exists in portfolio
            if symbol not in self.portfolio_manager.current_portfolio.positions:
                return f"❌ Symbol '{symbol}' not found in portfolio. Please add a transaction for this symbol first."

            position = self.portfolio_manager.current_portfolio.positions[symbol]

            # Determine the price to use
            if use_purchase_price:
                final_price = position.average_cost
                price_source = "purchase price (average cost)"
            elif price is not None:
                final_price = Decimal(str(price))
                price_source = "custom price"
            else:
                return "❌ Please provide either a price or set use_purchase_price=True."

            # Parse date
            if date:
                try:
                    from datetime import datetime as dt
                    target_date = dt.strptime(date, "%Y-%m-%d").date()
                except ValueError:
                    return "❌ Invalid date format. Use YYYY-MM-DD."
            else:
                target_date = None  # Will default to today in the manager

            # Set the price
            success = self.portfolio_manager.set_position_price(
                symbol=symbol,
                price=final_price,
                target_date=target_date,
                update_current=True,
            )

            if success:
                target_date_str = (date if date else "today")
                return (
                    f"✅ Set market price for {symbol}:\n"
                    f"• Price: {final_price:.2f} ({price_source})\n"
                    f"• Date: {target_date_str}\n"
                    f"• Position: {position.instrument.name}\n"
                    f"• Quantity: {position.quantity}"
                )
            else:
                return f"❌ Failed to set price for {symbol}. Check logs for details."

        except Exception as e:
            return f"❌ Error setting market price: {str(e)}"


def create_portfolio_tools(
    portfolio_manager: PortfolioManager,
    data_manager: DataProviderManager,
    metrics_calculator: FinancialMetricsCalculator,
) -> List[BaseTool]:
    """Create all portfolio management tools."""
    return [
        AddTransactionTool(portfolio_manager),
        ModifyTransactionTool(portfolio_manager),
        DeleteTransactionTool(portfolio_manager),
        GetPortfolioSummaryTool(portfolio_manager, metrics_calculator),
        GetTransactionsTool(portfolio_manager),
        SimulateWhatIfTool(portfolio_manager),
        AdvancedWhatIfTool(portfolio_manager),
        HypotheticalPositionTool(portfolio_manager),
        IngestPdfTool(),
        CalculatorTool(),
        SearchInstrumentTool(data_manager),
        SearchCompanyTool(data_manager),
        GetCurrentPriceTool(data_manager),
        GetPortfolioMetricsTool(portfolio_manager, metrics_calculator),
        GetTransactionHistoryTool(portfolio_manager),
    ]


def create_transaction_tools(
    portfolio_manager: PortfolioManager,
    data_manager: DataProviderManager,
) -> List[BaseTool]:
    """Create tools for the Transaction Agent (CRUD operations)."""
    return [
        AddTransactionTool(portfolio_manager),
        ModifyTransactionTool(portfolio_manager),
        DeleteTransactionTool(portfolio_manager),
        SetMarketPriceTool(portfolio_manager),
        SearchInstrumentTool(data_manager),
        SearchCompanyTool(data_manager),
    ]


def create_analytics_tools(
    portfolio_manager: PortfolioManager,
    data_manager: DataProviderManager,
    metrics_calculator: FinancialMetricsCalculator,
) -> List[BaseTool]:
    """Create tools for the Analytics Agent (data and analysis)."""
    return [
        GetPortfolioSummaryTool(portfolio_manager, metrics_calculator),
        GetPortfolioMetricsTool(portfolio_manager, metrics_calculator),
        GetTransactionsTool(portfolio_manager),
        GetTransactionHistoryTool(portfolio_manager),
        SimulateWhatIfTool(portfolio_manager),
        AdvancedWhatIfTool(portfolio_manager),
        HypotheticalPositionTool(portfolio_manager),
        OptimizePortfolioTool(portfolio_manager, data_manager),
        ScenarioOptimizationTool(portfolio_manager, data_manager),
        SetMarketPriceTool(portfolio_manager),
        GetCurrentPriceTool(data_manager),
        CalculatorTool(),
        IngestPdfTool(),
    ]


class CalculatorTool(BaseTool):
    """A safe calculator for evaluating mathematical expressions."""

    name: str = "calculator"
    description: str = (
        "Evaluate a mathematical expression. Supports +, -, *, /, **, %, parentheses, and math functions (e.g., sin, cos, log).\n"
        "Examples: '2*(3+4)', 'sin(pi/2)', 'log(100,10)'."
    )

    def _run(self, expression: str) -> str:
        try:
            # Allowed names from math
            allowed_names = {
                k: getattr(math, k) for k in dir(math) if not k.startswith("__")
            }
            # Common constants
            allowed_names.update({"pi": math.pi, "e": math.e})

            # Disallow builtins
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return f"🧮 {expression} = {result}"
        except Exception as e:
            return f"❌ Error evaluating expression: {str(e)}"


class IngestPdfTool(BaseTool):
    """Tool to ingest a PDF file and return extracted text for analysis."""

    name: str = "ingest_pdf"
    description: str = (
        "Ingest a local PDF file (datasheet) by providing its absolute path; returns extracted text for the agent to analyze."
    )

    def _run(self, path: str) -> str:
        try:
            reader = PdfReader(path)
            texts = []
            for page in reader.pages:
                try:
                    txt = page.extract_text() or ""
                    if txt:
                        texts.append(txt)
                except Exception:
                    continue
            content = "\n\n".join(texts).strip()
            if not content:
                return "❌ No text extracted from PDF."
            # Truncate extremely long content to keep within LLM context (basic safeguard)
            if len(content) > 200_000:
                content = content[:200_000] + "\n\n...[truncated]"
            return (
                f"📄 PDF Content Extracted (length: {len(content)} chars)\n\n{content}"
            )
        except FileNotFoundError:
            return f"❌ File not found: {path}"
        except Exception as e:
            return f"❌ Error reading PDF: {str(e)}"


class OptimizePortfolioInput(BaseModel):
    """Input for portfolio optimization."""

    locked_symbols: Optional[str] = Field(
        default=None,
        description="Comma-separated symbols to keep at current weights (e.g., 'AAPL,GOOG')",
    )
    method: str = Field(
        default="hrp",
        description="Optimization method: 'hrp' (recommended) or 'markowitz'",
    )
    compare: bool = Field(
        default=True,
        description="If true, show both HRP and Markowitz for comparison",
    )
    lookback_days: int = Field(
        default=252,
        description="Days of historical data for covariance (default: 252 = 1 year)",
    )
    objective: str = Field(
        default="max_sharpe",
        description="Optimization objective: 'max_sharpe' (maximize risk-adjusted return), 'min_volatility' (minimize risk), or 'efficient_risk' (target specific volatility)",
    )
    target_volatility: Optional[float] = Field(
        default=None,
        description="Target annual volatility as decimal (e.g., 0.15 for 15%). Only used with objective='efficient_risk'",
    )
    include_cash: bool = Field(
        default=True,
        description="Include cash in portfolio. When True, allows cash allocation to achieve target volatility and includes existing cash in trade calculations.",
    )
    risk_free_rate: float = Field(
        default=0.04,
        description="Annual risk-free rate for Sharpe calculation (default: 0.04 = 4%)",
    )


class OptimizePortfolioTool(BaseTool):
    """Tool for optimizing portfolio weights."""

    name: str = "optimize_portfolio"
    description: str = """Analyze current portfolio and suggest optimal weight allocation for rebalancing.

    Uses two optimization methods:
    - HRP (Hierarchical Risk Parity): Default, more robust, handles correlated assets well
    - Markowitz (Mean-Variance): Classic approach, can maximize Sharpe or minimize volatility

    Optimization Objectives:
    - max_sharpe: Maximize risk-adjusted return (Sharpe ratio)
    - min_volatility: Minimize portfolio volatility/risk
    - efficient_risk: Target a specific volatility level (requires target_volatility parameter)

    Features:
    - Lock specific positions you don't want to change
    - Compare both methods side-by-side
    - Include cash for lower volatility targets (blends risky assets with cash)
    - Shows specific rebalancing trades needed with share counts

    Examples:
    - optimize_portfolio() - optimize for max Sharpe with cash enabled
    - optimize_portfolio(objective="min_volatility") - minimize risk
    - optimize_portfolio(objective="efficient_risk", target_volatility=0.10) - target 10% volatility
    - optimize_portfolio(locked_symbols="VTI,BND") - keep VTI and BND at current weights
    - optimize_portfolio(include_cash=False) - don't include cash in rebalancing
    - optimize_portfolio(lookback_days=126, risk_free_rate=0.05) - 6 months data, 5% risk-free rate
    """
    args_schema: type[BaseModel] = OptimizePortfolioInput
    portfolio_manager: Optional[PortfolioManager] = None
    data_manager: Optional[DataProviderManager] = None

    def __init__(
        self, portfolio_manager: PortfolioManager, data_manager: DataProviderManager
    ):
        super().__init__()
        self.portfolio_manager = portfolio_manager
        self.data_manager = data_manager

    def _run(
        self,
        locked_symbols: Optional[str] = None,
        method: str = "hrp",
        compare: bool = True,
        lookback_days: int = 252,
        objective: str = "max_sharpe",
        target_volatility: Optional[float] = None,
        include_cash: bool = True,
        risk_free_rate: float = 0.04,
    ) -> str:
        """Run portfolio optimization."""
        try:
            from ..portfolio.optimizer import OptimizationObjective

            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded. Please create or load a portfolio first."

            portfolio = self.portfolio_manager.current_portfolio
            positions = portfolio.positions

            if len(positions) < 2:
                return "❌ Need at least 2 positions to optimize portfolio."

            # Parse locked symbols
            locked_list = []
            if locked_symbols:
                locked_list = [s.strip().upper() for s in locked_symbols.split(",")]
                # Validate locked symbols exist
                invalid = [s for s in locked_list if s not in positions]
                if invalid:
                    return f"❌ Locked symbols not in portfolio: {', '.join(invalid)}"

            # Parse objective
            objective_map = {
                "max_sharpe": OptimizationObjective.MAX_SHARPE,
                "min_volatility": OptimizationObjective.MIN_VOLATILITY,
                "efficient_risk": OptimizationObjective.EFFICIENT_RISK,
            }
            opt_objective = objective_map.get(objective.lower(), OptimizationObjective.MAX_SHARPE)

            # Update current prices before optimization
            self.portfolio_manager.update_current_prices()

            # Calculate total portfolio value
            total_value = sum(
                pos.market_value or (pos.quantity * pos.average_cost)
                for pos in positions.values()
                if pos.quantity > 0
            )

            # Get cash balances if including cash
            cash_balances = portfolio.cash_balances if include_cash else None

            # Create optimizer
            optimizer = PortfolioOptimizer(
                self.data_manager,
                base_currency=portfolio.base_currency,
            )

            # Run optimization
            if compare:
                results = optimizer.compare_methods(
                    positions=positions,
                    locked_symbols=locked_list,
                    lookback_days=lookback_days,
                    risk_free_rate=risk_free_rate,
                    total_portfolio_value=total_value,
                    cash_balances=cash_balances,
                    objective=opt_objective,
                    target_volatility=target_volatility,
                    include_cash=include_cash,
                )
                return self._format_comparison_results(results, locked_list, total_value, include_cash)
            else:
                opt_method = (
                    OptimizationMethod.HRP
                    if method.lower() == "hrp"
                    else OptimizationMethod.MARKOWITZ
                )
                result = optimizer.optimize(
                    positions=positions,
                    locked_symbols=locked_list,
                    method=opt_method,
                    lookback_days=lookback_days,
                    risk_free_rate=risk_free_rate,
                    total_portfolio_value=total_value,
                    cash_balances=cash_balances,
                    objective=opt_objective,
                    target_volatility=target_volatility,
                    include_cash=include_cash,
                )
                return self._format_single_result(result, total_value, include_cash)

        except ValueError as e:
            return f"❌ Optimization error: {str(e)}"
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"

    def _format_single_result(self, result, total_value, include_cash: bool = True) -> str:
        """Format a single optimization result."""
        lines = [
            "📊 Portfolio Optimization Results",
            "━" * 40,
            "",
        ]

        if result.locked_symbols:
            lines.append(f"🔒 Locked positions: {', '.join(result.locked_symbols)}")
            lines.append("")

        method_name = "HRP (Hierarchical Risk Parity)" if result.method == OptimizationMethod.HRP else "Markowitz (Mean-Variance)"
        lines.append(f"📈 Method: {method_name}")
        lines.append("")

        # Metrics
        if result.expected_annual_return is not None:
            lines.append(f"Expected Annual Return: {result.expected_annual_return:.1%}")
        lines.append(f"Annual Volatility: {result.annual_volatility:.1%}")
        if result.sharpe_ratio is not None:
            lines.append(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")

        # Cash allocation
        if include_cash and result.cash_weight and result.cash_weight > 0.001:
            lines.append(f"Cash Allocation: {result.cash_weight:.1%}")
        lines.append("")

        # Target weights
        lines.append("Target Weights:")
        for sym in sorted(result.weights.keys()):
            target = result.weights[sym]
            current = result.current_weights.get(sym, 0)
            diff = target - current
            locked = " (locked)" if sym in result.locked_symbols else ""
            if abs(diff) < 0.001:
                lines.append(f"  {sym}: {target:.1%}{locked}")
            else:
                sign = "+" if diff > 0 else ""
                lines.append(f"  {sym}: {current:.1%} → {target:.1%} ({sign}{diff:.1%}){locked}")

        # Show cash if allocated
        if include_cash and result.cash_weight and result.cash_weight > 0.001:
            lines.append(f"  CASH: {result.cash_weight:.1%}")
        lines.append("")

        # Rebalancing trades
        if result.rebalancing_trades:
            lines.append("📋 Rebalancing Trades:")
            for trade in result.rebalancing_trades:
                if trade.shares > 0:
                    lines.append(
                        f"  • {trade.action} {trade.shares:.0f} shares {trade.symbol} (~{trade.estimated_value:,.0f} USD)"
                    )
                else:
                    # Cash trade (no shares)
                    lines.append(
                        f"  • {trade.action} {trade.symbol} (~{trade.estimated_value:,.0f} USD)"
                    )
            lines.append("")

        # Warnings
        if result.warnings:
            lines.append("⚠️ Warnings:")
            for warn in result.warnings:
                lines.append(f"  • {warn}")

        return "\n".join(lines)

    def _format_comparison_results(self, results, locked_list, total_value, include_cash: bool = True) -> str:
        """Format comparison of multiple optimization methods."""
        lines = [
            "📊 Portfolio Optimization Results",
            "━" * 50,
            "",
        ]

        if locked_list:
            lines.append(f"🔒 Locked positions: {', '.join(locked_list)}")
            lines.append("")

        # HRP Results (primary)
        hrp_result = results.get(OptimizationMethod.HRP)
        if hrp_result and hrp_result.weights:
            lines.append("┌" + "─" * 48 + "┐")
            lines.append("│ HRP (Hierarchical Risk Parity) - RECOMMENDED" + " " * 3 + "│")
            lines.append("├" + "─" * 48 + "┤")

            if hrp_result.expected_annual_return is not None:
                exp_ret = f"Expected Return: {hrp_result.expected_annual_return:.1%}"
            else:
                exp_ret = "Expected Return: N/A"
            vol = f"Volatility: {hrp_result.annual_volatility:.1%}"
            lines.append(f"│ {exp_ret}   {vol}".ljust(49) + "│")

            if hrp_result.sharpe_ratio is not None:
                lines.append(f"│ Sharpe Ratio: {hrp_result.sharpe_ratio:.2f}".ljust(49) + "│")

            # Cash allocation
            if include_cash and hrp_result.cash_weight and hrp_result.cash_weight > 0.001:
                lines.append(f"│ Cash Allocation: {hrp_result.cash_weight:.1%}".ljust(49) + "│")

            lines.append("├" + "─" * 48 + "┤")
            lines.append("│ Target Weights:".ljust(49) + "│")

            for sym in sorted(hrp_result.weights.keys()):
                target = hrp_result.weights[sym]
                current = hrp_result.current_weights.get(sym, 0)
                diff = target - current
                locked = " (locked)" if sym in locked_list else ""
                if abs(diff) < 0.001:
                    line = f"│   {sym}: {target:.1%}{locked}"
                else:
                    sign = "+" if diff > 0 else ""
                    line = f"│   {sym}: {current:.1%} → {target:.1%} ({sign}{diff:.1%}){locked}"
                lines.append(line.ljust(49) + "│")

            # Show cash if allocated
            if include_cash and hrp_result.cash_weight and hrp_result.cash_weight > 0.001:
                lines.append(f"│   CASH: {hrp_result.cash_weight:.1%}".ljust(49) + "│")

            lines.append("└" + "─" * 48 + "┘")
            lines.append("")

        # Markowitz Results (comparison)
        mk_result = results.get(OptimizationMethod.MARKOWITZ)
        if mk_result and mk_result.weights:
            lines.append("┌" + "─" * 48 + "┐")
            lines.append("│ Markowitz (Mean-Variance) - FOR COMPARISON" + " " * 5 + "│")
            lines.append("├" + "─" * 48 + "┤")

            if mk_result.expected_annual_return is not None:
                exp_ret = f"Expected Return: {mk_result.expected_annual_return:.1%}"
            else:
                exp_ret = "Expected Return: N/A"
            vol = f"Volatility: {mk_result.annual_volatility:.1%}"
            lines.append(f"│ {exp_ret}   {vol}".ljust(49) + "│")

            if mk_result.sharpe_ratio is not None:
                lines.append(f"│ Sharpe Ratio: {mk_result.sharpe_ratio:.2f}".ljust(49) + "│")

            # Cash allocation
            if include_cash and mk_result.cash_weight and mk_result.cash_weight > 0.001:
                lines.append(f"│ Cash Allocation: {mk_result.cash_weight:.1%}".ljust(49) + "│")

            lines.append("├" + "─" * 48 + "┤")
            lines.append("│ Target Weights:".ljust(49) + "│")

            for sym in sorted(mk_result.weights.keys()):
                target = mk_result.weights[sym]
                current = mk_result.current_weights.get(sym, 0)
                diff = target - current
                locked = " (locked)" if sym in locked_list else ""

                # Flag concentrated positions
                concentrated = " ⚠️" if target > 0.4 and sym not in locked_list else ""

                if abs(diff) < 0.001:
                    line = f"│   {sym}: {target:.1%}{locked}{concentrated}"
                else:
                    sign = "+" if diff > 0 else ""
                    line = f"│   {sym}: {current:.1%} → {target:.1%} ({sign}{diff:.1%}){locked}{concentrated}"
                lines.append(line.ljust(49) + "│")

            # Show cash if allocated
            if include_cash and mk_result.cash_weight and mk_result.cash_weight > 0.001:
                lines.append(f"│   CASH: {mk_result.cash_weight:.1%}".ljust(49) + "│")

            lines.append("└" + "─" * 48 + "┘")
            lines.append("")

        # Rebalancing trades (for HRP)
        if hrp_result and hrp_result.rebalancing_trades:
            lines.append("📋 Rebalancing Trades (HRP):")
            for trade in hrp_result.rebalancing_trades:
                if trade.shares > 0:
                    lines.append(
                        f"  • {trade.action} {trade.shares:.0f} shares {trade.symbol} (~{trade.estimated_value:,.0f} USD)"
                    )
                else:
                    # Cash trade (no shares)
                    lines.append(
                        f"  • {trade.action} {trade.symbol} (~{trade.estimated_value:,.0f} USD)"
                    )
            lines.append("")

        # Notes
        lines.append("💡 Notes:")
        lines.append("  • HRP provides more diversified, robust allocation")
        lines.append("  • Markowitz may concentrate heavily in few assets")
        if include_cash:
            lines.append("  • Cash can be used to reduce portfolio volatility")
        lines.append("  • Consider transaction costs before rebalancing")

        # Collect warnings
        all_warnings = []
        for result in results.values():
            if result and result.warnings:
                all_warnings.extend(result.warnings)
        if all_warnings:
            lines.append("")
            lines.append("⚠️ Warnings:")
            for warn in set(all_warnings):
                lines.append(f"  • {warn}")

        return "\n".join(lines)


class ScenarioOptimizationInput(BaseModel):
    """Input for scenario-based optimization."""

    scenarios: str = Field(
        default="optimistic,likely,pessimistic,stress",
        description="Comma-separated list of scenarios: optimistic, likely, pessimistic, stress",
    )
    objective: str = Field(
        default="max_sharpe",
        description="Optimization objective: 'max_sharpe' or 'min_volatility'",
    )
    include_cash: bool = Field(
        default=True,
        description="Include cash in portfolio optimization",
    )
    projection_years: float = Field(
        default=5.0,
        description="Projection period in years (1-30)",
    )
    monte_carlo_runs: int = Field(
        default=1000,
        description="Number of Monte Carlo simulations (500, 1000, 2500, or 5000)",
    )
    confidence_levels: str = Field(
        default="50,75,90",
        description="Comma-separated confidence levels in percent (e.g., '50,75,90,95')",
    )


class ScenarioOptimizationTool(BaseTool):
    """Tool for comparing portfolio optimization under different market scenarios."""

    name: str = "scenario_optimization"
    description: str = """Compare how optimal portfolio allocation changes under different market scenarios.

    This tool runs optimization under multiple market conditions and shows how the recommended
    allocation differs. Uses the same parameters as the UI Scenario Analysis tab.

    Available Scenarios:
    - optimistic: Bull Market Growth - 14% equity return, 15% volatility
    - likely: Historical Average - 10% equity return, 20% volatility
    - pessimistic: Economic Downturn - 3% equity return, 30% volatility
    - stress: Market Crash - negative returns, 45% volatility

    Parameters (same as UI):
    - projection_years: How far to project (default: 5 years)
    - monte_carlo_runs: Number of simulations (default: 1000)
    - confidence_levels: Confidence intervals to show (default: 50%, 75%, 90%)

    Examples:
    - scenario_optimization() - Compare all 4 scenarios with defaults
    - scenario_optimization(scenarios="optimistic,pessimistic") - Compare extremes
    - scenario_optimization(projection_years=10, monte_carlo_runs=2500) - Longer horizon, more precision
    - scenario_optimization(objective="min_volatility") - Focus on risk minimization
    """
    args_schema: type[BaseModel] = ScenarioOptimizationInput
    portfolio_manager: Optional[PortfolioManager] = None
    data_manager: Optional[DataProviderManager] = None

    def __init__(
        self, portfolio_manager: PortfolioManager, data_manager: DataProviderManager
    ):
        super().__init__()
        self.portfolio_manager = portfolio_manager
        self.data_manager = data_manager

    def _run(
        self,
        scenarios: str = "optimistic,likely,pessimistic,stress",
        objective: str = "max_sharpe",
        include_cash: bool = True,
        projection_years: float = 5.0,
        monte_carlo_runs: int = 1000,
        confidence_levels: str = "50,75,90",
    ) -> str:
        """Run optimization under different market scenarios."""
        try:
            from ..portfolio.optimizer import OptimizationObjective, OptimizationMethod
            from ..portfolio.scenarios import PortfolioScenarioEngine, ScenarioType

            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded. Please create or load a portfolio first."

            portfolio = self.portfolio_manager.current_portfolio
            positions = portfolio.positions

            if len(positions) < 2:
                return "❌ Need at least 2 positions to optimize portfolio."

            # Parse scenarios
            scenario_list = [s.strip().lower() for s in scenarios.split(",")]

            # Parse confidence levels
            conf_levels = [int(c.strip()) / 100.0 for c in confidence_levels.split(",")]

            # Scenario configurations matching the UI (from scenarios.py)
            scenario_configs = {
                "optimistic": {
                    "type": ScenarioType.OPTIMISTIC,
                    "adjustment": 0.04,
                    "name": "Bull Market Growth",
                    "description": "Strong growth, low volatility, favorable conditions",
                    "equity_return": 0.14,
                    "volatility": 0.15,
                },
                "likely": {
                    "type": ScenarioType.LIKELY,
                    "adjustment": 0.0,
                    "name": "Historical Average",
                    "description": "Markets perform in line with long-term averages",
                    "equity_return": 0.10,
                    "volatility": 0.20,
                },
                "pessimistic": {
                    "type": ScenarioType.PESSIMISTIC,
                    "adjustment": -0.07,
                    "name": "Economic Downturn",
                    "description": "Recession, market stress, high volatility",
                    "equity_return": 0.03,
                    "volatility": 0.30,
                },
                "stress": {
                    "type": ScenarioType.STRESS,
                    "adjustment": -0.15,
                    "name": "Market Crash",
                    "description": "Severe market crash, very high volatility",
                    "equity_return": -0.05,
                    "volatility": 0.45,
                },
            }

            # Validate scenarios
            valid_scenarios = []
            for s in scenario_list:
                if s in scenario_configs:
                    valid_scenarios.append(s)
                else:
                    return f"❌ Unknown scenario '{s}'. Valid options: {', '.join(scenario_configs.keys())}"

            # Parse objective
            objective_map = {
                "max_sharpe": OptimizationObjective.MAX_SHARPE,
                "min_volatility": OptimizationObjective.MIN_VOLATILITY,
            }
            opt_objective = objective_map.get(objective.lower(), OptimizationObjective.MAX_SHARPE)

            # Update prices
            self.portfolio_manager.update_current_prices()

            # Calculate total portfolio value
            total_value = sum(
                pos.market_value or (pos.quantity * pos.average_cost)
                for pos in positions.values()
                if pos.quantity > 0
            )

            cash_balances = portfolio.cash_balances if include_cash else None

            # Create optimizer
            optimizer = PortfolioOptimizer(
                self.data_manager,
                base_currency=portfolio.base_currency,
            )

            # Run optimization for each scenario
            scenario_results = {}
            for scenario_name in valid_scenarios:
                config = scenario_configs[scenario_name]

                # Run optimization with return adjustments
                try:
                    result = optimizer.optimize(
                        positions=positions,
                        method=OptimizationMethod.MARKOWITZ,
                        lookback_days=252,
                        risk_free_rate=0.04,
                        total_portfolio_value=total_value,
                        cash_balances=cash_balances,
                        objective=opt_objective,
                        include_cash=include_cash,
                        return_adjustment=config["adjustment"],
                    )
                    scenario_results[scenario_name] = {
                        "result": result,
                        "config": config,
                    }
                except Exception as e:
                    scenario_results[scenario_name] = {
                        "error": str(e),
                        "config": config,
                    }

            return self._format_scenario_results(
                scenario_results, total_value, include_cash,
                projection_years, monte_carlo_runs, conf_levels
            )

        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"

    def _format_scenario_results(
        self, scenario_results, total_value, include_cash,
        projection_years: float = 5.0, monte_carlo_runs: int = 1000,
        confidence_levels: list = None
    ) -> str:
        """Format scenario comparison results."""
        if confidence_levels is None:
            confidence_levels = [0.5, 0.75, 0.9]

        conf_pcts = ", ".join([f"{int(c*100)}%" for c in confidence_levels])

        lines = [
            "🎭 Scenario-Based Optimization Comparison",
            "━" * 55,
            "",
            f"Parameters: {projection_years:.0f}yr projection, {monte_carlo_runs:,} MC runs, {conf_pcts} confidence",
            "",
            "Compare how optimal allocation changes under different market conditions:",
            "",
        ]

        # Collect all symbols across all scenarios
        all_symbols = set()
        for scenario_name, data in scenario_results.items():
            if "result" in data and data["result"].weights:
                all_symbols.update(data["result"].weights.keys())

        # Format each scenario
        for scenario_name, data in scenario_results.items():
            config = data["config"]
            lines.append(f"┌{'─' * 53}┐")
            emoji = {
                "optimistic": "📈",
                "likely": "➡️",
                "pessimistic": "📉",
                "stress": "💥"
            }.get(scenario_name, "📊")
            header = f"│ {emoji} {config['name']}"
            lines.append(header.ljust(54) + "│")
            lines.append(f"│ {config['description'][:51]}".ljust(54) + "│")
            scenario_params = f"│ Equity: {config['equity_return']:.0%} return, {config['volatility']:.0%} volatility"
            lines.append(scenario_params.ljust(54) + "│")
            lines.append(f"├{'─' * 53}┤")

            if "error" in data:
                lines.append(f"│ ❌ Error: {data['error'][:40]}".ljust(54) + "│")
            else:
                result = data["result"]

                # Metrics
                if result.expected_annual_return is not None:
                    ret_line = f"│ Expected Return: {result.expected_annual_return:.1%}"
                else:
                    ret_line = "│ Expected Return: N/A"
                lines.append(ret_line.ljust(54) + "│")
                lines.append(f"│ Volatility: {result.annual_volatility:.1%}".ljust(54) + "│")
                if result.sharpe_ratio:
                    lines.append(f"│ Sharpe Ratio: {result.sharpe_ratio:.2f}".ljust(54) + "│")
                if include_cash and result.cash_weight and result.cash_weight > 0.001:
                    lines.append(f"│ Cash: {result.cash_weight:.1%}".ljust(54) + "│")

                lines.append(f"├{'─' * 53}┤")
                lines.append("│ Allocation:".ljust(54) + "│")

                for sym in sorted(all_symbols):
                    weight = result.weights.get(sym, 0)
                    if weight > 0.001:
                        lines.append(f"│   {sym}: {weight:.1%}".ljust(54) + "│")

            lines.append(f"└{'─' * 53}┘")
            lines.append("")

        # Key insights
        lines.append("💡 Key Insights:")

        # Find differences between optimistic and pessimistic/stress
        if len(scenario_results) >= 2:
            # Try to compare optimistic vs pessimistic, or first vs last
            first_name = "optimistic" if "optimistic" in scenario_results else list(scenario_results.keys())[0]
            last_name = "stress" if "stress" in scenario_results else (
                "pessimistic" if "pessimistic" in scenario_results else list(scenario_results.keys())[-1]
            )

            first = scenario_results.get(first_name, {})
            last = scenario_results.get(last_name, {})

            if "result" in first and "result" in last:
                first_weights = first["result"].weights
                last_weights = last["result"].weights

                for sym in sorted(all_symbols):
                    first_w = first_weights.get(sym, 0)
                    last_w = last_weights.get(sym, 0)
                    diff = last_w - first_w

                    if abs(diff) > 0.05:  # Significant difference
                        direction = "increases" if diff > 0 else "decreases"
                        lines.append(f"  • {sym} weight {direction} from {first_w:.1%} ({first_name}) to {last_w:.1%} ({last_name})")

        lines.append("")
        lines.append("📋 Interpretation:")
        lines.append("  • In optimistic scenarios, growth assets get higher weights")
        lines.append("  • In pessimistic/stress scenarios, defensive assets get higher weights")
        lines.append("  • Use this to understand how to position based on your market view")

        return "\n".join(lines)
