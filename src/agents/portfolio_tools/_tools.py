"""
LangChain tools for portfolio management and analysis.
"""

import logging
import math
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import List, Optional

from langchain.tools import BaseTool
from pydantic import BaseModel
from pypdf import PdfReader

from ...data_providers.manager import DataProviderManager
from ...portfolio.manager import PortfolioManager
from ...portfolio.models import TransactionType
from ...portfolio.optimizer import OptimizationMethod, PortfolioOptimizer
from ...utils.metrics import FinancialMetricsCalculator
from .models import (
    AddTransactionInput,
    BulkAddTransactionsInput,
    BulkSetMarketPriceInput,
    CheckMarketDataAvailabilityInput,
    DeleteTransactionInput,
    FetchAndUpdatePricesInput,
    GetHistoricalInstrumentsInput,
    GetMetricsInput,
    GetPortfolioSnapshotInput,
    GetPortfolioSummaryInput,
    GetPriceInput,
    GetTransactionsInput,
    GetYTDPerformanceInput,
    InterpolatePricesInput,
    ModifyTransactionInput,
    OptimizePortfolioInput,
    ResolveInstrumentInput,
    ScenarioOptimizationInput,
    SearchInstrumentInput,
    SetDataProviderSymbolInput,
    SetMarketPriceInput,
    SetPriceCurrencyInput,
    TransactionItem,
    UpdateHistoricalMarketDataInput,
)


class AddTransactionTool(BaseTool):
    """Tool for adding transactions to the portfolio."""

    name: str = "add_transaction"
    description: str = """Add a transaction to the portfolio.

    TRANSACTION TYPES AND REQUIRED FIELDS:

    BUY/SELL (stocks, bonds, ETFs, crypto):
    - symbol or isin: REQUIRED (which instrument)
    - quantity: REQUIRED (number of shares/units)
    - price: REQUIRED (price per share/unit)
    - date: REQUIRED (YYYY-MM-DD format)
    - currency: optional (defaults to instrument currency)
    - instrument_type: optional (auto-detected if not specified)

    DEPOSIT/WITHDRAWAL (cash movements):
    - price: REQUIRED (the amount to deposit/withdraw)
    - currency: REQUIRED (USD, EUR, GBP, etc.)
    - date: REQUIRED (YYYY-MM-DD format)
    - symbol/quantity: NOT needed (automatically set)

    DIVIDEND (income from stocks):
    - symbol or isin: REQUIRED (which stock paid the dividend)
    - price: REQUIRED (the dividend amount received)
    - date: REQUIRED (YYYY-MM-DD format)
    - currency: optional (defaults to instrument currency)
    - quantity: NOT needed (automatically set to 1)

    FEES (broker fees, commissions):
    - price: REQUIRED (the fee amount)
    - currency: REQUIRED (USD, EUR, GBP, etc.)
    - date: REQUIRED (YYYY-MM-DD format)
    - symbol/quantity: NOT needed (automatically set)

    EXAMPLES:
    - Buy 50 AAPL at $150: transaction_type="buy", symbol="AAPL", quantity=50, price=150, date="2024-01-15"
    - Deposit $25,000: transaction_type="deposit", price=25000, currency="USD", date="2024-01-15"
    - Dividend from MSFT: transaction_type="dividend", symbol="MSFT", price=125.50, date="2024-01-15"
    - Broker fee: transaction_type="fees", price=9.99, currency="USD", date="2024-01-15"
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
            is_cash_only = txn_type in (TransactionType.DEPOSIT, TransactionType.WITHDRAWAL, TransactionType.FEES)
            is_dividend = txn_type == TransactionType.DIVIDEND

            # Determine the amount - prefer price, fall back to quantity
            if price > 0:
                amount = price
            elif quantity > 0:
                amount = quantity
            else:
                amount = 0

            if is_cash_only:
                # DEPOSIT/WITHDRAWAL/FEES: need amount, currency, date
                missing_fields = []
                if amount <= 0:
                    missing_fields.append("amount (the monetary value)")
                if not currency:
                    missing_fields.append("currency (USD, EUR, GBP, etc.)")
                if not date:
                    missing_fields.append("date (when did this occur)")

                if missing_fields:
                    return f"Missing required information for {transaction_type}: {', '.join(missing_fields)}. Please ask the user to provide this information."

                # Set up: symbol=CASH, quantity=1, price=amount
                symbol = "CASH"
                isin = None
                final_price = amount
                quantity = 1.0

            elif is_dividend:
                # DIVIDEND: need symbol, amount (as price), date
                missing_fields = []
                if not symbol and not isin:
                    missing_fields.append("symbol (which stock paid the dividend)")
                if amount <= 0:
                    missing_fields.append("amount (the dividend amount)")
                if not date:
                    missing_fields.append("date (when was it paid)")

                if missing_fields:
                    return f"Missing required information for dividend: {', '.join(missing_fields)}. Please ask the user to provide this information."

                # Set up: quantity=1, price=amount
                final_price = amount
                quantity = 1.0

            else:
                # BUY/SELL: need symbol, quantity, price, date
                missing_fields = []
                if not symbol and not isin:
                    missing_fields.append("symbol or ISIN (which instrument)")
                if quantity <= 0:
                    missing_fields.append("quantity (how many shares/units)")
                if price <= 0:
                    missing_fields.append("price (price per share/unit)")
                if not date:
                    missing_fields.append("date (when did this transaction occur)")

                if missing_fields:
                    return f"Missing required information for {transaction_type}: {', '.join(missing_fields)}. Please ask the user to provide this information."

                final_price = price

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

            # Handle bond price interpretation (percentages) - only for BUY/SELL
            if not is_cash_only and not is_dividend:
                if isin and isin.upper().startswith("XS"):  # XS ISINs are typically bonds
                    # If price looks like a percentage (between 0 and 200), treat as percentage of face value
                    if 0 < final_price <= 200:
                        # Keep the percentage as-is for bonds (98.85% -> 98.85)
                        notes = f"{notes or ''} (Price: {final_price}% of face value)".strip()

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
                currency_str = cur_obj.value if cur_obj else "USD"

                # Format success message based on transaction type
                if txn_type == TransactionType.DEPOSIT:
                    return f"✅ Deposited {final_price:,.2f} {currency_str}"
                elif txn_type == TransactionType.WITHDRAWAL:
                    return f"✅ Withdrew {final_price:,.2f} {currency_str}"
                elif txn_type == TransactionType.FEES:
                    return f"✅ Recorded fee of {final_price:,.2f} {currency_str}"
                elif txn_type == TransactionType.DIVIDEND:
                    label = symbol.upper() if symbol else (f"ISIN_{isin[:8]}" if isin else "Unknown")
                    return f"✅ Recorded dividend of {final_price:,.2f} {currency_str} from {label}"

                # BUY/SELL: Show what was actually stored
                if symbol:
                    label = symbol.upper()
                elif isin:
                    label = f"ISIN_{isin[:8]}"
                else:
                    label = "Unknown"

                # Get the actual instrument info to show the resolved name
                try:
                    if symbol and self.portfolio_manager.current_portfolio:
                        for txn in reversed(self.portfolio_manager.current_portfolio.transactions):
                            if txn.instrument.symbol == symbol.upper():
                                return f"✅ {transaction_type.upper()}: {quantity} {txn.instrument.symbol} ({txn.instrument.name}) @ {final_price}"
                except Exception:
                    pass

                return f"✅ {transaction_type.upper()}: {quantity} {label} @ {final_price}"
            else:
                return "❌ Failed to add transaction. Make sure a portfolio is loaded."

        except Exception as e:
            return f"❌ Error adding transaction: {str(e)}"


class BulkAddTransactionsTool(BaseTool):
    """Tool for adding multiple transactions at once."""

    name: str = "bulk_add_transactions"
    description: str = """Add multiple transactions to the portfolio in a single call.

    This is more efficient than calling add_transaction multiple times.
    Pass a list of transaction objects with the same fields as add_transaction.

    TRANSACTION TYPES AND REQUIRED FIELDS:

    BUY/SELL (stocks, bonds, ETFs, crypto):
    - symbol or isin: REQUIRED (which instrument)
    - quantity: REQUIRED (number of shares/units)
    - price: REQUIRED (price per share/unit)
    - date: REQUIRED (YYYY-MM-DD format)
    - currency: optional (defaults to instrument currency)
    - instrument_type: optional (auto-detected if not specified)

    DEPOSIT/WITHDRAWAL (cash movements):
    - price: REQUIRED (the amount to deposit/withdraw)
    - currency: REQUIRED (USD, EUR, GBP, etc.)
    - date: REQUIRED (YYYY-MM-DD format)
    - symbol/quantity: NOT needed

    DIVIDEND (income from stocks):
    - symbol or isin: REQUIRED (which stock paid the dividend)
    - price: REQUIRED (the dividend amount received)
    - date: REQUIRED (YYYY-MM-DD format)
    - currency: optional (defaults to instrument currency)

    FEES (broker fees, commissions):
    - price: REQUIRED (the fee amount)
    - currency: REQUIRED (USD, EUR, GBP, etc.)
    - date: REQUIRED (YYYY-MM-DD format)

    EXAMPLE INPUT:
    {
        "transactions": [
            {"transaction_type": "deposit", "price": 50000, "currency": "USD", "date": "2024-01-01"},
            {"transaction_type": "buy", "symbol": "AAPL", "quantity": 100, "price": 150.0, "date": "2024-01-02"},
            {"transaction_type": "buy", "symbol": "MSFT", "quantity": 50, "price": 350.0, "date": "2024-01-02"},
            {"transaction_type": "dividend", "symbol": "AAPL", "price": 25.50, "date": "2024-03-15"}
        ]
    }
    """
    args_schema: type[BaseModel] = BulkAddTransactionsInput
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _run(self, transactions: List[TransactionItem]) -> str:
        """Add multiple transactions to the portfolio."""
        if not self.portfolio_manager.current_portfolio:
            return "❌ No portfolio loaded. Please create or load a portfolio first."

        if not transactions:
            return "❌ No transactions provided."

        # Convert Pydantic models to dicts if needed
        txn_list = []
        for txn in transactions:
            if isinstance(txn, dict):
                txn_list.append(txn)
            else:
                txn_list.append(txn.model_dump() if hasattr(txn, "model_dump") else txn.dict())

        results = []
        success_count = 0
        error_count = 0

        txn_type_map = {
            "buy": TransactionType.BUY,
            "sell": TransactionType.SELL,
            "dividend": TransactionType.DIVIDEND,
            "deposit": TransactionType.DEPOSIT,
            "withdrawal": TransactionType.WITHDRAWAL,
            "withdraw": TransactionType.WITHDRAWAL,
            "fees": TransactionType.FEES,
        }

        for i, txn in enumerate(txn_list, 1):
            try:
                transaction_type = txn.get("transaction_type", "").lower()
                txn_type = txn_type_map.get(transaction_type)

                if not txn_type:
                    error_count += 1
                    results.append(f"#{i}: ❌ Invalid transaction type: {transaction_type}")
                    continue

                symbol = txn.get("symbol")
                isin = txn.get("isin")
                instrument_type = txn.get("instrument_type")
                quantity = float(txn.get("quantity", 1.0))
                price = float(txn.get("price", 0))
                currency = txn.get("currency")
                date_str = txn.get("date")
                notes = txn.get("notes")

                # Validate required fields based on transaction type
                is_cash_only = txn_type in (
                    TransactionType.DEPOSIT,
                    TransactionType.WITHDRAWAL,
                    TransactionType.FEES,
                )
                is_dividend = txn_type == TransactionType.DIVIDEND

                # Determine the amount
                amount = price if price > 0 else quantity

                if is_cash_only:
                    if amount <= 0 or not currency or not date_str:
                        error_count += 1
                        results.append(
                            f"#{i}: ❌ {transaction_type} requires price, currency, and date"
                        )
                        continue
                    symbol = "CASH"
                    isin = None
                    final_price = amount
                    quantity = 1.0
                elif is_dividend:
                    if (not symbol and not isin) or amount <= 0 or not date_str:
                        error_count += 1
                        results.append(
                            f"#{i}: ❌ dividend requires symbol, price, and date"
                        )
                        continue
                    final_price = amount
                    quantity = 1.0
                else:
                    if (not symbol and not isin) or quantity <= 0 or price <= 0 or not date_str:
                        error_count += 1
                        results.append(
                            f"#{i}: ❌ {transaction_type} requires symbol, quantity, price, and date"
                        )
                        continue
                    final_price = price

                # Parse date
                try:
                    timestamp = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    error_count += 1
                    results.append(f"#{i}: ❌ Invalid date format: {date_str}")
                    continue

                # Parse currency
                from src.portfolio.models import Currency as Cur

                cur_obj = None
                if currency:
                    try:
                        cur_obj = Cur(currency.upper())
                    except ValueError:
                        error_count += 1
                        results.append(f"#{i}: ❌ Invalid currency: {currency}")
                        continue

                # Add transaction
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
                    success_count += 1
                    label = symbol.upper() if symbol else (isin[:8] if isin else "CASH")
                    if txn_type == TransactionType.DEPOSIT:
                        results.append(f"#{i}: ✅ Deposited {final_price:,.2f} {currency or 'USD'}")
                    elif txn_type == TransactionType.WITHDRAWAL:
                        results.append(f"#{i}: ✅ Withdrew {final_price:,.2f} {currency or 'USD'}")
                    elif txn_type == TransactionType.FEES:
                        results.append(f"#{i}: ✅ Fee {final_price:,.2f} {currency or 'USD'}")
                    elif txn_type == TransactionType.DIVIDEND:
                        results.append(f"#{i}: ✅ Dividend {final_price:,.2f} from {label}")
                    else:
                        results.append(f"#{i}: ✅ {transaction_type.upper()} {quantity} {label} @ {final_price}")
                else:
                    error_count += 1
                    results.append(f"#{i}: ❌ Failed to add transaction")

            except Exception as e:
                error_count += 1
                results.append(f"#{i}: ❌ Error: {str(e)}")

        # Build summary
        summary = f"Bulk transaction results: {success_count} succeeded, {error_count} failed\n"
        summary += "\n".join(results)

        return summary


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

            # Get positions with prices from MarketDataStore (includes FX-adjusted P&L)
            positions = self.portfolio_manager.get_positions_with_prices()
            total_value = self.portfolio_manager.get_portfolio_value()

            # Note: Use 'USD' instead of '$' to avoid Streamlit LaTeX interpretation
            summary = [
                f"Portfolio: {portfolio.name}",
                f"Total Value: {total_value:,.2f} {portfolio.base_currency.value}",
                f"Created: {portfolio.created_at.strftime('%Y-%m-%d')}",
                "",
            ]

            # Cash balances
            if portfolio.cash_balances:
                summary.append("Cash Balances:")
                for currency, amount in portfolio.cash_balances.items():
                    code = getattr(currency, "value", str(currency))
                    summary.append(f"- {code}: {amount:,.2f}")
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

                    # Get currency for display (use base currency from position data)
                    display_currency = pos.get('currency', 'USD')

                    # Format price (avoid $ to prevent Streamlit LaTeX issues)
                    price_str = f"{pos['current_price']:,.2f} {display_currency}" if pos['current_price'] else "N/A"

                    # Format P&L (avoid $ to prevent Streamlit LaTeX issues)
                    pnl_str = ""
                    if pos["unrealized_pnl"] is not None:
                        pnl = float(pos["unrealized_pnl"])
                        pnl_pct = float(pos["unrealized_pnl_percent"] or 0)
                        sign = "+" if pnl >= 0 else ""
                        pnl_str = f" ({sign}{pnl:,.2f} {display_currency}, {sign}{pnl_pct:.1f}%)"

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


class GetPortfolioSnapshotTool(BaseTool):
    """Tool for getting portfolio state at a specific historical date."""

    name: str = "get_portfolio_snapshot"
    description: str = (
        "Get portfolio positions, cash balances, and total value at a specific historical date. "
        "Use this to see what the portfolio looked like on any past date."
    )
    args_schema: type[BaseModel] = GetPortfolioSnapshotInput
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _run(self, target_date: str, include_local_currency: bool = False) -> str:
        """Get portfolio snapshot at a specific date."""
        try:
            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            # Parse date
            try:
                dt = datetime.strptime(target_date, "%Y-%m-%d").date()
            except ValueError:
                return f"❌ Invalid date format: {target_date}. Use YYYY-MM-DD."

            portfolio = self.portfolio_manager.current_portfolio
            base = portfolio.base_currency.value

            # Get positions at date
            positions = self.portfolio_manager.get_positions_with_prices(dt)

            # Get portfolio history for cash and total value
            history = self.portfolio_manager._get_portfolio_history()
            if not history:
                return "❌ Unable to calculate historical portfolio state."

            cash_balances = history.get_cash_at_date(dt)
            total_value = history.get_value_at_date(dt)

            # Build output
            lines = [
                f"📊 Portfolio Snapshot: {portfolio.name}",
                f"📅 Date: {target_date}",
                f"💰 Total Value: {total_value:,.2f} {base}",
                "",
            ]

            # Cash balances
            if cash_balances:
                lines.append("💵 Cash Balances:")
                for currency, amount in cash_balances.items():
                    currency_code = currency.value if hasattr(currency, 'value') else currency
                    lines.append(f"  • {currency_code}: {amount:,.2f}")
                lines.append("")

            # Positions
            if positions:
                lines.append(f"📈 Positions ({len(positions)}):")
                # Group by type
                by_type = {}
                for pos in positions:
                    itype = pos.get("instrument_type", "other")
                    if itype not in by_type:
                        by_type[itype] = []
                    by_type[itype].append(pos)

                for itype, type_positions in sorted(by_type.items()):
                    lines.append(f"  [{itype.upper()}]")
                    for pos in type_positions:
                        symbol = pos.get("symbol", "???")
                        qty = pos.get("quantity", 0)
                        price = pos.get("current_price")
                        market_value = pos.get("market_value")
                        pnl = pos.get("unrealized_pnl")
                        pnl_pct = pos.get("unrealized_pnl_percent")
                        local_ccy = pos.get("original_currency", base)
                        fx_rate = pos.get("fx_rate", 1)

                        price_str = f"@ {price:,.2f} {base}" if price else "(no price)"
                        value_str = f"= {market_value:,.2f} {base}" if market_value else ""

                        pnl_str = ""
                        if pnl is not None and pnl_pct is not None:
                            sign = "+" if pnl >= 0 else ""
                            pnl_str = f" ({sign}{pnl:,.2f} {base}, {sign}{pnl_pct:.1f}%)"

                        line = f"    • {symbol}: {qty:,.4g} {price_str} {value_str}{pnl_str}"

                        if include_local_currency and local_ccy != base and price and fx_rate and fx_rate != 1:
                            local_price = float(price) / float(fx_rate)
                            local_value = float(market_value) / float(fx_rate) if market_value else None
                            local_str = f" [local: {local_price:,.2f}"
                            if local_value:
                                local_str += f" = {local_value:,.2f}"
                            local_str += f" {local_ccy}]"
                            line += local_str

                        lines.append(line)
            else:
                lines.append("📈 Positions: None")

            return "\n".join(lines)

        except Exception as e:
            return f"❌ Error getting portfolio snapshot: {str(e)}"


class GetTransactionsTool(BaseTool):
    """Tool for exporting transactions and history with metrics-ready fields."""

    name: str = "get_transactions"
    description: str = (
        "Return transactions with optional filters. Supports date range, symbol, and type filtering. "
        "Use start_date/end_date for date ranges, symbol for specific instruments, limit to cap results."
    )
    args_schema: type[BaseModel] = GetTransactionsInput
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        symbol: Optional[str] = None,
        transaction_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> str:
        try:
            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            txns = self.portfolio_manager.current_portfolio.transactions

            # Parse date filters
            start_dt = None
            end_dt = None
            if start_date:
                try:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                except ValueError:
                    return f"❌ Invalid start_date format: {start_date}. Use YYYY-MM-DD."
            if end_date:
                try:
                    # End of day for end_date
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
                        hour=23, minute=59, second=59
                    )
                except ValueError:
                    return f"❌ Invalid end_date format: {end_date}. Use YYYY-MM-DD."

            # Apply filters
            filtered = []
            for t in txns:
                # Date filter
                if start_dt and t.timestamp < start_dt:
                    continue
                if end_dt and t.timestamp > end_dt:
                    continue
                # Symbol filter (case-insensitive)
                if symbol and t.instrument.symbol.upper() != symbol.upper():
                    continue
                # Transaction type filter
                if transaction_type and t.transaction_type.value.lower() != transaction_type.lower():
                    continue
                filtered.append(t)

            # Sort by timestamp
            filtered = sorted(filtered, key=lambda x: x.timestamp)

            # Apply limit (take most recent if limit specified)
            if limit and len(filtered) > limit:
                filtered = filtered[-limit:]

            # Build header with filter info
            filter_parts = []
            if start_date:
                filter_parts.append(f"from {start_date}")
            if end_date:
                filter_parts.append(f"to {end_date}")
            if symbol:
                filter_parts.append(f"symbol={symbol}")
            if transaction_type:
                filter_parts.append(f"type={transaction_type}")

            filter_desc = f" ({', '.join(filter_parts)})" if filter_parts else " (all)"
            total_count = len(self.portfolio_manager.current_portfolio.transactions)
            header = f"🧾 Transactions{filter_desc}: {len(filtered)}/{total_count}"

            lines = [
                header,
                "id,timestamp,symbol,type,quantity,price,currency,notes",
            ]
            for t in filtered:
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
        "Simulate portfolio history for a date range excluding certain symbols or transactions; returns end total value and basic stats."
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

            # Baseline (no exclusions) - returns DataFrame
            baseline_df = self.portfolio_manager.simulate_portfolio_history(
                start_date, end_date
            )

            # What-if with exclusions - returns DataFrame
            whatif_df = self.portfolio_manager.simulate_portfolio_history(
                start_date,
                end_date,
                exclude_symbols=symbols,
                exclude_transaction_ids=ids,
            )

            if whatif_df.empty or baseline_df.empty:
                return "No data generated for the specified range. Update market data first."

            end_value = float(whatif_df["total_value"].iloc[-1])
            start_value = float(whatif_df["total_value"].iloc[0])
            total_return = (
                ((end_value - start_value) / start_value * 100)
                if start_value > 0
                else 0.0
            )

            base_end = float(baseline_df["total_value"].iloc[-1])
            base_start = float(baseline_df["total_value"].iloc[0])
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
            projection_years: Years to project (0.5-10)
            monte_carlo_runs: Number of simulations (100-5000)
            modify_positions: Position modifications string, e.g.,
                            "AAPL:+50%,MSFT:-25%,GOOGL:=150" (increase by 50%, decrease by 25%, set to 150 shares)
            add_positions: New positions to add. Formats supported:
                         - With price: "NVDA:100@$800,TSLA:50@$250"
                         - Auto-fetch price: "NVDA:100,TSLA:50" (fetches current market prices)
                         - Mixed: "NVDA:100@$800,AAPL:25" (explicit price for NVDA, auto-fetch for AAPL)
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
            current_snapshot = self.portfolio_manager.create_current_snapshot()

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
        """Parse new position string like 'NVDA:100@$800,TSLA:50@$250' or 'NVDA:100' (auto-fetch price)."""
        new_positions = {}
        if not add_string:
            return new_positions

        for item in add_string.split(','):
            if ':' not in item:
                continue

            symbol, details = item.strip().split(':', 1)
            symbol = symbol.strip().upper()

            if '@' in details:
                quantity_str, price_str = details.split('@', 1)
                quantity = float(quantity_str.strip())
                price_str = price_str.strip().replace('$', '')

                # If price is empty or zero, auto-fetch current market price
                if not price_str or float(price_str) <= 0:
                    fetched_price = self.portfolio_manager.data_manager.get_current_price(symbol)
                    if fetched_price is None:
                        continue  # Skip this position if we can't get the price
                    price = float(fetched_price)
                else:
                    price = float(price_str)

                new_positions[symbol] = {
                    'quantity': quantity,
                    'price': price
                }
            else:
                # No @ sign - just quantity, auto-fetch price
                quantity = float(details.strip())
                fetched_price = self.portfolio_manager.data_manager.get_current_price(symbol)
                if fetched_price is None:
                    continue  # Skip this position if we can't get the price
                price = float(fetched_price)

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
        from decimal import Decimal
        from src.portfolio.scenarios import (
            ScenarioConfiguration,
            ScenarioType,
            MarketAssumptions,
            AssetClassAssumptions,
            PortfolioScenarioEngine,
        )
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

        # If user didn't override market assumptions, use predefined scenario settings
        use_predefined = (
            mapped_scenario_type in {
                ScenarioType.OPTIMISTIC,
                ScenarioType.LIKELY,
                ScenarioType.PESSIMISTIC,
                ScenarioType.STRESS,
            }
            and not stress_test
            and market_return == 0.08
            and market_volatility == 0.20
        )

        if use_predefined:
            predefined = PortfolioScenarioEngine().create_predefined_scenarios(
                Decimal("0")
            ).get(mapped_scenario_type.value)
            if predefined:
                return ScenarioConfiguration(
                    scenario_type=mapped_scenario_type,
                    name=predefined.name,
                    description=predefined.description,
                    projection_years=projection_years,
                    monte_carlo_runs=monte_carlo_runs,
                    market_assumptions=predefined.market_assumptions,
                    asset_class_assumptions=predefined.asset_class_assumptions,
                    recurring_deposits=recurring_deposits,
                )

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
        "under different market scenarios. Returns the symbol's historical volatility, risk level, "
        "annual return, and Monte Carlo projections. Provide a symbol and either quantity or "
        "investment_amount. If purchase_price is omitted or 0, the current market price will be "
        "fetched automatically."
    )
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _run(
        self,
        symbol: str,
        quantity: float = 0,
        purchase_price: float = 0,
        investment_amount: str = "",
        scenario: str = "likely",
        time_horizon: float = 1.0,
    ) -> str:
        """Test a hypothetical position in the portfolio.

        Args:
            symbol: Stock symbol to test (e.g., "AAPL", "MSFT")
            quantity: Number of shares to hypothetically purchase (use 0 if using investment_amount)
            purchase_price: Price per share (0 or omit to auto-fetch current market price)
            investment_amount: Alternative to quantity - dollar amount to invest (e.g., "$5000")
            scenario: Market scenario to test (optimistic, likely, pessimistic, stress)
            time_horizon: Years to project the investment (0.5 to 5.0)
        """
        try:
            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            symbol = symbol.upper().strip()

            # Auto-fetch current market price if not provided
            if purchase_price <= 0:
                fetched_price = self.portfolio_manager.data_manager.get_current_price(symbol)
                if fetched_price is None:
                    return f"❌ Could not fetch current price for {symbol}. Please provide a purchase_price or check the symbol."
                purchase_price = float(fetched_price)

            # Fetch historical data to calculate volatility and returns
            symbol_volatility, symbol_annual_return, price_change_1y = self._calculate_symbol_metrics(symbol)

            # Parse investment amount if provided
            if investment_amount:
                amount_str = investment_amount.replace('$', '').replace(',', '')
                try:
                    amount = float(amount_str)
                    quantity = amount / purchase_price
                except ValueError:
                    return f"❌ Invalid investment amount: {investment_amount}"

            if quantity <= 0:
                return "❌ Quantity must be positive. Provide either quantity or investment_amount."

            # Validate parameters
            time_horizon = max(0.5, min(5.0, time_horizon))
            symbol = symbol.upper().strip()

            # Get current portfolio value for context
            current_snapshot = self.portfolio_manager.create_current_snapshot()
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

            # Build symbol metrics section
            metrics_lines = []
            if symbol_volatility is not None:
                metrics_lines.append(f"   • Historical Volatility: {symbol_volatility:.1f}% (annualized)")
            if symbol_annual_return is not None:
                metrics_lines.append(f"   • Historical Annual Return: {symbol_annual_return:+.1f}%")
            if price_change_1y is not None:
                metrics_lines.append(f"   • 1-Year Price Change: {price_change_1y:+.1f}%")

            # Determine risk level based on volatility
            risk_level = "Unknown"
            if symbol_volatility is not None:
                if symbol_volatility < 20:
                    risk_level = "🟢 Low"
                elif symbol_volatility < 35:
                    risk_level = "🟡 Moderate"
                elif symbol_volatility < 50:
                    risk_level = "🟠 High"
                else:
                    risk_level = "🔴 Very High"
                metrics_lines.append(f"   • Risk Level: {risk_level}")

            header = [
                f"🧪 Hypothetical Position Analysis",
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"",
                f"💼 Hypothetical Investment:",
                f"   • Symbol: {symbol}",
                f"   • Quantity: {quantity:,.2f} shares",
                f"   • Purchase Price: {purchase_price:,.2f} USD",
                f"   • Total Investment: {investment_value:,.2f} USD ({investment_pct:.1f}% of portfolio)",
                f"   • Scenario: {scenario.title()}",
                f"   • Time Horizon: {time_horizon:.1f} years",
                f"",
            ]

            # Add symbol metrics section if we have any data
            if metrics_lines:
                header.extend([
                    f"📈 Symbol Metrics ({symbol}):",
                ] + metrics_lines + [""])

            header.extend([
                f"📊 Impact on Portfolio:",
                ""
            ])

            return "\n".join(header) + result.split("📊 Portfolio Modifications Applied:")[1] if "📊 Portfolio Modifications Applied:" in result else result

        except Exception as e:
            return f"❌ Error testing hypothetical position: {str(e)}"

    def _calculate_symbol_metrics(self, symbol: str) -> tuple:
        """Calculate historical volatility and returns for a symbol.

        Returns:
            Tuple of (annualized_volatility_pct, annualized_return_pct, 1y_price_change_pct)
            Any value may be None if data is unavailable.
        """
        import numpy as np
        from datetime import date, timedelta

        try:
            # Get 1 year of historical data
            end_date = date.today()
            start_date = end_date - timedelta(days=365)

            prices = self.portfolio_manager.data_manager.get_historical_prices(
                symbol, start_date, end_date
            )

            if not prices or len(prices) < 20:
                # Not enough data for meaningful calculations
                return None, None, None

            # Extract close prices and sort by date
            price_data = sorted(
                [(p.date, float(p.close_price)) for p in prices if p.close_price],
                key=lambda x: x[0]
            )

            if len(price_data) < 20:
                return None, None, None

            closes = [p[1] for p in price_data]

            # Calculate daily returns
            returns = []
            for i in range(1, len(closes)):
                if closes[i-1] > 0:
                    daily_return = (closes[i] - closes[i-1]) / closes[i-1]
                    returns.append(daily_return)

            if len(returns) < 10:
                return None, None, None

            # Annualized volatility (std dev of daily returns * sqrt(252 trading days))
            daily_std = np.std(returns)
            annualized_volatility = daily_std * np.sqrt(252) * 100  # Convert to percentage

            # Annualized return (compound daily returns)
            avg_daily_return = np.mean(returns)
            annualized_return = ((1 + avg_daily_return) ** 252 - 1) * 100  # Convert to percentage

            # 1-year price change
            first_price = closes[0]
            last_price = closes[-1]
            price_change_1y = ((last_price - first_price) / first_price) * 100 if first_price > 0 else None

            return annualized_volatility, annualized_return, price_change_1y

        except Exception:
            # If anything fails, return None values - don't break the main analysis
            return None, None, None


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


class CheckMarketDataAvailabilityTool(BaseTool):
    """Tool for checking if market data is available for an instrument without fetching full data."""

    name: str = "check_market_data_availability"
    description: str = """Check if market data is available for an instrument without fetching the full data.

    Use this tool to verify data access BEFORE attempting to fetch historical data or add positions.

    Accepts:
    - ISIN (e.g., US0378331005)
    - Symbol (e.g., AAPL, TSLA)
    - Name (e.g., Apple, Tesla)

    Returns:
    - Whether the instrument is recognized by data providers
    - Whether price data can be fetched (if verify_price_data=True)
    - Basic instrument information if found

    This is a lightweight check that doesn't fetch full historical data.
    """
    args_schema: type[BaseModel] = CheckMarketDataAvailabilityInput
    data_manager: Optional[DataProviderManager] = None

    def __init__(self, data_manager: DataProviderManager):
        super().__init__()
        self.data_manager = data_manager

    def _run(
        self,
        isin: Optional[str] = None,
        symbol: Optional[str] = None,
        name: Optional[str] = None,
        verify_price_data: bool = True,
    ) -> str:
        """Check if market data is available for the specified instrument."""
        try:
            resolved_symbol = None
            instrument_info = None
            lookup_method = None

            # Priority 1: ISIN lookup
            if isin:
                isin = isin.strip().upper()
                lookup_method = "ISIN"
                instrument_info = self.data_manager.search_by_isin(isin)
                if instrument_info:
                    resolved_symbol = instrument_info.symbol

            # Priority 2: Symbol lookup
            if not resolved_symbol and symbol:
                symbol = symbol.strip().upper()
                lookup_method = "symbol"
                # Validate symbol exists
                if self.data_manager.validate_symbol(symbol):
                    resolved_symbol = symbol
                    instrument_info = self.data_manager.get_instrument_info(symbol)

            # Priority 3: Name search
            if not resolved_symbol and name:
                name = name.strip()
                lookup_method = "name"
                # Try company name search
                results = self.data_manager.search_by_company_name(name)
                if not results:
                    results = self.data_manager.search_instruments(name)
                if results:
                    # Return list of candidates - don't auto-select
                    return self._format_search_candidates(results, name)

            # No input provided
            if not isin and not symbol and not name:
                return "Please provide at least one of: isin, symbol, or name to check."

            # Instrument not found
            if not resolved_symbol:
                return self._format_not_found(isin, symbol, name, lookup_method)

            # Instrument found - now check price data availability if requested
            price_available = False
            current_price = None

            if verify_price_data:
                current_price = self.data_manager.get_current_price(resolved_symbol)
                price_available = current_price is not None

            return self._format_availability_result(
                resolved_symbol=resolved_symbol,
                instrument_info=instrument_info,
                lookup_method=lookup_method,
                lookup_value=isin or symbol,
                price_available=price_available,
                current_price=current_price,
                verify_price_data=verify_price_data,
            )

        except Exception as e:
            return f"Error checking market data availability: {str(e)}"

    def _format_availability_result(
        self,
        resolved_symbol: str,
        instrument_info,
        lookup_method: str,
        lookup_value: str,
        price_available: bool,
        current_price,
        verify_price_data: bool,
    ) -> str:
        """Format the availability check result."""
        lines = ["MARKET DATA AVAILABILITY CHECK", "=" * 30, ""]

        # Overall status
        if verify_price_data:
            if price_available:
                lines.append("Status: AVAILABLE - Market data can be fetched")
            else:
                lines.append("Status: PARTIAL - Instrument found but price data unavailable")
        else:
            lines.append("Status: FOUND - Instrument recognized by data providers")

        lines.append("")
        lines.append(f"Lookup method: {lookup_method} ({lookup_value})")
        lines.append(f"Resolved symbol: {resolved_symbol}")

        if instrument_info:
            lines.append("")
            lines.append("Instrument Details:")
            lines.append(f"  Name: {instrument_info.name}")
            lines.append(f"  Type: {instrument_info.instrument_type.value}")
            lines.append(f"  Currency: {instrument_info.currency.value}")
            if instrument_info.isin:
                lines.append(f"  ISIN: {instrument_info.isin}")
            if instrument_info.exchange:
                lines.append(f"  Exchange: {instrument_info.exchange}")

        if verify_price_data:
            lines.append("")
            lines.append("Price Data Check:")
            if price_available:
                lines.append(f"  Current price: {current_price}")
                lines.append("  Historical data: Likely available")
            else:
                lines.append("  Current price: Not available")
                lines.append("  Note: Price data may be delayed or unavailable for this instrument")

        return "\n".join(lines)

    def _format_not_found(
        self,
        isin: Optional[str],
        symbol: Optional[str],
        name: Optional[str],
        lookup_method: str,
    ) -> str:
        """Format a not-found result."""
        lines = ["MARKET DATA AVAILABILITY CHECK", "=" * 30, ""]
        lines.append("Status: NOT FOUND - Instrument not recognized")
        lines.append("")

        if lookup_method == "ISIN":
            lines.append(f"ISIN '{isin}' was not found in any data provider.")
            lines.append("Suggestions:")
            lines.append("  - Verify the ISIN is correct")
            lines.append("  - Try searching by symbol or company name instead")
        elif lookup_method == "symbol":
            lines.append(f"Symbol '{symbol}' was not found in any data provider.")
            lines.append("Suggestions:")
            lines.append("  - Check the symbol spelling")
            lines.append("  - Try searching by company name")
            lines.append("  - The instrument may not be covered by available data providers")
        else:
            lines.append(f"No results found for '{name}'.")
            lines.append("Suggestions:")
            lines.append("  - Try a different search term")
            lines.append("  - Use the exact symbol or ISIN if known")

        return "\n".join(lines)

    def _format_search_candidates(self, results: list, search_term: str) -> str:
        """Format search candidates when multiple matches found."""
        lines = ["MARKET DATA AVAILABILITY CHECK", "=" * 30, ""]
        lines.append(f"Status: MULTIPLE MATCHES - Found {len(results)} potential instruments")
        lines.append("")
        lines.append(f"Search term: '{search_term}'")
        lines.append("")
        lines.append("Matching instruments:")

        for i, instrument in enumerate(results[:5], 1):
            lines.append(f"  {i}. {instrument.symbol} - {instrument.name}")
            lines.append(f"     Type: {instrument.instrument_type.value}, Currency: {instrument.currency.value}")
            if instrument.isin:
                lines.append(f"     ISIN: {instrument.isin}")

        if len(results) > 5:
            lines.append(f"  ... and {len(results) - 5} more")

        lines.append("")
        lines.append("To check availability for a specific instrument, use its symbol or ISIN.")

        return "\n".join(lines)


class GetCurrentPriceTool(BaseTool):
    """Tool for getting current price of an instrument."""

    name: str = "get_current_price"
    description: str = (
        "Get the current market price of a stock, bond, ETF, or other financial instrument."
    )
    args_schema: type[BaseModel] = GetPriceInput
    data_manager: Optional[DataProviderManager] = None
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(
        self,
        data_manager: DataProviderManager,
        portfolio_manager: Optional[PortfolioManager] = None,
    ):
        super().__init__()
        self.data_manager = data_manager
        self.portfolio_manager = portfolio_manager

    def _run(self, symbol: str) -> str:
        """Get current price."""
        try:
            portfolio_symbol = symbol.upper().strip()

            # Check if we have a data_provider_symbol mapping for this symbol
            lookup_symbol = portfolio_symbol
            if self.portfolio_manager and self.portfolio_manager.current_portfolio:
                position = self.portfolio_manager.current_portfolio.positions.get(portfolio_symbol)
                if position and position.instrument.data_provider_symbol:
                    lookup_symbol = position.instrument.data_provider_symbol.upper().strip()
                    logging.info(f"Using stored data_provider_symbol '{lookup_symbol}' for {portfolio_symbol}")

            price = self.data_manager.get_current_price(lookup_symbol)

            if price is None:
                return f"❌ Could not get current price for {portfolio_symbol}"

            # Apply price_currency conversion if set
            display_currency = None
            if self.portfolio_manager and self.portfolio_manager.current_portfolio:
                position = self.portfolio_manager.current_portfolio.positions.get(portfolio_symbol)
                if position:
                    price_currency = position.instrument.price_currency
                    target_currency = position.instrument.currency
                    if price_currency and price_currency != target_currency:
                        fx_rate = self.portfolio_manager._get_exchange_rate(price_currency, target_currency)
                        if fx_rate:
                            price = price * fx_rate
                    display_currency = target_currency.value

            # Also get instrument info for context
            info = self.data_manager.get_instrument_info(lookup_symbol)
            name = info.name if info else portfolio_symbol
            currency_label = display_currency or (info.currency.value if info and info.currency else "USD")

            return f"💰 **{portfolio_symbol}** ({name}): {price:.2f} {currency_label}"

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

    def _run(self, days: int = 365, benchmark: str = "SPY", start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
        """Calculate portfolio metrics."""
        try:
            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            # Resolve date range: explicit start/end take priority over days
            if start_date:
                start_date = date.fromisoformat(start_date)
                end_date = date.fromisoformat(end_date) if end_date else date.today()
            else:
                end_date = date.today()
                start_date = end_date - timedelta(days=days)

            try:
                history_df = self.portfolio_manager.get_portfolio_history(start_date, end_date)
            except Exception as e:
                return f"❌ Error loading portfolio history: {str(e)}"

            if history_df.empty or len(history_df) < 2:
                return "❌ Insufficient historical data for metrics calculation. Need at least 2 data points. Update market data first."

            metrics = self.metrics_calculator.calculate_metrics_from_df(
                history_df, benchmark_symbol=benchmark
            )

            if "error" in metrics:
                return f"❌ {metrics['error']}"

            result = [
                f"📊 **Portfolio Metrics** ({start_date} → {end_date} vs {benchmark})",
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


class ModifyTransactionTool(BaseTool):
    """Tool for modifying existing transactions."""

    name: str = "modify_transaction"
    description: str = """Modify an existing transaction by its ID.

    You can modify:
    - quantity: New number of shares/units
    - price: New price per share/unit
    - date: New transaction date (YYYY-MM-DD)
    - notes: New notes for the transaction
    - instrument_type: Type of instrument (stock, etf, bond, crypto, cash, mutual_fund, option, future)

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
        instrument_type: Optional[str] = None,
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

            if instrument_type is not None:
                from ..portfolio.models import InstrumentType
                try:
                    new_type = InstrumentType(instrument_type.lower())
                    old_type = transaction.instrument.instrument_type.value
                    transaction.instrument.instrument_type = new_type
                    modifications.append(f"instrument_type: {old_type} → {instrument_type}")
                except ValueError:
                    valid_types = [t.value for t in InstrumentType]
                    return f"❌ Invalid instrument_type. Valid options: {', '.join(valid_types)}"

            if not modifications:
                return "❌ No modifications specified. Provide at least one field to modify."

            # Recalculate positions
            portfolio.recalculate_positions()

            # Save the portfolio
            self.portfolio_manager.storage.save_portfolio(portfolio)

            # Invalidate portfolio history cache since transactions changed
            self.portfolio_manager._invalidate_portfolio_history()

            return (
                f"✅ Modified transaction {transaction_id[:8]}...\n"
                f"Changes: {', '.join(modifications)}"
            )

        except Exception as e:
            return f"❌ Error modifying transaction: {str(e)}"


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

            # Invalidate portfolio history cache since transactions changed
            self.portfolio_manager._invalidate_portfolio_history()

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


class SetMarketPriceTool(BaseTool):
    """Tool for setting or updating market prices for instruments."""

    name: str = "set_market_price"
    description: str = """Set or update the market price for an instrument on a specific date.

    Use cases:
    - When price lookup fails and user wants to use purchase price as market price
    - When user wants to manually set a custom price for an instrument
    - When correcting historical prices in market data

    Parameters:
    - symbol: The instrument symbol (required)
    - price: The price to set (required unless use_purchase_price is True)
    - currency: Currency of the price (e.g., USD, CHF, EUR). If different from the
                instrument's native currency, it will be automatically converted.
                Defaults to the instrument's native currency if not specified.
    - date: Date for the price in YYYY-MM-DD format (optional, defaults to today)
    - use_purchase_price: If True, uses the position's average_cost as the market price

    Examples:
    - "Set AAPL price to $150" → set_market_price(symbol="AAPL", price=150, currency="USD")
    - "Set VTEQ_SWISS to CHF 176.12" → set_market_price(symbol="VTEQ_SWISS", price=176.12, currency="CHF")
    - "Set VTEQ_SWISS to $225 USD" → set_market_price(symbol="VTEQ_SWISS", price=225, currency="USD")
    - "Use purchase price for XYZ" → set_market_price(symbol="XYZ", use_purchase_price=True)
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
        currency: Optional[str] = None,
        date: Optional[str] = None,
        use_purchase_price: bool = False,
    ) -> str:
        """Set market price for an instrument."""
        try:
            from ..portfolio.models import Currency as CurrencyEnum

            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            symbol = symbol.upper().strip()

            # Check if symbol exists in current portfolio
            position = self.portfolio_manager.current_portfolio.positions.get(symbol)
            is_sold_instrument = position is None

            # Parse currency if provided
            price_currency = None
            if currency:
                try:
                    price_currency = CurrencyEnum(currency.upper())
                except ValueError:
                    valid_currencies = [c.value for c in CurrencyEnum]
                    return f"❌ Invalid currency '{currency}'. Valid options: {', '.join(valid_currencies)}"

            # Determine the price to use
            if use_purchase_price:
                if is_sold_instrument:
                    return f"❌ Cannot use purchase price for sold instrument '{symbol}'. Please provide an explicit price."
                final_price = position.average_cost
                price_source = "purchase price (average cost)"
                # When using purchase price, currency is the instrument's native currency
                price_currency = position.instrument.currency
            elif price is not None:
                final_price = Decimal(str(price))
                price_source = f"custom price"
                if price_currency:
                    price_source += f" in {price_currency.value}"
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

            # For sold instruments, we can only update historical market data
            if is_sold_instrument and target_date is None:
                return (
                    f"❌ Symbol '{symbol}' is a sold instrument (no current position). "
                    f"Please specify a date to update historical market data."
                )

            # Set the price with currency
            success = self.portfolio_manager.set_position_price(
                symbol=symbol,
                price=final_price,
                target_date=target_date,
                update_current=not is_sold_instrument,
                currency=price_currency,
            )

            if success:
                target_date_str = (date if date else "today")
                # Get native currency for display
                native_currency = position.instrument.currency.value if position else "N/A"
                currency_note = ""
                if price_currency and position and price_currency != position.instrument.currency:
                    currency_note = f"\n• Note: Converted from {price_currency.value} to {native_currency}"

                if is_sold_instrument:
                    return (
                        f"✅ Set historical price for sold instrument {symbol}:\n"
                        f"• Price: {final_price:.2f} ({price_source})\n"
                        f"• Date: {target_date_str}\n"
                        f"• Note: Updated historical market data (instrument was sold)"
                    )
                else:
                    return (
                        f"✅ Set market price for {symbol}:\n"
                        f"• Price: {final_price:.2f} ({price_source})\n"
                        f"• Native currency: {native_currency}\n"
                        f"• Date: {target_date_str}\n"
                        f"• Position: {position.instrument.name}\n"
                        f"• Quantity: {position.quantity}{currency_note}"
                    )
            else:
                if is_sold_instrument:
                    return (
                        f"❌ Failed to set price for sold instrument {symbol}. "
                        f"Make sure market data exists for {date or 'today'}."
                    )
                return f"❌ Failed to set price for {symbol}. Check logs for details."

        except Exception as e:
            return f"❌ Error setting market price: {str(e)}"


class FetchAndUpdatePricesTool(BaseTool):
    """Tool for fetching prices from data provider and updating market data."""

    name: str = "fetch_and_update_prices"
    description: str = """Fetch historical prices from the data provider and update market data.

    Use cases:
    - Update historical prices for an instrument from market data
    - Sync portfolio with latest market prices for a date range
    - Handle cases where portfolio symbol differs from provider symbol

    Parameters:
    - symbol: The symbol in your portfolio (required)
    - start_date: Start date in YYYY-MM-DD format (required)
    - end_date: End date in YYYY-MM-DD format (required)
    - provider_symbol: The symbol to use with the data provider (optional)
                       Use when portfolio symbol differs from provider symbol
                       Example: Portfolio has "BTC" but provider needs "BTC-USD"

    Examples:
    - Update AAPL prices: fetch_and_update_prices(symbol="AAPL", start_date="2024-01-01", end_date="2024-01-31")
    - Update Bitcoin (BTC in portfolio, BTC-USD for provider):
      fetch_and_update_prices(symbol="BTC", start_date="2024-01-01", end_date="2024-01-31", provider_symbol="BTC-USD")

    If this fails, use bulk_set_market_price to manually enter prices.
    """
    args_schema: type[BaseModel] = FetchAndUpdatePricesInput
    portfolio_manager: Optional[PortfolioManager] = None
    data_manager: Optional[DataProviderManager] = None

    def __init__(self, portfolio_manager: PortfolioManager, data_manager: DataProviderManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager
        self.data_manager = data_manager

    def _run(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        provider_symbol: Optional[str] = None,
    ) -> str:
        """Fetch prices from provider and update portfolio."""
        try:
            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            portfolio_symbol = symbol.upper().strip()

            # Check if symbol exists in current portfolio
            position = self.portfolio_manager.current_portfolio.positions.get(portfolio_symbol)
            is_sold_instrument = position is None

            # Determine the lookup symbol:
            # 1. Use explicitly provided provider_symbol if given
            # 2. Otherwise, check if position has a stored data_provider_symbol
            # 3. Fall back to the portfolio symbol
            if provider_symbol:
                lookup_symbol = provider_symbol.upper().strip()
            elif position and position.instrument.data_provider_symbol:
                lookup_symbol = position.instrument.data_provider_symbol.upper().strip()
                logging.info(f"Using stored data_provider_symbol '{lookup_symbol}' for {portfolio_symbol}")
            else:
                lookup_symbol = portfolio_symbol

            # For sold instruments, we can still update historical market data entries
            # but we inform the user about limitations
            if is_sold_instrument:
                logging.info(f"Updating prices for sold instrument {portfolio_symbol}")

            # Parse dates
            try:
                from datetime import datetime as dt
                start = dt.strptime(start_date, "%Y-%m-%d").date()
                end = dt.strptime(end_date, "%Y-%m-%d").date()
            except ValueError:
                return "❌ Invalid date format. Use YYYY-MM-DD."

            if start > end:
                return "❌ Start date must be before or equal to end date."

            # If an ad-hoc provider_symbol was given, apply it to the instrument
            # so _fetch_and_store_prices picks it up automatically
            if provider_symbol and position:
                position.instrument.data_provider_symbol = lookup_symbol

            instrument = position.instrument if position else None
            success_count, failed_date_list = self.portfolio_manager._fetch_and_store_prices(
                portfolio_symbol, start, end, instrument
            )
            failed_dates = [str(d) for d in failed_date_list]

            if success_count == 0:
                return (
                    f"❌ No price data returned for '{lookup_symbol}' from {start_date} to {end_date}.\n\n"
                    f"Possible reasons:\n"
                    f"• Symbol not found in data provider\n"
                    f"• No trading data for this date range\n"
                    f"• Data provider API issue\n\n"
                    f"💡 Try:\n"
                    f"• Use a different provider_symbol (e.g., 'BTC-USD' instead of 'BTC')\n"
                    f"• Use bulk_set_market_price to manually enter prices"
                )

            # Build result message
            if is_sold_instrument:
                result_lines = [
                    f"✅ Price update for sold instrument {portfolio_symbol}:",
                    f"• Data source: {lookup_symbol}",
                    f"• Successfully updated: {success_count} historical market data entries",
                    f"• Date range: {start_date} to {end_date}",
                    f"• Note: This is a sold instrument - only historical market data entries were updated",
                ]
            else:
                result_lines = [
                    f"✅ Price update for {portfolio_symbol} ({position.instrument.name}):",
                    f"• Data source: {lookup_symbol}" + (" (provider symbol - saved for future updates)" if provider_symbol else ""),
                    f"• Successfully updated: {success_count} prices",
                    f"• Date range: {start_date} to {end_date}",
                ]

            if failed_dates:
                result_lines.append(f"• Failed/skipped dates: {len(failed_dates)}")
                if len(failed_dates) <= 5:
                    result_lines.append(f"  {', '.join(failed_dates)}")

            return "\n".join(result_lines)

        except Exception as e:
            return f"❌ Error fetching and updating prices: {str(e)}"


class SetDataProviderSymbolTool(BaseTool):
    """Tool for setting the data provider symbol for a portfolio position."""

    name: str = "set_data_provider_symbol"
    description: str = """Set the data provider symbol for a portfolio position.

    Use this when the portfolio symbol differs from the symbol used by the data provider.
    Once set, all future price lookups (Quick Refresh, Update Market Data) will use the
    data provider symbol automatically.

    Examples:
    - Bitcoin: set_data_provider_symbol(symbol="BTC", data_provider_symbol="BTC-USD")
    - Ethereum: set_data_provider_symbol(symbol="ETH", data_provider_symbol="ETH-USD")
    - Custom bond: set_data_provider_symbol(symbol="CORP_BOND", data_provider_symbol="LQD")

    This is automatically done when using fetch_and_update_prices with a provider_symbol,
    but you can use this tool to set it manually.
    """
    args_schema: type[BaseModel] = SetDataProviderSymbolInput
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _run(
        self,
        symbol: str,
        data_provider_symbol: str,
    ) -> str:
        """Set the data provider symbol for a position."""
        try:
            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            portfolio_symbol = symbol.upper().strip()
            provider_symbol = data_provider_symbol.upper().strip()

            if portfolio_symbol not in self.portfolio_manager.current_portfolio.positions:
                return f"❌ Symbol '{portfolio_symbol}' not found in portfolio."

            success = self.portfolio_manager.set_data_provider_symbol(
                portfolio_symbol, provider_symbol
            )

            if success:
                position = self.portfolio_manager.current_portfolio.positions[portfolio_symbol]
                return (
                    f"✅ Data provider symbol set for {portfolio_symbol}:\n"
                    f"• Instrument: {position.instrument.name}\n"
                    f"• Portfolio symbol: {portfolio_symbol}\n"
                    f"• Data provider symbol: {provider_symbol}\n\n"
                    f"Future price updates (Quick Refresh, Update Market Data) will now use '{provider_symbol}' to fetch prices."
                )
            else:
                return f"❌ Failed to set data provider symbol for {portfolio_symbol}."

        except Exception as e:
            return f"❌ Error setting data provider symbol: {str(e)}"


class SetPriceCurrencyTool(BaseTool):
    """Tool for setting the price currency for a portfolio position."""

    name: str = "set_price_currency"
    description: str = """Set the price currency for a portfolio position.

    Use this when the data provider returns prices in a different currency than the
    instrument is tracked in. The system will automatically convert fetched prices
    to the instrument's portfolio currency before storing.

    Example workflow for CNKY (tracked in JPY, listed on LSE in GBX):
      1. set_data_provider_symbol(symbol="CNKY", data_provider_symbol="CNKY.L")
      2. set_price_currency(symbol="CNKY", price_currency="GBP")
    → Future price refreshes will fetch CNKY.L in GBX, auto-convert to GBP, then to JPY.

    Note: GBX (pence) is automatically converted to GBP by the data provider before
    this currency conversion step runs, so always use GBP (not GBX) as the price_currency
    for LSE-listed instruments.
    """
    args_schema: type[BaseModel] = SetPriceCurrencyInput
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _run(self, symbol: str, price_currency: str) -> str:
        """Set the price currency for a position."""
        try:
            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            portfolio_symbol = symbol.upper().strip()

            if portfolio_symbol not in self.portfolio_manager.current_portfolio.positions:
                return f"❌ Symbol '{portfolio_symbol}' not found in portfolio."

            from ..portfolio.models import Currency
            try:
                currency = Currency(price_currency.upper().strip())
            except ValueError:
                supported = ", ".join(c.value for c in Currency)
                return f"❌ Unsupported currency '{price_currency}'. Supported: {supported}"

            success = self.portfolio_manager.set_price_currency(portfolio_symbol, currency)

            if success:
                position = self.portfolio_manager.current_portfolio.positions[portfolio_symbol]
                target_currency = position.instrument.currency.value
                return (
                    f"✅ Price currency set for {portfolio_symbol}:\n"
                    f"• Instrument: {position.instrument.name}\n"
                    f"• Provider returns prices in: {currency.value}\n"
                    f"• Prices will be stored in: {target_currency}\n\n"
                    f"Future price refreshes will automatically convert {currency.value} → {target_currency} before storing."
                )
            else:
                return f"❌ Failed to set price currency for {portfolio_symbol}."

        except Exception as e:
            return f"❌ Error setting price currency: {str(e)}"


class BulkSetMarketPriceTool(BaseTool):
    """Tool for bulk setting market prices for one or more instruments across multiple dates."""

    name: str = "bulk_set_market_price"
    description: str = """Set market prices for one or more instruments across multiple dates at once.

    Use cases:
    - Entering historical price data manually when market data isn't available
    - Importing price history for instruments without data providers
    - Correcting multiple historical prices in market data
    - Bulk updating prices for multiple symbols in a single call

    Parameters:
    - symbol: The instrument symbol (optional - only needed for single-symbol formats)
    - prices: Price data in one of three formats:
        1. Simple format (requires symbol): "YYYY-MM-DD:price,YYYY-MM-DD:price,..."
        2. Single-symbol JSON (requires symbol): '[{"date":"2024-01-01","price":150.0},{"date":"2024-01-02","price":152.5}]'
        3. Multi-symbol JSON (symbol not needed): '[{"symbol":"AAPL","date":"2024-01-01","price":150.0,"currency":"USD"}]'
    - currency: Currency for all prices (for simple/single-symbol formats). Defaults to instrument's native currency.

    Examples:
    - Simple format with currency:
      bulk_set_market_price(symbol="VTEQ_SWISS", prices="2024-01-01:176.12,2024-01-02:178.50", currency="CHF")

    - Single-symbol JSON:
      bulk_set_market_price(symbol="AAPL", prices='[{"date":"2024-01-01","price":150.0}]', currency="USD")

    - Multi-symbol JSON with per-entry currency:
      bulk_set_market_price(prices='[{"symbol":"AAPL","date":"2024-01-01","price":150.0,"currency":"USD"},{"symbol":"VTEQ_SWISS","date":"2024-01-01","price":176.12,"currency":"CHF"}]')
    """
    args_schema: type[BaseModel] = BulkSetMarketPriceInput
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _parse_prices(self, prices: str) -> tuple[list, list, bool, dict]:
        """Parse prices from simple, single-symbol JSON, or multi-symbol JSON format.

        Returns:
            Tuple of (price_entries, errors, is_multi_symbol, entry_currencies) where:
            - price_entries is list of (date, Decimal) tuples for single-symbol format
            - price_entries is list of (symbol, date, Decimal) tuples for multi-symbol format
            - is_multi_symbol indicates which format was detected
            - entry_currencies is dict mapping (symbol, date) -> currency for multi-symbol format
        """
        import json
        from datetime import datetime as dt
        from ..portfolio.models import Currency as CurrencyEnum

        price_entries = []
        errors = []
        entry_currencies = {}  # (symbol, date) -> Currency for multi-symbol format

        # Try JSON format first
        prices_stripped = prices.strip()
        if prices_stripped.startswith("["):
            try:
                data = json.loads(prices_stripped)
                if not data:
                    return [], ["Empty JSON array"], False, {}

                # Detect if this is multi-symbol format (items have "symbol" key)
                is_multi_symbol = isinstance(data[0], dict) and "symbol" in data[0]

                for item in data:
                    if not isinstance(item, dict):
                        errors.append(f"Invalid item in JSON array: {item}")
                        continue

                    date_str = item.get("date")
                    price_val = item.get("price")

                    if not date_str or price_val is None:
                        errors.append(f"Missing 'date' or 'price' in: {item}")
                        continue

                    try:
                        entry_date = dt.strptime(str(date_str), "%Y-%m-%d").date()
                    except ValueError:
                        errors.append(f"Invalid date '{date_str}' - use YYYY-MM-DD format")
                        continue

                    try:
                        entry_price = Decimal(str(price_val))
                        if entry_price <= 0:
                            errors.append(f"Invalid price '{price_val}' - must be positive")
                            continue
                    except Exception:
                        errors.append(f"Invalid price '{price_val}' - must be a number")
                        continue

                    # Parse currency if provided in JSON entry
                    entry_currency = None
                    if "currency" in item:
                        try:
                            entry_currency = CurrencyEnum(item["currency"].upper())
                        except ValueError:
                            errors.append(f"Invalid currency '{item['currency']}' in: {item}")
                            continue

                    if is_multi_symbol:
                        symbol_str = item.get("symbol")
                        if not symbol_str:
                            errors.append(f"Missing 'symbol' in multi-symbol format: {item}")
                            continue
                        symbol = symbol_str.upper().strip()
                        price_entries.append((symbol, entry_date, entry_price))
                        if entry_currency:
                            entry_currencies[(symbol, entry_date)] = entry_currency
                    else:
                        price_entries.append((entry_date, entry_price))

                return price_entries, errors, is_multi_symbol, entry_currencies
            except json.JSONDecodeError:
                # Not valid JSON, fall through to simple format
                pass

        # Simple format: "date:price,date:price,..."
        for entry in prices.split(","):
            entry = entry.strip()
            if not entry:
                continue

            if ":" not in entry:
                errors.append(f"Invalid format '{entry}' - expected 'YYYY-MM-DD:price'")
                continue

            parts = entry.split(":", 1)
            date_str = parts[0].strip()
            price_str = parts[1].strip()

            try:
                entry_date = dt.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                errors.append(f"Invalid date '{date_str}' - use YYYY-MM-DD format")
                continue

            try:
                entry_price = Decimal(price_str)
                if entry_price <= 0:
                    errors.append(f"Invalid price '{price_str}' - must be positive")
                    continue
            except Exception:
                errors.append(f"Invalid price '{price_str}' - must be a number")
                continue

            price_entries.append((entry_date, entry_price))

        return price_entries, errors, False, {}

    def _run(self, symbol: Optional[str] = None, prices: str = "", currency: Optional[str] = None) -> str:
        """Bulk set market prices for one or more instruments."""
        try:
            from ..portfolio.models import Currency as CurrencyEnum

            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            # Parse currency if provided
            price_currency = None
            if currency:
                try:
                    price_currency = CurrencyEnum(currency.upper())
                except ValueError:
                    valid_currencies = [c.value for c in CurrencyEnum]
                    return f"❌ Invalid currency '{currency}'. Valid options: {', '.join(valid_currencies)}"

            # Parse price data (supports simple, single-symbol JSON, and multi-symbol JSON formats)
            price_entries, errors, is_multi_symbol, entry_currencies = self._parse_prices(prices)

            if not price_entries:
                error_msg = "No valid price entries found."
                if errors:
                    error_msg += f"\nErrors:\n" + "\n".join(f"• {e}" for e in errors)
                return f"❌ {error_msg}"

            # Handle multi-symbol format (may have per-entry currencies)
            if is_multi_symbol:
                return self._run_multi_symbol(price_entries, errors, entry_currencies)

            # Handle single-symbol format (requires symbol parameter)
            if not symbol:
                return "❌ Symbol parameter is required for single-symbol price format. Use multi-symbol JSON format if you want to set prices for multiple symbols."

            return self._run_single_symbol(symbol, price_entries, errors, price_currency)

        except Exception as e:
            return f"❌ Error in bulk price update: {str(e)}"

    def _run_single_symbol(self, symbol: str, price_entries: list, errors: list, currency=None) -> str:
        """Handle single-symbol bulk price update.

        Args:
            symbol: The instrument symbol
            price_entries: List of (date, price) tuples
            errors: List of parsing errors
            currency: Optional Currency enum for all prices
        """
        symbol = symbol.upper().strip()

        # Check if symbol exists in current portfolio
        position = self.portfolio_manager.current_portfolio.positions.get(symbol)
        is_sold_instrument = position is None

        # For sold instruments, we can still update historical market data entries
        if is_sold_instrument:
            logging.info(f"Bulk updating prices for sold instrument {symbol}")

        # Sort by date
        price_entries.sort(key=lambda x: x[0])

        # Apply each price with currency
        success_count = 0
        failed_dates = []

        for entry_date, entry_price in price_entries:
            success = self.portfolio_manager.set_position_price(
                symbol=symbol,
                price=entry_price,
                target_date=entry_date,
                update_current=(entry_date == date.today() and not is_sold_instrument),
                currency=currency,
            )
            if success:
                success_count += 1
            else:
                failed_dates.append(str(entry_date))

        # Build result message
        currency_note = f" (currency: {currency.value})" if currency else ""
        if is_sold_instrument:
            result_lines = [
                f"✅ Bulk price update for sold instrument {symbol}{currency_note}:",
                f"• Successfully updated: {success_count} historical market data entries",
                f"• Date range: {price_entries[0][0]} to {price_entries[-1][0]}",
                f"• Note: This is a sold instrument - only historical market data entries were updated",
            ]
        else:
            native_currency = position.instrument.currency.value
            result_lines = [
                f"✅ Bulk price update for {symbol} ({position.instrument.name}):",
                f"• Successfully set: {success_count} prices{currency_note}",
                f"• Native currency: {native_currency}",
                f"• Date range: {price_entries[0][0]} to {price_entries[-1][0]}",
            ]
            if currency and currency.value != native_currency:
                result_lines.append(f"• Note: Prices converted from {currency.value} to {native_currency}")

        if failed_dates:
            result_lines.append(f"• Failed dates: {', '.join(failed_dates)}")

        if errors:
            result_lines.append(f"\n⚠️ Parsing warnings:")
            result_lines.extend(f"• {e}" for e in errors[:5])
            if len(errors) > 5:
                result_lines.append(f"  ... and {len(errors) - 5} more")

        return "\n".join(result_lines)

    def _is_isin(self, symbol: str) -> bool:
        """Check if a symbol looks like an ISIN.

        ISIN format: 2-letter country code + 9 alphanumeric + 1 check digit = 12 chars
        Common prefixes: US, XS, DE, CH, IE, LU, FR, GB, NL, etc.
        """
        import re
        if len(symbol) != 12:
            return False
        # Must start with 2 letters and be alphanumeric
        return bool(re.match(r'^[A-Z]{2}[A-Z0-9]{10}$', symbol))

    def _resolve_isin_to_symbol(self, isin: str) -> tuple[str, bool]:
        """Try to resolve an ISIN to a portfolio symbol.

        Searches current portfolio positions and transaction history for
        an instrument with the matching ISIN.

        Args:
            isin: The ISIN to resolve

        Returns:
            Tuple of (symbol, was_resolved) where was_resolved indicates if
            a match was found. If not found, returns original ISIN.
        """
        if not self.portfolio_manager.current_portfolio:
            return isin, False

        isin_upper = isin.upper().strip()

        # Search current positions
        for symbol, position in self.portfolio_manager.current_portfolio.positions.items():
            if position.instrument.isin and position.instrument.isin.upper() == isin_upper:
                return symbol, True

        # Search transaction history for historical instruments
        for txn in self.portfolio_manager.current_portfolio.transactions:
            if txn.instrument.isin and txn.instrument.isin.upper() == isin_upper:
                return txn.instrument.symbol, True

        return isin, False

    def _resolve_symbols_in_entries(self, price_entries: list) -> tuple[list, dict, list]:
        """Resolve ISINs to portfolio symbols in price entries.

        Args:
            price_entries: List of (symbol, date, price) tuples

        Returns:
            Tuple of (resolved_entries, isin_mappings, unresolved_isins) where:
            - isin_mappings shows which ISINs were resolved to portfolio symbols
            - unresolved_isins lists ISINs that couldn't be mapped to portfolio symbols
        """
        resolved_entries = []
        isin_mappings = {}  # ISIN -> portfolio symbol
        unresolved_isins = []  # ISINs that couldn't be resolved

        for symbol, entry_date, entry_price in price_entries:
            if self._is_isin(symbol):
                resolved_symbol, was_resolved = self._resolve_isin_to_symbol(symbol)
                if was_resolved:
                    isin_mappings[symbol] = resolved_symbol
                    resolved_entries.append((resolved_symbol, entry_date, entry_price))
                else:
                    # Store under ISIN but track as unresolved
                    unresolved_isins.append(symbol)
                    resolved_entries.append((symbol, entry_date, entry_price))
            else:
                resolved_entries.append((symbol, entry_date, entry_price))

        return resolved_entries, isin_mappings, unresolved_isins

    def _run_multi_symbol(self, price_entries: list, errors: list, entry_currencies: dict = None) -> str:
        """Handle multi-symbol bulk price update.

        Args:
            price_entries: List of (symbol, date, price) tuples
            errors: List of parsing errors
            entry_currencies: Dict mapping (symbol, date) -> Currency for per-entry currencies
        """
        entry_currencies = entry_currencies or {}

        # Resolve ISINs to portfolio symbols
        resolved_entries, isin_mappings, unresolved_isins = self._resolve_symbols_in_entries(price_entries)

        # Update entry_currencies keys with resolved symbols
        resolved_currencies = {}
        for (symbol, entry_date), currency in entry_currencies.items():
            resolved_symbol = isin_mappings.get(symbol, symbol)
            resolved_currencies[(resolved_symbol, entry_date)] = currency

        # Group entries by symbol for reporting
        from collections import defaultdict
        entries_by_symbol: dict = defaultdict(list)
        for symbol, entry_date, entry_price in resolved_entries:
            entries_by_symbol[symbol].append((entry_date, entry_price))

        # For multi-symbol with per-entry currencies, we need to call set_position_price individually
        if resolved_currencies:
            success_count = 0
            for symbol, entry_date, entry_price in resolved_entries:
                currency = resolved_currencies.get((symbol, entry_date))
                success = self.portfolio_manager.set_position_price(
                    symbol=symbol,
                    price=entry_price,
                    target_date=entry_date,
                    update_current=(entry_date == date.today()),
                    currency=currency,
                )
                if success:
                    success_count += 1
        else:
            # Use batch update for efficiency when no per-entry currencies
            success_count = self.portfolio_manager.set_positions_prices_batch(resolved_entries)

        # Build result message
        total_entries = len(resolved_entries)
        symbols_updated = list(entries_by_symbol.keys())

        # Find date range across all entries
        all_dates = [entry_date for _, entry_date, _ in resolved_entries]
        min_date = min(all_dates)
        max_date = max(all_dates)

        result_lines = [
            f"✅ Multi-symbol bulk price update completed:",
            f"• Symbols updated: {', '.join(symbols_updated)}",
            f"• Total prices set: {success_count}/{total_entries}",
            f"• Date range: {min_date} to {max_date}",
        ]

        # Show ISIN resolutions if any
        if isin_mappings:
            result_lines.append(f"\n🔗 ISIN resolutions ({len(isin_mappings)}):")
            for isin, portfolio_symbol in sorted(isin_mappings.items()):
                result_lines.append(f"• {isin} → {portfolio_symbol}")

        # Warn about unresolved ISINs (stored under ISIN, may not be found in lookups)
        if unresolved_isins:
            unique_unresolved = sorted(set(unresolved_isins))
            result_lines.append(f"\n⚠️ Unresolved ISINs ({len(unique_unresolved)}) - stored under ISIN key:")
            for isin in unique_unresolved[:10]:
                result_lines.append(f"• {isin}")
            if len(unique_unresolved) > 10:
                result_lines.append(f"  ... and {len(unique_unresolved) - 10} more")
            result_lines.append("  Note: These ISINs don't match any portfolio position. Prices stored but may not be used.")

        # Add per-symbol breakdown if multiple symbols
        if len(symbols_updated) > 1:
            result_lines.append(f"\nPer-symbol breakdown:")
            for symbol in sorted(symbols_updated):
                count = len(entries_by_symbol[symbol])
                position = self.portfolio_manager.current_portfolio.positions.get(symbol)
                if position:
                    result_lines.append(f"• {symbol} ({position.instrument.name}): {count} prices")
                else:
                    result_lines.append(f"• {symbol} (historical): {count} prices")

        if errors:
            result_lines.append(f"\n⚠️ Parsing warnings:")
            result_lines.extend(f"• {e}" for e in errors[:5])
            if len(errors) > 5:
                result_lines.append(f"  ... and {len(errors) - 5} more")

        return "\n".join(result_lines)


def create_portfolio_tools(
    portfolio_manager: PortfolioManager,
    data_manager: DataProviderManager,
    metrics_calculator: FinancialMetricsCalculator,
) -> List[BaseTool]:
    """Create all portfolio management tools."""
    return [
        AddTransactionTool(portfolio_manager),
        BulkAddTransactionsTool(portfolio_manager),
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
        BulkAddTransactionsTool(portfolio_manager),
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


class GetHistoricalInstrumentsTool(BaseTool):
    """Tool for listing instruments that were held during a date range but are no longer in current positions."""

    name: str = "get_historical_instruments"
    description: str = """List instruments that were held in a date range but are no longer in current positions.

    Use cases:
    - Find sold instruments that need market data updates
    - See complete instrument history for a period
    - Identify gaps in market data coverage
    - Verify which historical instruments have transactions in a date range

    Parameters:
    - start_date: Start date in YYYY-MM-DD format (required)
    - end_date: End date in YYYY-MM-DD format (required)

    Returns:
    - List of symbols with their transaction history summary
    - Flags indicating if market data exists for the date range
    - Comparison with current positions
    """
    args_schema: type[BaseModel] = GetHistoricalInstrumentsInput
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _run(
        self,
        start_date: str,
        end_date: str,
    ) -> str:
        """List historical instruments in a date range."""
        try:
            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            # Parse dates
            try:
                from datetime import datetime as dt
                start = dt.strptime(start_date, "%Y-%m-%d").date()
                end = dt.strptime(end_date, "%Y-%m-%d").date()
            except ValueError:
                return "❌ Invalid date format. Use YYYY-MM-DD."

            if start > end:
                return "❌ Start date must be before or equal to end date."

            # Get instruments from transactions in date range
            historical_instruments = self.portfolio_manager.get_instruments_in_date_range(start, end)

            if not historical_instruments:
                return f"No instruments with transactions found between {start_date} and {end_date}."

            # Get current positions for comparison
            current_positions = set(self.portfolio_manager.current_portfolio.positions.keys())

            # Categorize instruments
            still_held = []
            sold = []

            for symbol, instrument in historical_instruments.items():
                # Count transactions for this symbol in date range
                txn_count = 0
                buy_count = 0
                sell_count = 0
                for txn in self.portfolio_manager.current_portfolio.transactions:
                    txn_date = txn.timestamp.date()
                    if start <= txn_date <= end and txn.instrument.symbol == symbol:
                        txn_count += 1
                        if txn.transaction_type.value == "buy":
                            buy_count += 1
                        elif txn.transaction_type.value == "sell":
                            sell_count += 1

                # Check if market data exists
                has_market_data = False
                market_data_store = self.portfolio_manager.market_data_store
                if market_data_store:
                    price = market_data_store.get_price_with_fallback(symbol, end)
                    has_market_data = price is not None

                info = {
                    "symbol": symbol,
                    "name": instrument.name,
                    "currency": instrument.currency.value,
                    "txn_count": txn_count,
                    "buy_count": buy_count,
                    "sell_count": sell_count,
                    "has_market_data": has_market_data,
                    "data_provider_symbol": instrument.data_provider_symbol,
                }

                if symbol in current_positions:
                    still_held.append(info)
                else:
                    sold.append(info)

            # Format output
            lines = [
                f"📊 Instruments with activity from {start_date} to {end_date}:",
                "",
            ]

            if sold:
                lines.append(f"🔴 SOLD INSTRUMENTS (not in current positions): {len(sold)}")
                lines.append("")
                for info in sold:
                    market_status = "✅ has market data" if info["has_market_data"] else "⚠️ NO market data"
                    provider_info = f" (provider: {info['data_provider_symbol']})" if info["data_provider_symbol"] else ""
                    lines.append(f"  • {info['symbol']}: {info['name']}")
                    lines.append(f"    Currency: {info['currency']}{provider_info}")
                    lines.append(f"    Transactions: {info['txn_count']} ({info['buy_count']} buys, {info['sell_count']} sells)")
                    lines.append(f"    Market data: {market_status}")
                    lines.append("")

            if still_held:
                lines.append(f"🟢 STILL HELD (in current positions): {len(still_held)}")
                lines.append("")
                for info in still_held:
                    market_status = "✅" if info["has_market_data"] else "⚠️"
                    lines.append(f"  • {info['symbol']}: {info['name']} - {info['txn_count']} transactions {market_status}")

            # Add summary
            lines.append("")
            lines.append("📋 Summary:")
            lines.append(f"  • Total instruments with activity: {len(historical_instruments)}")
            lines.append(f"  • Still held: {len(still_held)}")
            lines.append(f"  • Sold/closed: {len(sold)}")

            needs_data = [s["symbol"] for s in sold if not s["has_market_data"]]
            if needs_data:
                lines.append("")
                lines.append("💡 Tip: Use update_historical_market_data to fetch prices for all instruments")
                lines.append(f"   (including sold ones): {', '.join(needs_data)}")

            return "\n".join(lines)

        except Exception as e:
            return f"❌ Error getting historical instruments: {str(e)}"


class UpdateHistoricalMarketDataTool(BaseTool):
    """Tool for updating market data for all instruments (including sold ones) in a date range."""

    name: str = "update_historical_market_data"
    description: str = """Update historical market data for one or all instruments in a date range.

    Unlike the regular market data update, this tool includes sold instruments.
    Respects price_currency settings — prices are converted to the instrument's
    portfolio currency before storing (e.g. GBP → JPY for CNKY).

    Parameters:
    - start_date: Start date in YYYY-MM-DD format (required)
    - end_date: End date in YYYY-MM-DD format (required)
    - symbol: Specific instrument to update (recommended). If omitted, ALL instruments are updated.
    - include_historical: Include sold instruments (default True, ignored when symbol is provided)

    Prefer providing a symbol to avoid long-running bulk updates.
    """
    args_schema: type[BaseModel] = UpdateHistoricalMarketDataInput
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _run(
        self,
        start_date: str,
        end_date: str,
        symbol: Optional[str] = None,
        include_historical: bool = True,
    ) -> str:
        """Update market data for one or all instruments in date range."""
        try:
            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            # Parse dates
            try:
                from datetime import datetime as dt
                start = dt.strptime(start_date, "%Y-%m-%d").date()
                end = dt.strptime(end_date, "%Y-%m-%d").date()
            except ValueError:
                return "❌ Invalid date format. Use YYYY-MM-DD."

            if start > end:
                return "❌ Start date must be before or equal to end date."

            # Get current positions
            current_positions = set(self.portfolio_manager.current_portfolio.positions.keys())

            # Get historical instruments if requested (skip when targeting a single symbol)
            historical_instruments = {}
            if include_historical and not symbol:
                historical_instruments = self.portfolio_manager.get_instruments_in_date_range(start, end)

            # Count what we're updating
            historical_only = set(historical_instruments.keys()) - current_positions

            # Perform the update
            results = self.portfolio_manager.update_market_data(
                start_date=start,
                end_date=end,
                include_historical=include_historical,
                symbol=symbol,
            )

            if not results:
                return f"No instruments to update for the period {start_date} to {end_date}."

            # Categorize results
            success_current = []
            success_historical = []
            failed = []

            for symbol, success in results.items():
                if success:
                    if symbol in historical_only:
                        success_historical.append(symbol)
                    else:
                        success_current.append(symbol)
                else:
                    failed.append(symbol)

            # Format output
            lines = [
                f"📊 Market Data Update: {start_date} to {end_date}",
                "",
            ]

            if success_current:
                lines.append(f"✅ Current positions updated: {len(success_current)}")
                lines.append(f"   {', '.join(sorted(success_current))}")
                lines.append("")

            if success_historical:
                lines.append(f"✅ Historical/sold instruments updated: {len(success_historical)}")
                lines.append(f"   {', '.join(sorted(success_historical))}")
                lines.append("")

            if failed:
                lines.append(f"❌ Failed to update: {len(failed)}")
                lines.append(f"   {', '.join(sorted(failed))}")
                lines.append("")
                lines.append("💡 Tip: Use fetch_and_update_prices with a provider_symbol for failed instruments,")
                lines.append("   or use bulk_set_market_price to manually enter prices.")

            # Summary
            lines.append("")
            lines.append("📋 Summary:")
            lines.append(f"  • Total instruments processed: {len(results)}")
            lines.append(f"  • Successfully updated: {len(success_current) + len(success_historical)}")
            if include_historical:
                lines.append(f"  • Including historical/sold: {len(historical_only)} instruments")
            lines.append(f"  • Failed: {len(failed)}")

            return "\n".join(lines)

        except Exception as e:
            return f"❌ Error updating historical market data: {str(e)}"


class GetYTDPerformanceTool(BaseTool):
    """Tool for getting Year-to-Date performance of all portfolio positions."""

    name: str = "get_ytd_performance"
    description: str = (
        "Get Year-to-Date (YTD) performance for all portfolio positions and the portfolio overall. "
        "Compares current prices to Dec 31 of the prior year."
    )
    args_schema: type[BaseModel] = GetYTDPerformanceInput
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _run(self, as_of_date: Optional[str] = None) -> str:
        """Get YTD performance for all positions."""
        try:
            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            # Parse date
            target_date = None
            if as_of_date:
                try:
                    target_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()
                except ValueError:
                    return f"❌ Invalid date format: {as_of_date}. Use YYYY-MM-DD."

            data = self.portfolio_manager.get_ytd_performance(target_date)

            if "error" in data:
                return f"❌ {data['error']}"

            portfolio_name = data["portfolio_name"]
            as_of = data["as_of_date"]
            year_end = data["year_end_date"]
            base_ccy = data["base_currency"]
            positions = data["positions"]
            portfolio = data.get("portfolio", {})

            lines = [
                f"📊 YTD Performance — Portfolio: {portfolio_name}",
                f"📅 As of: {as_of} | Reference: {year_end}",
            ]

            # Portfolio-level YTD
            if portfolio:
                sign = "+" if portfolio["ytd_pct"] >= 0 else ""
                val_sign = "+" if portfolio["ytd_value_change"] >= 0 else ""
                lines.append(
                    f"💰 Portfolio YTD: {sign}{portfolio['ytd_pct']:.2f}% "
                    f"({val_sign}{portfolio['ytd_value_change']:,.2f} {base_ccy})"
                )
            else:
                lines.append("💰 Portfolio YTD: N/A (insufficient data)")

            lines.append("")

            if not positions:
                lines.append("No positions found.")
                return "\n".join(lines)

            # Table header
            lines.append(
                f"{'Instrument':<20}| {'Ccy':^5}| {'Price (Dec 31)':>14} | {'Price (Now)':>11} | {'YTD %':>8} | {'YTD Chg':>12}"
            )
            lines.append("-" * 20 + "|" + "-" * 5 + "|" + "-" * 16 + "|" + "-" * 13 + "|" + "-" * 10 + "|" + "-" * 13)

            for pos in sorted(positions, key=lambda p: abs(p.get("ytd_pct") or 0), reverse=True):
                symbol = pos["symbol"]
                ccy = pos["currency"]
                ye_price = pos["year_end_price"]
                cur_price = pos["current_price"]
                ytd_pct = pos["ytd_pct"]
                ytd_chg = pos["ytd_value_change"]
                since = pos.get("since_inception", False)

                ye_str = f"{ye_price:>14,.2f}" if ye_price is not None else f"{'N/A':>14}"
                cur_str = f"{cur_price:>11,.2f}" if cur_price is not None else f"{'N/A':>11}"

                if ytd_pct is not None:
                    sign = "+" if ytd_pct >= 0 else ""
                    pct_str = f"{sign}{ytd_pct:.1f}%"
                    if since:
                        pct_str += "*"
                    pct_str = f"{pct_str:>8}"
                else:
                    pct_str = f"{'N/A':>8}"

                if ytd_chg is not None:
                    chg_sign = "+" if ytd_chg >= 0 else ""
                    chg_str = f"{chg_sign}{ytd_chg:>11,.2f}"
                else:
                    chg_str = f"{'N/A':>12}"

                # Truncate symbol to 19 chars
                sym_display = symbol[:19]
                lines.append(
                    f"{sym_display:<20}| {ccy:^5}|{ye_str} | {cur_str} |{pct_str} |{chg_str}"
                )

            # Footnote for since-inception entries
            if any(p.get("since_inception") for p in positions):
                lines.append("")
                lines.append("* Position bought after Jan 1 — return is since inception (vs avg cost), not YTD.")

            return "\n".join(lines)

        except Exception as e:
            return f"❌ Error getting YTD performance: {str(e)}"


class InterpolatePricesTool(BaseTool):
    """Tool for interpolating missing market prices using linear interpolation."""

    name: str = "interpolate_prices"
    description: str = """Fill in missing market prices using linear interpolation between known values.

    Use cases:
    - Filling gaps in market data for bonds/structured products without live feeds
    - Smoothing out missing data points between known prices
    - Fixing portfolio valuation gaps caused by missing historical data

    How it works:
    - Finds the nearest available price before and after the date range
    - Calculates daily price change using linear interpolation
    - Fills in all missing dates between the boundary prices
    - Skips dates that already have prices (won't overwrite existing data)

    Parameters:
    - symbols: Comma-separated list of symbols (e.g., "GLENCORE_2028,BAYER_2026")
              If not provided, interpolates all current portfolio positions
    - start_date: Start of date range to fill (YYYY-MM-DD)
    - end_date: End of date range to fill (YYYY-MM-DD)

    Example:
        interpolate_prices(symbols="GLENCORE_2028,BAYER_2026", start_date="2026-02-01", end_date="2026-02-20")
    """
    args_schema: type[BaseModel] = InterpolatePricesInput
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _run(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[str] = None,
    ) -> str:
        """Execute price interpolation."""
        from datetime import datetime as dt

        try:
            # Parse dates
            try:
                start = dt.strptime(start_date, "%Y-%m-%d").date()
                end = dt.strptime(end_date, "%Y-%m-%d").date()
            except ValueError as e:
                return f"❌ Invalid date format: {e}. Use YYYY-MM-DD."

            if start >= end:
                return "❌ Start date must be before end date."

            # Get symbols to process
            if symbols:
                symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
            else:
                # Use all current portfolio positions
                if not self.portfolio_manager.current_portfolio:
                    return "❌ No portfolio loaded. Please load a portfolio first."
                positions = self.portfolio_manager.get_positions()
                symbol_list = [p.get("symbol", "").upper() for p in positions if p.get("symbol")]

            if not symbol_list:
                return "❌ No symbols to interpolate."

            # Get market data store
            market_data_store = self.portfolio_manager._market_data_store

            # Interpolate each symbol
            results = []
            total_interpolated = 0
            errors = []

            for symbol in symbol_list:
                try:
                    count = market_data_store.interpolate_prices(symbol, start, end)
                    if count > 0:
                        results.append(f"  • {symbol}: {count} prices interpolated")
                        total_interpolated += count
                    else:
                        results.append(f"  • {symbol}: no gaps found or missing boundary prices")
                except Exception as e:
                    errors.append(f"  • {symbol}: {str(e)}")

            # Build response
            lines = [
                "📊 **Price Interpolation Results**",
                f"📅 Date range: {start_date} to {end_date}",
                f"🔢 Total interpolated: {total_interpolated} prices",
                "",
            ]

            if results:
                lines.append("**Results:**")
                lines.extend(results)

            if errors:
                lines.append("")
                lines.append("**Errors:**")
                lines.extend(errors)

            return "\n".join(lines)

        except Exception as e:
            return f"❌ Error interpolating prices: {str(e)}"
