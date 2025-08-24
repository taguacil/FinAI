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


class SearchInstrumentInput(PortfolioToolInput):
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
    description: str = """Add a transaction to the portfolio. Supports:
    - buy/sell stocks, bonds, ETFs: specify symbol, quantity, price
    - deposit/withdraw cash: use 'CASH' as symbol
    - dividends: specify symbol and amount
    - fees: use 'fees' as transaction type with CASH symbol
    Examples:
    - "I bought 50 shares of AAPL at $150"
    - "I sold 25 TSLA shares at $200 yesterday"
    - "I bought 100 TLT bonds at $90.50"
    - "I purchased 50 BIL treasury bills at $98.75"
    - "I deposited $5000 cash"
    - "I paid $5 in trading fees"
    """
    args_schema: type[BaseModel] = AddTransactionInput
    portfolio_manager: PortfolioManager | None = None

    def __init__(self, portfolio_manager: PortfolioManager):
        super().__init__()
        self.portfolio_manager = portfolio_manager

    def _run(
        self,
        symbol: Optional[str] = None,
        isin: Optional[str] = None,
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

            # Determine timestamp: prefer explicit date, else days_ago
            if date:
                try:
                    timestamp = datetime.strptime(date, "%Y-%m-%d")
                except ValueError:
                    return "Invalid date format. Use YYYY-MM-DD."
            else:
                timestamp = datetime.now() - timedelta(days=days_ago)

            # Choose identifier: symbol or ISIN; for cash movements default to CASH
            # Handle different transaction scenarios
            if txn_type in (TransactionType.DEPOSIT, TransactionType.WITHDRAWAL):
                identifier = "CASH"
            elif isin and symbol:
                # Both ISIN and symbol provided - use symbol as identifier
                identifier = symbol
            elif isin:
                # Only ISIN provided - try to find symbol or use ISIN as fallback
                identifier = isin
            elif symbol:
                # Only symbol provided - use symbol as identifier
                identifier = symbol
            else:
                return "Please provide either a symbol or an ISIN."

            # Normalize cash amount semantics for deposits/withdrawals
            if identifier == "CASH" and txn_type in (
                TransactionType.DEPOSIT,
                TransactionType.WITHDRAWAL,
            ):
                # If only quantity provided, treat as amount; use quantity=1
                if quantity > 0 and price == 0:
                    price = quantity
                    quantity = 1.0
                # If neither positive, invalid
                if price <= 0:
                    return "Please provide a positive amount for deposit/withdrawal. Use 'price' as amount or 'quantity' alone."
                # Force quantity=1 for cash movements
                quantity = 1.0

            # Prepare currency
            from src.portfolio.models import Currency as Cur

            cur_obj = None
            if currency:
                try:
                    cur_obj = Cur(currency.upper())
                except ValueError:
                    return f"Invalid currency: {currency}. Use one of {[c.value for c in Cur]}"

            # Handle bond price interpretation (percentages)
            final_price = price
            if isin and isin.upper().startswith('XS'):  # XS ISINs are typically bonds
                # If price looks like a percentage (between 0 and 200), treat as percentage of face value
                if 0 < price <= 200:
                    # Keep the percentage as-is for bonds (98.85% -> 98.85)
                    # This is the standard way bonds are quoted
                    final_price = price
                    notes = f"{notes or ''} (Price: {price}% of face value)".strip()

            # Add transaction
            # Determine the transaction symbol
            if isin and not symbol:
                # Only ISIN provided - try to find symbol or use placeholder
                # The portfolio manager will handle symbol discovery
                transaction_symbol = ""  # Let portfolio manager find/create symbol
            else:
                # Symbol provided or no ISIN - use the identifier
                transaction_symbol = identifier.upper()

            success = self.portfolio_manager.add_transaction(
                symbol=transaction_symbol,
                transaction_type=txn_type,
                quantity=Decimal(str(quantity)),
                price=Decimal(str(final_price)),
                timestamp=timestamp,
                notes=notes,
                isin=(isin.upper() if isin else None),
                currency=cur_obj,
            )

            if success:
                # Show the actual symbol if available, otherwise show ISIN
                if symbol:
                    label = symbol.upper()
                elif isin:
                    label = f"ISIN_{isin[:8]}"
                else:
                    label = "Unknown"
                return f"✅ Added {transaction_type} transaction: {quantity} {label} @ ${price}"
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
    portfolio_manager: PortfolioManager | None = None
    metrics_calculator: FinancialMetricsCalculator | None = None

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
                return (
                    "❌ No portfolio loaded. Please create or load a portfolio first."
                )

            portfolio = self.portfolio_manager.current_portfolio

            # Note: Prices are not automatically updated - use existing data only
            # Users should manually update prices via the UI if needed

            # Get positions summary
            positions = self.portfolio_manager.get_position_summary()
            total_value = self.portfolio_manager.get_portfolio_value()

            summary = [
                f"📊 **Portfolio Summary: {portfolio.name}**",
                f"💰 **Total Value**: ${total_value:,.2f} {portfolio.base_currency.value}",
                f"📅 **Created**: {portfolio.created_at.strftime('%Y-%m-%d')}",
                "",
            ]

            # Cash balances
            if portfolio.cash_balances:
                summary.append("💵 **Cash Balances:**")
                for currency, amount in portfolio.cash_balances.items():
                    code = getattr(currency, "value", str(currency))
                    summary.append(f"  • {code}: {amount:,.2f}")
                summary.append("")

            # Positions
            if positions:
                summary.append("📈 **Current Positions:**")
                for pos in positions:
                    pnl_str = ""
                    if pos["unrealized_pnl"] is not None:
                        pnl = float(pos["unrealized_pnl"])
                        pnl_pct = float(pos["unrealized_pnl_percent"] or 0)
                        pnl_emoji = "📈" if pnl >= 0 else "📉"
                        pnl_str = f" | {pnl_emoji} {pnl:+.2f} ({pnl_pct:+.1f}%)"

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

                    summary.append(
                        f"  • **{pos['symbol']}** ({pos['name']}) [{instrument_type.upper()}]: "
                        f"{pos['quantity']} {unit_label} @ ${pos['current_price'] or 'N/A'}"
                        f"{pnl_str}"
                    )
                summary.append("")

            # Basic metrics if requested
            if include_metrics:
                try:
                    metrics = self.portfolio_manager.get_performance_metrics()
                    if metrics and "error" not in metrics:
                        summary.append("📊 **Performance Metrics:**")
                        summary.append(
                            f"  • Total Return: {metrics.get('total_return_percent', 0):.2f}%"
                        )
                        summary.append(
                            f"  • Volatility: {metrics.get('annualized_volatility_percent', 0):.2f}%"
                        )
                        summary.append("")
                except Exception:
                    # Metrics calculation failed, continue without them
                    pass

            return "\n".join(summary)

        except Exception as e:
            return f"❌ Error getting portfolio summary: {str(e)}"


class GetTransactionsTool(BaseTool):
    """Tool for exporting transactions and history with metrics-ready fields."""

    name: str = "get_transactions"
    description: str = (
        "Return all transactions with key fields for analysis (ids, timestamps, symbols, quantities, prices, currency)."
    )
    portfolio_manager: PortfolioManager | None = None

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
    portfolio_manager: PortfolioManager | None = None

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
                f"• Start value: ${start_value:,.2f}\n"
                f"• End value: ${end_value:,.2f}\n"
                f"• Total return: {total_return:.2f}%\n"
                f"• Baseline end: ${base_end:,.2f} | Baseline return: {base_return:.2f}%\n"
                f"• Δ End value vs baseline: {delta_value:+,.2f} | Δ Return: {delta_return:+.2f}%"
            )
        except Exception as e:
            return f"❌ Error running simulation: {str(e)}"


class SearchInstrumentTool(BaseTool):
    """Tool for searching financial instruments."""

    name: str = "search_instrument"
    description: str = (
        "Search for stocks, bonds, ETFs, or other financial instruments by symbol or company name."
    )
    args_schema: type[BaseModel] = SearchInstrumentInput
    data_manager: DataProviderManager | None = None

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


class GetCurrentPriceTool(BaseTool):
    """Tool for getting current price of an instrument."""

    name: str = "get_current_price"
    description: str = (
        "Get the current market price of a stock, bond, ETF, or other financial instrument."
    )
    args_schema: type[BaseModel] = GetPriceInput
    data_manager: DataProviderManager | None = None

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

            return f"💰 **{symbol.upper()}** ({name}): ${price:.2f}"

        except Exception as e:
            return f"❌ Error getting price for {symbol}: {str(e)}"


class GetPortfolioMetricsTool(BaseTool):
    """Tool for getting detailed portfolio metrics."""

    name: str = "get_portfolio_metrics"
    description: str = (
        "Calculate detailed portfolio performance metrics including volatility, Sharpe ratio, drawdown, alpha, and beta."
    )
    args_schema: type[BaseModel] = GetMetricsInput
    portfolio_manager: PortfolioManager | None = None
    metrics_calculator: FinancialMetricsCalculator | None = None

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
    portfolio_manager: PortfolioManager | None = None

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
                        f"@ ${txn['price']} (Total: ${txn['total_value']})"
                    )
                elif txn_type == "DIVIDEND":
                    result.append(
                        f"• {date_str}: DIVIDEND {symbol} ${txn['total_value']}"
                    )
                else:
                    result.append(f"• {date_str}: {txn_type} ${txn['total_value']}")

            return "\n".join(result)

        except Exception as e:
            return f"❌ Error getting transaction history: {str(e)}"


def create_portfolio_tools(
    portfolio_manager: PortfolioManager,
    data_manager: DataProviderManager,
    metrics_calculator: FinancialMetricsCalculator,
) -> List[BaseTool]:
    """Create all portfolio management tools."""
    return [
        AddTransactionTool(portfolio_manager),
        GetPortfolioSummaryTool(portfolio_manager, metrics_calculator),
        GetTransactionsTool(portfolio_manager),
        SimulateWhatIfTool(portfolio_manager),
        IngestPdfTool(),
        CalculatorTool(),
        SearchInstrumentTool(data_manager),
        GetCurrentPriceTool(data_manager),
        GetPortfolioMetricsTool(portfolio_manager, metrics_calculator),
        GetTransactionHistoryTool(portfolio_manager),
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
