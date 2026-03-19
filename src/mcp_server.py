"""
MCP (Model Context Protocol) server exposing all portfolio agent tools.

Supports two transports:
  - SSE (HTTP, default): for Cursor and other HTTP-based MCP clients
  - stdio: for Claude Desktop and other process-based MCP clients

Usage:
    # SSE mode (default) - starts HTTP server
    uv run python -m src.mcp_server

    # stdio mode - for Claude Desktop
    uv run python -m src.mcp_server --stdio

Environment variables:
    PORTFOLIO_NAME - Name of portfolio to load on startup (default: first available)
    DATA_DIR - Data directory path (default: "data")
    FINAI_API_KEY - API key for Bearer token auth (required for SSE mode)
    HOST - Server host (default: "localhost", SSE mode only)
    PORT - Server port (default: 8000, SSE mode only)
"""

import logging
import os
import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_providers.manager import DataProviderManager
from src.portfolio.manager import PortfolioManager
from src.portfolio.storage import FileBasedStorage
from src.services.market_data_service import MarketDataService
from src.utils.metrics import FinancialMetricsCalculator

# Tool classes
from src.agents.portfolio_tools import (
    AddTransactionTool,
    AdvancedWhatIfTool,
    BulkAddTransactionsTool,
    BulkSetMarketPriceTool,
    CalculatorTool,
    CheckMarketDataAvailabilityTool,
    DeleteTransactionTool,
    FetchAndUpdatePricesTool,
    GetCurrentPriceTool,
    GetHistoricalInstrumentsTool,
    GetPortfolioMetricsTool,
    GetPortfolioSnapshotTool,
    GetPortfolioSummaryTool,
    GetYTDPerformanceTool,
    GetTransactionHistoryTool,
    GetTransactionsTool,
    HypotheticalPositionTool,
    IngestPdfTool,
    InterpolatePricesTool,
    ModifyTransactionTool,
    OptimizePortfolioTool,
    ResolveInstrumentTool,
    ScenarioOptimizationTool,
    SearchCompanyTool,
    SearchInstrumentTool,
    SetDataProviderSymbolTool,
    SetPriceCurrencyTool,
    SetMarketPriceTool,
    SimulateWhatIfTool,
    UpdateHistoricalMarketDataTool,
)
from src.agents.tools.market_data_tools import (
    FetchHistoricalFXRatesTool,
    GetBatchPricesTool,
    GetDataFreshnessTool,
    GetFXRateTool,
    GetMovingAverageSignalTool,
    GetPriceHistoryTool,
    RefreshDataTool,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize services ---
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_PROJECT_ROOT)  # Ensure relative paths (fx_cache, etc.) resolve correctly
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(_PROJECT_ROOT, "data"))
PORTFOLIO_NAME = os.environ.get("PORTFOLIO_NAME", "")

storage = FileBasedStorage(DATA_DIR)
data_manager = DataProviderManager()
market_data_service = MarketDataService(data_manager)
portfolio_manager = PortfolioManager(storage, market_data_service, data_dir=DATA_DIR)
metrics_calculator = FinancialMetricsCalculator(data_manager)

# Load portfolio
available = storage.list_portfolios()
if PORTFOLIO_NAME and PORTFOLIO_NAME in available:
    portfolio_manager.load_portfolio(PORTFOLIO_NAME)
    logger.info(f"Loaded portfolio: {PORTFOLIO_NAME}")
elif available:
    portfolio_manager.load_portfolio(available[0])
    logger.info(f"Loaded portfolio: {available[0]}")
else:
    logger.warning(
        "No portfolios found. Tools requiring a portfolio will fail until one is created."
    )

# --- Instantiate LangChain tools ---
_add_transaction = AddTransactionTool(portfolio_manager)
_bulk_add_transactions = BulkAddTransactionsTool(portfolio_manager)
_modify_transaction = ModifyTransactionTool(portfolio_manager)
_delete_transaction = DeleteTransactionTool(portfolio_manager)
_get_portfolio_summary = GetPortfolioSummaryTool(portfolio_manager, metrics_calculator)
_get_portfolio_snapshot = GetPortfolioSnapshotTool(portfolio_manager)
_get_transactions = GetTransactionsTool(portfolio_manager)
_simulate_what_if = SimulateWhatIfTool(portfolio_manager)
_advanced_what_if = AdvancedWhatIfTool(portfolio_manager)
_hypothetical_position = HypotheticalPositionTool(portfolio_manager)
_search_instrument = SearchInstrumentTool(data_manager)
_search_company = SearchCompanyTool(data_manager)
_resolve_instrument = ResolveInstrumentTool(data_manager)
_check_market_data = CheckMarketDataAvailabilityTool(data_manager)
_get_current_price = GetCurrentPriceTool(data_manager, portfolio_manager)
_get_portfolio_metrics = GetPortfolioMetricsTool(portfolio_manager, metrics_calculator)
_get_transaction_history = GetTransactionHistoryTool(portfolio_manager)
_set_market_price = SetMarketPriceTool(portfolio_manager)
_bulk_set_market_price = BulkSetMarketPriceTool(portfolio_manager)
_fetch_and_update_prices = FetchAndUpdatePricesTool(portfolio_manager, data_manager)
_set_data_provider_symbol = SetDataProviderSymbolTool(portfolio_manager)
_set_price_currency = SetPriceCurrencyTool(portfolio_manager)
_calculator = CalculatorTool()
_ingest_pdf = IngestPdfTool()
_optimize_portfolio = OptimizePortfolioTool(portfolio_manager, data_manager)
_scenario_optimization = ScenarioOptimizationTool(portfolio_manager, data_manager)
_get_price_history = GetPriceHistoryTool(portfolio_manager.market_data_store)
_get_fx_rate = GetFXRateTool(market_data_service)
_get_batch_prices = GetBatchPricesTool(market_data_service)
_fetch_historical_fx_rates = FetchHistoricalFXRatesTool(data_manager)
_get_data_freshness = GetDataFreshnessTool(market_data_service)
_refresh_data = RefreshDataTool(market_data_service, portfolio_manager)
_get_historical_instruments = GetHistoricalInstrumentsTool(portfolio_manager)
_update_historical_market_data = UpdateHistoricalMarketDataTool(portfolio_manager)
_get_ytd_performance = GetYTDPerformanceTool(portfolio_manager)
_interpolate_prices = InterpolatePricesTool(portfolio_manager)
_get_ma_signal = GetMovingAverageSignalTool(portfolio_manager.market_data_store)

# --- Create MCP server ---
mcp = FastMCP("FinAI Portfolio")


# =====================================================================
# MCP Resources - Provide context to clients
# =====================================================================


@mcp.resource("portfolio://context")
def portfolio_context() -> str:
    """Current portfolio context including positions, balances, and value."""
    if not portfolio_manager.current_portfolio:
        return "No portfolio loaded. Use list_portfolios and select_portfolio to load one."
    p = portfolio_manager.current_portfolio
    lines = [
        f"Portfolio: {p.name}",
        f"ID: {p.id}",
        f"Total Value: {portfolio_manager.get_portfolio_value():.2f} {p.base_currency.value}",
        f"Positions: {len(p.positions)}",
        "",
        "Cash Balances:",
    ]
    for currency, balance in p.cash_balances.items():
        lines.append(f"  {currency.value}: {balance:.2f}")
    lines.append("")
    lines.append("Positions:")
    for symbol, pos in p.positions.items():
        lines.append(
            f"  {symbol}: {pos.quantity} shares @ avg cost {pos.average_cost:.2f}"
        )
    return "\n".join(lines)


# =====================================================================
# MCP Prompts - Workflow instructions for clients
# =====================================================================


@mcp.prompt()
def transaction_workflow() -> str:
    """Instructions for executing portfolio transactions correctly."""
    return """You are helping manage a financial portfolio. Follow these rules for transactions:

MANDATORY REQUIREMENTS:
- BUY/SELL: Need quantity, price, and date from the user. Ask if any are missing.
- DEPOSIT/WITHDRAWAL: Need amount, currency, and date. Ask if any are missing.
- NEVER guess or invent values.

INSTRUMENT RESOLUTION (before any transaction):
1. ISIN provided → resolve_instrument(isin=...) → proceed
2. Symbol provided → resolve_instrument(symbol=...) → proceed
3. Name/description → resolve_instrument(name=...) → show results → wait for user confirmation

AFTER ADDING A TRANSACTION:
If market price lookup fails for the instrument, ask user if they want to:
a) Use purchase price as market price (set_market_price with use_purchase_price=True)
b) Enter a custom price
c) Leave it unavailable

RULES:
- Respect user-specified instrument_type (bond, stock, etf) and currency exactly
- For bonds with ISINs, always use the ISIN parameter
- Confirm before modifying or deleting transactions
- Use get_transactions or get_transaction_history to find transaction IDs"""


@mcp.prompt()
def analysis_workflow() -> str:
    """Instructions for portfolio analysis and market data queries."""
    return """You are analyzing a financial portfolio. Follow these guidelines:

BEFORE ANALYSIS:
1. Check data freshness with get_data_freshness
2. If data is stale (>1 hour), use refresh_data before calculations
3. Use get_portfolio_summary for an overview of current state

AVAILABLE METRICS (via get_portfolio_metrics):
- Sharpe, Sortino, Calmar ratios
- Alpha, Beta, Information ratio
- Volatility, VaR, CVaR
- Maximum drawdown and duration
- Time-weighted and money-weighted returns
- Default benchmark: SPY

SCENARIO ANALYSIS:
- simulate_what_if: Simple exclusion scenarios (remove symbols/transactions)
- advanced_what_if: Monte Carlo with custom market assumptions
- test_hypothetical_position: Test a position before buying
- optimize_portfolio: Suggest optimal weight allocation (HRP or Markowitz)
- scenario_optimization: Compare allocations under different market conditions

GUIDELINES:
- Be quantitative and specific
- Explain what metrics mean in practical terms
- Note data age when reporting prices
- Compare to benchmarks when relevant
- Highlight risks and opportunities
- This is educational - recommend consulting financial professionals"""


# =====================================================================
# Portfolio Management Tools
# =====================================================================


@mcp.tool()
def list_portfolios() -> str:
    """List all available portfolios.

    Returns portfolio IDs and names. Use select_portfolio to switch between them.
    """
    available_ids = storage.list_portfolios()
    if not available_ids:
        return "No portfolios found."

    lines = ["Available portfolios:", ""]
    for pid in available_ids:
        portfolio = storage.load_portfolio(pid)
        if portfolio:
            name = portfolio.name or pid
            active = " (active)" if (
                portfolio_manager.current_portfolio
                and portfolio_manager.current_portfolio.id == pid
            ) else ""
            lines.append(f"- {pid}: {name}{active}")
        else:
            lines.append(f"- {pid}")
    return "\n".join(lines)


@mcp.tool()
def select_portfolio(portfolio_id: str) -> str:
    """Switch to a different portfolio.

    Args:
        portfolio_id: The ID of the portfolio to load (use list_portfolios to see available IDs)
    """
    available_ids = storage.list_portfolios()
    if portfolio_id not in available_ids:
        return f"Portfolio '{portfolio_id}' not found. Available: {', '.join(available_ids)}"

    portfolio_manager.load_portfolio(portfolio_id)
    name = ""
    if portfolio_manager.current_portfolio:
        name = portfolio_manager.current_portfolio.name or portfolio_id
    return f"Switched to portfolio: {name} ({portfolio_id})"


# =====================================================================
# Transaction Tools
# =====================================================================


@mcp.tool()
def add_transaction(
    transaction_type: str,
    price: float,
    quantity: float = 1.0,
    symbol: Optional[str] = None,
    isin: Optional[str] = None,
    instrument_type: Optional[str] = None,
    currency: Optional[str] = None,
    date: Optional[str] = None,
    days_ago: int = 0,
    notes: Optional[str] = None,
) -> str:
    """Add a transaction to the portfolio.

    For BUY/SELL: specify symbol, quantity (shares), and price (per share).
    For DEPOSIT/WITHDRAWAL: specify price (the amount), currency, and date.
    For DIVIDEND: specify symbol, price (dividend amount), and date.

    IMPORTANT - Bond Pricing:
    For bonds, prices are typically quoted as percentage of par (e.g., 96.32% of face value).
    You must convert to decimal form: enter 0.9632 (not 96.32).
    The quantity should be the face value (nominal amount).
    Market value = quantity (face value) × price (as decimal).
    Example: Buying 50,000 face value bond at 96.32% → quantity=50000, price=0.9632

    Args:
        transaction_type: Type: buy, sell, dividend, deposit, withdrawal, fees
        price: Price per share (buy/sell) or total amount (deposit/withdrawal/dividend)
        quantity: Number of shares (default 1.0, only needed for buy/sell)
        symbol: Stock symbol (e.g., AAPL, TSLA) - not needed for deposit/withdrawal
        isin: ISIN identifier (e.g., US0378331005)
        instrument_type: Type: stock, etf, bond, crypto, cash, mutual_fund, option, future
        currency: Currency code (e.g., USD, EUR) - REQUIRED for deposit/withdrawal
        date: Trade date in YYYY-MM-DD format - REQUIRED
        days_ago: Fallback: how many days ago (0 for today)
        notes: Additional notes
    """
    return _add_transaction._run(
        symbol=symbol,
        isin=isin,
        instrument_type=instrument_type,
        transaction_type=transaction_type,
        quantity=quantity,
        price=price,
        date=date,
        days_ago=days_ago,
        notes=notes,
        currency=currency,
    )


@mcp.tool()
def bulk_add_transactions(transactions: list) -> str:
    """Add multiple transactions to the portfolio in a single call.

    More efficient than calling add_transaction multiple times.

    IMPORTANT - Bond Pricing:
    For bonds, prices are typically quoted as percentage of par (e.g., 96.32% of face value).
    You must convert to decimal form: enter 0.9632 (not 96.32).
    The quantity should be the face value (nominal amount).
    Market value = quantity (face value) × price (as decimal).
    Example: Buying 50,000 face value bond at 96.32% → quantity=50000, price=0.9632

    Args:
        transactions: List of transaction objects. Each transaction should have:
            - transaction_type: buy, sell, dividend, deposit, withdrawal, fees
            - price: Price per share (buy/sell) or amount (deposit/withdrawal/dividend)
            - date: Trade date in YYYY-MM-DD format (REQUIRED)
            - symbol: Stock symbol (for buy/sell/dividend)
            - isin: ISIN identifier (alternative to symbol)
            - quantity: Number of shares (for buy/sell, default 1.0)
            - currency: Currency code (REQUIRED for deposit/withdrawal)
            - instrument_type: stock, etf, bond, crypto, etc. (optional)
            - notes: Additional notes (optional)

    Example:
        [
            {"transaction_type": "deposit", "price": 50000, "currency": "USD", "date": "2024-01-01"},
            {"transaction_type": "buy", "symbol": "AAPL", "quantity": 100, "price": 150.0, "date": "2024-01-02"},
            {"transaction_type": "buy", "symbol": "MSFT", "quantity": 50, "price": 350.0, "date": "2024-01-02"},
            {"transaction_type": "dividend", "symbol": "AAPL", "price": 25.50, "date": "2024-03-15"}
        ]
    """
    return _bulk_add_transactions._run(transactions=transactions)


@mcp.tool()
def modify_transaction(
    transaction_id: str,
    quantity: Optional[float] = None,
    price: Optional[float] = None,
    date: Optional[str] = None,
    notes: Optional[str] = None,
    instrument_type: Optional[str] = None,
) -> str:
    """Modify an existing transaction's details.

    IMPORTANT - Bond Pricing:
    For bonds, prices are typically quoted as percentage of par (e.g., 96.32% of face value).
    You must convert to decimal form: enter 0.9632 (not 96.32).

    Args:
        transaction_id: The ID of the transaction to modify
        quantity: New quantity (leave None to keep current)
        price: New price (leave None to keep current)
        date: New date in YYYY-MM-DD format (leave None to keep current)
        notes: New notes (leave None to keep current)
        instrument_type: New instrument type: stock, etf, bond, crypto, cash, mutual_fund, option, future (leave None to keep current)
    """
    return _modify_transaction._run(
        transaction_id=transaction_id,
        quantity=quantity,
        price=price,
        date=date,
        notes=notes,
        instrument_type=instrument_type,
    )


@mcp.tool()
def delete_transaction(transaction_id: str, confirm: bool = True) -> str:
    """Delete a transaction from the portfolio.

    Args:
        transaction_id: The ID of the transaction to delete
        confirm: Safety flag, must be True to proceed with deletion
    """
    return _delete_transaction._run(transaction_id=transaction_id, confirm=confirm)


@mcp.tool()
def resolve_instrument(
    isin: Optional[str] = None,
    symbol: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """Resolve an instrument's identity. Priority: ISIN > Symbol > Name search.

    Use before add_transaction when the user provides a company name instead of symbol,
    or when unsure about the correct identifier.

    Args:
        isin: ISIN identifier (highest priority)
        symbol: Stock symbol
        name: Company name or description (triggers search)
    """
    return _resolve_instrument._run(isin=isin, symbol=symbol, name=name)


@mcp.tool()
def search_instrument(query: str) -> str:
    """Search for financial instruments by symbol or name.

    Args:
        query: Search query (symbol or company name)
    """
    return _search_instrument._run(query=query)


@mcp.tool()
def search_company(query: str) -> str:
    """Search for company information and details.

    Args:
        query: Company name or symbol to search for
    """
    return _search_company._run(query=query)


# =====================================================================
# Portfolio Analysis Tools
# =====================================================================


@mcp.tool()
def get_portfolio_summary(include_metrics: bool = True) -> str:
    """Get a comprehensive summary of the current portfolio including positions, values, and performance.

    Args:
        include_metrics: Whether to include performance metrics (default True)
    """
    return _get_portfolio_summary._run(include_metrics=include_metrics)


@mcp.tool()
def get_portfolio_snapshot(target_date: str, include_local_currency: bool = False) -> str:
    """Get portfolio positions, cash balances, and total value at a specific historical date.

    Args:
        target_date: Date to get snapshot for in YYYY-MM-DD format
        include_local_currency: If True, also show price and value in the instrument's native currency
    """
    return _get_portfolio_snapshot._run(
        target_date=target_date,
        include_local_currency=include_local_currency,
    )


@mcp.tool()
def get_ytd_performance(as_of_date: Optional[str] = None) -> str:
    """Get Year-to-Date performance for all portfolio positions and the portfolio overall.

    Compares current prices to Dec 31 of the prior year. Shows per-instrument
    and portfolio-level YTD returns.

    Args:
        as_of_date: Date to calculate YTD as of in YYYY-MM-DD format (defaults to today)
    """
    return _get_ytd_performance._run(as_of_date=as_of_date)


@mcp.tool()
def get_transactions(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    symbol: Optional[str] = None,
    transaction_type: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """Get transactions with optional filters.

    Args:
        start_date: Start date in YYYY-MM-DD format (inclusive)
        end_date: End date in YYYY-MM-DD format (inclusive)
        symbol: Filter by instrument symbol (e.g., AAPL)
        transaction_type: Filter by type: buy, sell, dividend, deposit, withdrawal, fees
        limit: Maximum number of transactions to return (most recent)
    """
    return _get_transactions._run(
        start_date=start_date,
        end_date=end_date,
        symbol=symbol,
        transaction_type=transaction_type,
        limit=limit,
    )


@mcp.tool()
def get_transaction_history(days: int = 30) -> str:
    """Get recent transaction history.

    Args:
        days: Number of days to look back (default 30)
    """
    return _get_transaction_history._run(days=days)


@mcp.tool()
def get_portfolio_metrics(days: int = 365, benchmark: str = "SPY") -> str:
    """Calculate detailed portfolio performance metrics including returns, volatility, Sharpe ratio.

    Args:
        days: Number of days to analyze (default 365)
        benchmark: Benchmark symbol for comparison (default SPY)
    """
    return _get_portfolio_metrics._run(days=days, benchmark=benchmark)


# =====================================================================
# Market Data Tools
# =====================================================================


@mcp.tool()
def get_current_price(symbol: str) -> str:
    """Get the current market price for a symbol.

    Args:
        symbol: Stock/instrument symbol (e.g., AAPL, TSLA)
    """
    return _get_current_price._run(symbol=symbol)


@mcp.tool()
def set_market_price(
    symbol: str,
    price: Optional[float] = None,
    currency: Optional[str] = None,
    date: Optional[str] = None,
    use_purchase_price: bool = False,
) -> str:
    """Set or update the market price for an instrument.

    Use when price lookup fails or to manually set a custom price.

    IMPORTANT - Currency:
    Always specify the currency of the price you're entering. If the currency differs
    from the instrument's native currency, it will be automatically converted.
    Example: For a CHF-denominated fund, you can enter either:
    - price=176.12, currency="CHF" (native currency, no conversion)
    - price=225.92, currency="USD" (will be converted to CHF for storage)

    IMPORTANT - Bond Pricing:
    For bonds, prices are typically quoted as percentage of par (e.g., 96.32% of face value).
    You must convert to decimal form: enter 0.9632 (not 96.32).
    Market value = quantity (face value) × price (as decimal).
    Example: 50,000 face value bond at 96.32% → enter price=0.9632 → value = 48,160

    Args:
        symbol: The instrument symbol
        price: The price to set (required unless use_purchase_price is True).
               For bonds: use decimal (0.9632 for 96.32% of par).
               For stocks/ETFs: use absolute price per share.
        currency: Currency of the price (e.g., USD, CHF, EUR, GBP). If different from
                  the instrument's native currency, it will be converted automatically.
                  Defaults to the instrument's native currency if not specified.
        date: Date for the price in YYYY-MM-DD format (defaults to today)
        use_purchase_price: If True, uses the position's average_cost as the market price
    """
    return _set_market_price._run(
        symbol=symbol,
        price=price,
        currency=currency,
        date=date,
        use_purchase_price=use_purchase_price,
    )


@mcp.tool()
def bulk_set_market_price(prices: str, symbol: Optional[str] = None, currency: Optional[str] = None) -> str:
    """Set market prices for an instrument across multiple dates at once.

    Use for entering historical price data manually when market data isn't available.

    IMPORTANT - Currency:
    Always specify the currency of the prices you're entering:
    - For simple/single-symbol formats, use the currency parameter
    - For multi-symbol JSON, include "currency" in each entry
    Example: bulk_set_market_price(symbol="VTEQ_SWISS", prices="2024-01-01:176.12", currency="CHF")
    Example: '[{"symbol":"VTEQ_SWISS","date":"2024-01-01","price":176.12,"currency":"CHF"}]'

    IMPORTANT - Bond Pricing:
    For bonds, prices are typically quoted as percentage of par (e.g., 96.32% of face value).
    You must convert to decimal form: enter 0.9632 (not 96.32).
    Market value = quantity (face value) × price (as decimal).
    Example: 50,000 face value bond at 96.32% → price=0.9632 → value = 48,160

    Args:
        symbol: The instrument symbol (e.g., AAPL, CORP_BOND). Optional for multi-symbol JSON.
        prices: Price data in one of these formats:
                1. Simple: "2024-01-01:150.0,2024-01-02:152.5,2024-01-03:148.0"
                2. JSON: '[{"date":"2024-01-01","price":150.0},{"date":"2024-01-02","price":152.5}]'
                3. Multi-symbol JSON: '[{"symbol":"AAPL","date":"2024-01-01","price":150,"currency":"USD"}]'
        currency: Currency of the prices (e.g., USD, CHF, EUR). For simple/single-symbol formats.
                  For multi-symbol JSON, specify currency per entry instead.

    Note: In multi-symbol mode, ISINs are automatically resolved to portfolio symbols
    if the instrument has a matching ISIN stored (e.g., XS2472298335 -> GLENCORE_2028).
    """
    return _bulk_set_market_price._run(symbol=symbol, prices=prices, currency=currency)


@mcp.tool()
def fetch_and_update_prices(
    symbol: str,
    start_date: str,
    end_date: str,
    provider_symbol: Optional[str] = None,
) -> str:
    """Fetch historical prices from data provider and update market data.

    Use this to sync portfolio with market prices. If the portfolio symbol differs
    from the data provider symbol, use provider_symbol to specify the lookup symbol.

    Args:
        symbol: The symbol in your portfolio (e.g., BTC, CORP_BOND)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        provider_symbol: Symbol to use with data provider if different (e.g., BTC-USD for Bitcoin)

    If this fails, use bulk_set_market_price to manually enter prices.
    """
    return _fetch_and_update_prices._run(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        provider_symbol=provider_symbol,
    )


@mcp.tool()
def set_data_provider_symbol(
    symbol: str,
    data_provider_symbol: str,
) -> str:
    """Set the data provider symbol for a portfolio position.

    Use this when the portfolio symbol differs from the symbol used by the data provider.
    Once set, all future price lookups (Quick Refresh, Update Market Data) will use the
    data provider symbol automatically.

    Args:
        symbol: Portfolio symbol (the symbol used in your portfolio)
        data_provider_symbol: Symbol to use with the data provider (e.g., BTC-USD for Bitcoin)

    Examples:
        - Bitcoin: set_data_provider_symbol(symbol="BTC", data_provider_symbol="BTC-USD")
        - Ethereum: set_data_provider_symbol(symbol="ETH", data_provider_symbol="ETH-USD")
    """
    return _set_data_provider_symbol._run(
        symbol=symbol,
        data_provider_symbol=data_provider_symbol,
    )


@mcp.tool()
def set_price_currency(symbol: str, price_currency: str) -> str:
    """Set the price currency for a portfolio position.

    Use this when the data provider returns prices in a different currency than the
    instrument is tracked in. The system will automatically convert fetched prices
    to the instrument's portfolio currency before storing.

    Example workflow for CNKY (tracked in JPY, fetched from LSE in GBX/GBP):
        set_data_provider_symbol(symbol="CNKY", data_provider_symbol="CNKY.L")
        set_price_currency(symbol="CNKY", price_currency="GBP")

    Note: GBX (pence) is auto-converted to GBP by the data provider, so always
    use GBP (not GBX) for LSE-listed instruments.

    Args:
        symbol: Portfolio symbol (the symbol used in your portfolio)
        price_currency: Currency the data provider returns prices in (e.g., GBP, USD, EUR)
    """
    return _set_price_currency._run(symbol=symbol, price_currency=price_currency)


@mcp.tool()
def check_market_data_availability(
    isin: Optional[str] = None,
    symbol: Optional[str] = None,
    name: Optional[str] = None,
    verify_price_data: bool = True,
) -> str:
    """Check if market data is available for an instrument without fetching full data.

    Accepts ISIN, symbol, or name. Verifies data provider access before fetching.

    Args:
        isin: ISIN identifier (e.g., US0378331005)
        symbol: Stock symbol (e.g., AAPL)
        name: Company name (e.g., Apple)
        verify_price_data: Whether to verify price data can be fetched (default True)
    """
    return _check_market_data._run(
        isin=isin, symbol=symbol, name=name, verify_price_data=verify_price_data
    )


@mcp.tool()
def get_price_history(symbol: str, start_date: str, end_date: str) -> str:
    """Get historical OHLCV price data for a symbol over a date range.

    Args:
        symbol: Stock/instrument symbol (e.g., AAPL, TSLA)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    """
    return _get_price_history._run(
        symbol=symbol, start_date=start_date, end_date=end_date
    )


@mcp.tool()
def get_moving_average_signal(
    symbol: str,
    short_period: int = 50,
    long_period: int = 200,
) -> str:
    """Calculate moving average crossover signal for a symbol.

    Computes short-term MA (default 50-day) minus long-term MA (default 200-day).

    Signal interpretation:
    - Positive difference = BUY signal (Golden Cross territory)
    - Negative difference = SELL signal (Death Cross territory)

    Args:
        symbol: Stock/instrument symbol (e.g., AAPL, MSFT)
        short_period: Short-term MA period in days (default 50)
        long_period: Long-term MA period in days (default 200)
    """
    return _get_ma_signal._run(
        symbol=symbol,
        short_period=short_period,
        long_period=long_period,
    )


@mcp.tool()
def get_fx_rate(
    from_currency: str,
    to_currency: str,
    as_of: Optional[str] = None,
) -> str:
    """Get the exchange rate between two currencies.

    Args:
        from_currency: Source currency code (e.g., EUR, GBP)
        to_currency: Target currency code (e.g., USD)
        as_of: Optional date in YYYY-MM-DD format (defaults to latest)
    """
    return _get_fx_rate._run(
        from_currency=from_currency, to_currency=to_currency, as_of=as_of
    )


@mcp.tool()
def fetch_historical_fx_rates(
    from_currency: str,
    to_currency: str,
    start_date: str,
    end_date: str,
) -> str:
    """Fetch and store historical FX rates for a currency pair over a date range.

    Args:
        from_currency: Source currency code (e.g., EUR, GBP)
        to_currency: Target currency code (e.g., USD, CHF)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    """
    return _fetch_historical_fx_rates._run(
        from_currency=from_currency,
        to_currency=to_currency,
        start_date=start_date,
        end_date=end_date,
    )


@mcp.tool()
def get_batch_prices(symbols: str) -> str:
    """Get current prices for multiple symbols at once.

    Args:
        symbols: Comma-separated list of symbols (e.g., "AAPL,MSFT,GOOGL")
    """
    return _get_batch_prices._run(symbols=symbols)


@mcp.tool()
def get_data_freshness() -> str:
    """Check the freshness/staleness of cached market data for all portfolio positions."""
    return _get_data_freshness._run()


@mcp.tool()
def refresh_data() -> str:
    """Force a fresh fetch of market data for all portfolio positions and FX rates."""
    return _refresh_data._run()


@mcp.tool()
def get_historical_instruments(start_date: str, end_date: str) -> str:
    """List instruments that were held in a date range but are no longer in current positions.

    Use cases:
    - Find sold instruments that need market data updates
    - See complete instrument history for a period
    - Identify gaps in market data coverage

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    """
    return _get_historical_instruments._run(start_date=start_date, end_date=end_date)


@mcp.tool()
def update_historical_market_data(
    start_date: str,
    end_date: str,
    symbol: Optional[str] = None,
    include_historical: bool = True,
) -> str:
    """Update historical market data for one or all instruments in a date range.

    Respects price_currency settings — prices are converted to the instrument's
    portfolio currency before storing (e.g. GBP → JPY for CNKY).

    Prefer providing a symbol to avoid slow bulk updates.

    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        symbol: Specific instrument to update (recommended). If omitted, ALL instruments are updated.
        include_historical: Include sold instruments (default True, ignored when symbol is provided)
    """
    return _update_historical_market_data._run(
        start_date=start_date,
        end_date=end_date,
        symbol=symbol,
        include_historical=include_historical,
    )


@mcp.tool()
def interpolate_prices(
    start_date: str,
    end_date: str,
    symbols: Optional[str] = None,
) -> str:
    """Fill in missing market prices using linear interpolation between known values.

    Use cases:
    - Filling gaps in market data for bonds/structured products without live feeds
    - Smoothing out missing data points between known prices
    - Fixing portfolio valuation gaps caused by missing historical data

    How it works:
    - Finds the nearest available price before and after the date range
    - Calculates daily price change using linear interpolation
    - Fills in all missing dates between the boundary prices
    - Skips dates that already have prices (won't overwrite existing data)

    Args:
        start_date: Start of date range to fill (YYYY-MM-DD)
        end_date: End of date range to fill (YYYY-MM-DD)
        symbols: Comma-separated list of symbols (e.g., "GLENCORE_2028,BAYER_2026").
                 If not provided, interpolates all current portfolio positions.
    """
    return _interpolate_prices._run(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
    )


# =====================================================================
# Scenario & Optimization Tools
# =====================================================================


@mcp.tool()
def simulate_what_if(
    start: str,
    end: str,
    exclude_symbols: str = "",
    exclude_txn_ids: str = "",
) -> str:
    """Simulate portfolio performance excluding certain transactions or symbols.

    Args:
        start: Start date in YYYY-MM-DD format
        end: End date in YYYY-MM-DD format
        exclude_symbols: Comma-separated symbols to exclude (e.g., "AAPL,TSLA")
        exclude_txn_ids: Comma-separated transaction IDs to exclude
    """
    return _simulate_what_if._run(
        start=start,
        end=end,
        exclude_symbols=exclude_symbols,
        exclude_txn_ids=exclude_txn_ids,
    )


@mcp.tool()
def advanced_what_if(
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
    """Run advanced what-if scenarios with Monte Carlo simulation.

    Args:
        scenario_type: Scenario type (optimistic, likely, pessimistic, stress, custom)
        projection_years: Years to project forward (0.5 to 10)
        monte_carlo_runs: Number of Monte Carlo simulation runs (100-5000)
        modify_positions: Position modifications, e.g. "AAPL:+50%,MSFT:-25%,GOOGL:=150"
        add_positions: New positions to add. Formats:
            - With price: "NVDA:100@$800,TSLA:50@$250"
            - Auto-fetch price: "NVDA:100,TSLA:50" (will fetch current market prices)
        market_return: Expected annual market return (decimal, e.g. 0.08 for 8%)
        market_volatility: Expected annual market volatility (decimal, e.g. 0.20 for 20%)
        recurring_deposits: Monthly recurring deposit amount in USD
        stress_test: Whether to include stress testing conditions
    """
    return _advanced_what_if._run(
        scenario_type=scenario_type,
        projection_years=projection_years,
        monte_carlo_runs=monte_carlo_runs,
        modify_positions=modify_positions,
        add_positions=add_positions,
        market_return=market_return,
        market_volatility=market_volatility,
        recurring_deposits=recurring_deposits,
        stress_test=stress_test,
    )


@mcp.tool()
def test_hypothetical_position(
    symbol: str,
    quantity: float = 0,
    purchase_price: float = 0,
    investment_amount: str = "",
    scenario: str = "likely",
    time_horizon: float = 1.0,
) -> str:
    """Test a hypothetical position to see projected outcomes before buying.

    Analyzes the symbol and returns:
    - Historical volatility (annualized) and risk level
    - Historical annual return and 1-year price change
    - Monte Carlo simulation projections under the selected scenario

    Args:
        symbol: Stock symbol to test
        quantity: Number of shares (use 0 if using investment_amount)
        purchase_price: Price per share (0 to auto-fetch current market price)
        investment_amount: Total investment amount (alternative to quantity), e.g. "$5000"
        scenario: Market scenario (optimistic, likely, pessimistic, stress)
        time_horizon: Projection period in years (0.5 to 5.0)

    Example: test_hypothetical_position("AAPL", investment_amount="$5000") will
    auto-fetch current AAPL price and calculate shares accordingly.
    """
    return _hypothetical_position._run(
        symbol=symbol,
        quantity=quantity,
        purchase_price=purchase_price,
        investment_amount=investment_amount,
        scenario=scenario,
        time_horizon=time_horizon,
    )


@mcp.tool()
def optimize_portfolio(
    locked_symbols: Optional[str] = None,
    method: str = "hrp",
    compare: bool = True,
    lookback_days: int = 252,
    objective: str = "max_sharpe",
    target_volatility: Optional[float] = None,
    include_cash: bool = True,
    risk_free_rate: float = 0.04,
) -> str:
    """Analyze current portfolio and suggest optimal weight allocation for rebalancing.

    Methods: HRP (Hierarchical Risk Parity) or Markowitz (Mean-Variance).
    Objectives: max_sharpe, min_volatility, efficient_risk.

    Args:
        locked_symbols: Comma-separated symbols to keep at current weight (e.g., "AAPL,MSFT")
        method: Optimization method: "hrp" or "markowitz"
        compare: Whether to compare results from both methods
        lookback_days: Days of historical data to use (default 252 = 1 year)
        objective: Optimization objective (max_sharpe, min_volatility, efficient_risk)
        target_volatility: Target volatility for efficient_risk objective
        include_cash: Whether to include cash position in optimization
        risk_free_rate: Risk-free rate assumption (default 0.04 = 4%)
    """
    return _optimize_portfolio._run(
        locked_symbols=locked_symbols,
        method=method,
        compare=compare,
        lookback_days=lookback_days,
        objective=objective,
        target_volatility=target_volatility,
        include_cash=include_cash,
        risk_free_rate=risk_free_rate,
    )


@mcp.tool()
def scenario_optimization(
    scenarios: str = "optimistic,likely,pessimistic,stress",
    objective: str = "max_sharpe",
    include_cash: bool = True,
    projection_years: float = 5.0,
    monte_carlo_runs: int = 1000,
    confidence_levels: str = "50,75,90",
) -> str:
    """Compare how optimal portfolio allocation changes under different market scenarios.

    Available scenarios: optimistic (bull market), likely (historical average),
    pessimistic (economic downturn), stress (market crash).

    Args:
        scenarios: Comma-separated scenarios to evaluate
        objective: Optimization objective (max_sharpe, min_volatility)
        include_cash: Whether to include cash position
        projection_years: Years to project forward
        monte_carlo_runs: Number of Monte Carlo runs per scenario
        confidence_levels: Comma-separated confidence levels for projections
    """
    return _scenario_optimization._run(
        scenarios=scenarios,
        objective=objective,
        include_cash=include_cash,
        projection_years=projection_years,
        monte_carlo_runs=monte_carlo_runs,
        confidence_levels=confidence_levels,
    )


# =====================================================================
# Utility Tools
# =====================================================================


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports basic arithmetic, percentages, and common functions.

    Args:
        expression: Mathematical expression to evaluate (e.g., "100 * 1.05 ** 10")
    """
    return _calculator._run(expression=expression)


@mcp.tool()
def ingest_pdf(path: str) -> str:
    """Extract and summarize content from a PDF file (e.g., brokerage statements).

    Args:
        path: Path to the PDF file to ingest
    """
    return _ingest_pdf._run(path=path)


# --- Entry point ---


def main():
    """Run the MCP server.

    Supports two transports:
      --stdio   : For Claude Desktop and other stdio-based clients (no auth needed)
      --sse     : HTTP server with SSE (default), requires FINAI_API_KEY
    """
    if "--stdio" in sys.argv:
        mcp.run(transport="stdio")
    else:
        _run_sse()


def _run_sse():
    """Run the SSE (HTTP) server with API key auth."""
    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route

    host = os.environ.get("HOST", "localhost")
    port = int(os.environ.get("PORT", "8000"))
    api_key = os.environ.get("FINAI_API_KEY", "finai-api-key")

    if not api_key:
        logger.error("FINAI_API_KEY environment variable is required")
        sys.exit(1)

    # SSE transport with /messages endpoint for client posts
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            await mcp._mcp_server.run(
                read_stream,
                write_stream,
                mcp._mcp_server.create_initialization_options(),
            )

    # Auth middleware - validates Bearer token on all requests
    class APIKeyAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer ") or auth_header[7:] != api_key:
                return JSONResponse(
                    {"error": "Unauthorized - invalid or missing API key"},
                    status_code=401,
                )
            return await call_next(request)

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
        middleware=[Middleware(APIKeyAuthMiddleware)],
    )

    logger.info(f"Starting FinAI MCP server on {host}:{port}")
    logger.info(f"SSE endpoint: http://{host}:{port}/sse")
    logger.info("API key auth enabled")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
