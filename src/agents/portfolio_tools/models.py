"""
Pydantic input models for portfolio tools.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class PortfolioToolInput(BaseModel):
    """Base input model for portfolio tools."""

    pass


class AddTransactionInput(PortfolioToolInput):
    """Input for adding a transaction."""

    symbol: Optional[str] = Field(
        default=None, description="Stock symbol (e.g., AAPL, TSLA). Not needed for deposit/withdrawal."
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
    quantity: float = Field(
        default=1.0,
        description="Number of shares (for buy/sell). Default 1.0, not needed for deposit/withdrawal."
    )
    price: float = Field(
        description="Price per share (buy/sell) OR the amount (deposit/withdrawal/dividend)"
    )
    currency: Optional[str] = Field(
        default=None,
        description="Currency code (e.g., USD, EUR, GBP, CHF). REQUIRED for deposits/withdrawals.",
    )
    date: Optional[str] = Field(
        default=None,
        description="Trade date in YYYY-MM-DD format. REQUIRED for all transactions.",
    )
    days_ago: int = Field(
        default=0, description="Fallback: how many days ago (0 for today)"
    )
    notes: Optional[str] = Field(default=None, description="Additional notes")


class TransactionItem(BaseModel):
    """A single transaction item for bulk operations."""

    symbol: Optional[str] = Field(
        default=None,
        description="Stock symbol (e.g., AAPL, TSLA). Not needed for deposit/withdrawal.",
    )
    isin: Optional[str] = Field(
        default=None, description="ISIN identifier (e.g., US0378331005)"
    )
    instrument_type: Optional[str] = Field(
        default=None,
        description="Type: stock, etf, bond, crypto, cash, mutual_fund, option, future. Auto-detected if not specified.",
    )
    transaction_type: str = Field(
        description="Type: buy, sell, dividend, deposit, withdrawal, fees"
    )
    quantity: float = Field(
        default=1.0,
        description="Number of shares (for buy/sell). Default 1.0, not needed for deposit/withdrawal.",
    )
    price: float = Field(
        description="Price per share (buy/sell) OR the amount (deposit/withdrawal/dividend)"
    )
    currency: Optional[str] = Field(
        default=None,
        description="Currency code (e.g., USD, EUR, GBP, CHF). REQUIRED for deposits/withdrawals.",
    )
    date: str = Field(description="Trade date in YYYY-MM-DD format. REQUIRED.")
    notes: Optional[str] = Field(default=None, description="Additional notes")


class BulkAddTransactionsInput(BaseModel):
    """Input for bulk adding transactions."""

    transactions: List[TransactionItem] = Field(
        description="List of transactions to add"
    )


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
    start_date: Optional[str] = Field(default=None, description="Start date in YYYY-MM-DD format. When provided together with end_date, overrides 'days'.")
    end_date: Optional[str] = Field(default=None, description="End date in YYYY-MM-DD format. Defaults to today when start_date is provided.")


class GetPortfolioSnapshotInput(BaseModel):
    """Input for getting portfolio snapshot at a specific date."""

    target_date: str = Field(
        description="Date to get portfolio snapshot for in YYYY-MM-DD format"
    )
    include_local_currency: bool = Field(
        default=False,
        description="If True, also show position price and value in the instrument's native currency",
    )


class GetTransactionsInput(BaseModel):
    """Input for getting transactions with optional filters."""

    start_date: Optional[str] = Field(
        default=None,
        description="Start date filter in YYYY-MM-DD format (inclusive)",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date filter in YYYY-MM-DD format (inclusive)",
    )
    symbol: Optional[str] = Field(
        default=None,
        description="Filter by instrument symbol (e.g., AAPL, MSFT)",
    )
    transaction_type: Optional[str] = Field(
        default=None,
        description="Filter by type: buy, sell, dividend, deposit, withdrawal, fees",
    )
    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of transactions to return (most recent first if filtering)",
    )


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


class CheckMarketDataAvailabilityInput(BaseModel):
    """Input for checking market data availability."""

    isin: Optional[str] = Field(
        default=None, description="ISIN identifier to check"
    )
    symbol: Optional[str] = Field(
        default=None, description="Stock/instrument symbol to check"
    )
    name: Optional[str] = Field(
        default=None, description="Company or instrument name to search"
    )
    verify_price_data: bool = Field(
        default=True,
        description="If True, also verify that price data can be fetched (lightweight test)"
    )


class ModifyTransactionInput(BaseModel):
    """Input for modifying a transaction."""

    transaction_id: str = Field(description="ID of the transaction to modify")
    quantity: Optional[float] = Field(default=None, description="New quantity (optional)")
    price: Optional[float] = Field(default=None, description="New price (optional)")
    date: Optional[str] = Field(
        default=None, description="New date in YYYY-MM-DD format (optional)"
    )
    notes: Optional[str] = Field(default=None, description="New notes (optional)")
    instrument_type: Optional[str] = Field(
        default=None,
        description="New instrument type: stock, etf, bond, crypto, cash, mutual_fund, option, future (optional)"
    )


class DeleteTransactionInput(BaseModel):
    """Input for deleting a transaction."""

    transaction_id: str = Field(description="ID of the transaction to delete")
    confirm: bool = Field(
        default=True, description="Set to True to confirm deletion"
    )


class SetMarketPriceInput(BaseModel):
    """Input for setting market price."""

    symbol: str = Field(description="Stock/instrument symbol (e.g., AAPL, TSLA)")
    price: Optional[float] = Field(
        default=None,
        description="Price to set. Required unless use_purchase_price is True.",
    )
    currency: Optional[str] = Field(
        default=None,
        description="Currency of the price (e.g., USD, CHF, EUR). If different from the instrument's native currency, it will be converted. Defaults to the instrument's native currency.",
    )
    date: Optional[str] = Field(
        default=None,
        description="Date for the price in YYYY-MM-DD format. Defaults to today.",
    )
    use_purchase_price: bool = Field(
        default=False,
        description="If True, uses the position's purchase price (average_cost) as the market price instead of the 'price' parameter.",
    )


class FetchAndUpdatePricesInput(BaseModel):
    """Input for fetching prices from provider and updating portfolio."""

    symbol: str = Field(
        description="Portfolio symbol (the symbol used in your portfolio, e.g., BTC, CORP_BOND)"
    )
    start_date: str = Field(
        description="Start date in YYYY-MM-DD format"
    )
    end_date: str = Field(
        description="End date in YYYY-MM-DD format"
    )
    provider_symbol: Optional[str] = Field(
        default=None,
        description="Symbol to use with the data provider if different from portfolio symbol (e.g., BTC-USD for Bitcoin, AAPL for Apple). If not provided, uses the portfolio symbol."
    )


class SetDataProviderSymbolInput(BaseModel):
    """Input for setting the data provider symbol for a position."""

    symbol: str = Field(
        description="Portfolio symbol (the symbol used in your portfolio)"
    )
    data_provider_symbol: str = Field(
        description="Symbol to use with the data provider (e.g., BTC-USD for Bitcoin on Yahoo Finance)"
    )


class SetPriceCurrencyInput(BaseModel):
    """Input for setting the price currency for a position."""

    symbol: str = Field(
        description="Portfolio symbol (the symbol used in your portfolio)"
    )
    price_currency: str = Field(
        description=(
            "Currency the data provider returns prices in (e.g., GBP when provider returns "
            "GBX/GBP but the instrument is tracked in JPY). "
            "Supported values: USD, EUR, GBP, JPY, CHF, CAD, AUD."
        )
    )


class BulkSetMarketPriceInput(BaseModel):
    """Input for bulk setting market prices."""

    symbol: Optional[str] = Field(
        default=None,
        description="Stock/instrument symbol (e.g., AAPL, TSLA). Optional when using multi-symbol JSON format."
    )
    prices: str = Field(
        description="""Price data in one of three formats:
        1. Simple (requires symbol): "YYYY-MM-DD:price,YYYY-MM-DD:price,..." (e.g., "2024-01-01:150.0,2024-01-02:152.5")
        2. Single-symbol JSON (requires symbol): '[{"date":"2024-01-01","price":150.0},{"date":"2024-01-02","price":152.5}]'
        3. Multi-symbol JSON (symbol not needed): '[{"symbol":"AAPL","date":"2024-01-01","price":150.0,"currency":"USD"},{"symbol":"VTEQ_SWISS","date":"2024-01-01","price":176.12,"currency":"CHF"}]'"""
    )
    currency: Optional[str] = Field(
        default=None,
        description="Currency for all prices in simple/single-symbol format (e.g., USD, CHF, EUR). For multi-symbol JSON, specify currency per entry. Defaults to each instrument's native currency."
    )


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


class GetHistoricalInstrumentsInput(BaseModel):
    """Input for getting historical instruments."""

    start_date: str = Field(
        description="Start date in YYYY-MM-DD format"
    )
    end_date: str = Field(
        description="End date in YYYY-MM-DD format"
    )


class UpdateHistoricalMarketDataInput(BaseModel):
    """Input for updating historical market data."""

    start_date: str = Field(
        description="Start date in YYYY-MM-DD format"
    )
    end_date: str = Field(
        description="End date in YYYY-MM-DD format"
    )
    symbol: Optional[str] = Field(
        default=None,
        description="If provided, only update this specific instrument. If omitted, all instruments are updated."
    )
    include_historical: bool = Field(
        default=True,
        description="Whether to include sold instruments from transaction history (ignored when symbol is provided)"
    )


class GetYTDPerformanceInput(BaseModel):
    """Input for getting YTD performance."""

    as_of_date: Optional[str] = Field(
        default=None,
        description="Date to calculate YTD performance as of, in YYYY-MM-DD format (defaults to today)",
    )


class InterpolatePricesInput(BaseModel):
    """Input for interpolating missing market prices."""

    symbols: Optional[str] = Field(
        default=None,
        description="Comma-separated list of symbols to interpolate (e.g., 'AAPL,MSFT'). If not provided, interpolates all portfolio positions."
    )
    start_date: str = Field(
        description="Start date of the range to fill in YYYY-MM-DD format"
    )
    end_date: str = Field(
        description="End date of the range to fill in YYYY-MM-DD format"
    )
