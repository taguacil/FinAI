"""
MarketDataService tools for the Analytics Agent.

Exposes market data functionality including price history, FX rates,
batch pricing, and data freshness monitoring.
"""

from datetime import date
from typing import List, Optional

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ...portfolio.manager import PortfolioManager
from ...portfolio.models import Currency
from ...services.market_data_service import MarketDataService


class GetPriceHistoryInput(BaseModel):
    """Input for price history lookup."""

    symbol: str = Field(description="Stock/instrument symbol (e.g., AAPL, TSLA)")
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")


class GetPriceHistoryTool(BaseTool):
    """Tool for getting historical price data."""

    name: str = "get_price_history"
    description: str = """Get historical price data for a symbol over a date range.

    Returns daily OHLCV (Open, High, Low, Close, Volume) data.
    Useful for analyzing price trends, calculating returns, and technical analysis.

    Example: Get AAPL prices from 2024-01-01 to 2024-06-30
    """
    args_schema: type[BaseModel] = GetPriceHistoryInput
    market_data_service: Optional[MarketDataService] = None

    def __init__(self, market_data_service: MarketDataService):
        super().__init__()
        self.market_data_service = market_data_service

    def _run(self, symbol: str, start_date: str, end_date: str) -> str:
        """Get historical prices."""
        try:
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)

            df = self.market_data_service.get_price_history(
                symbol.upper(), start, end
            )

            if df.empty:
                return f"❌ No price history found for {symbol.upper()} between {start_date} and {end_date}"

            # Format as readable table
            result = [
                f"📈 **Price History for {symbol.upper()}**",
                f"Period: {start_date} to {end_date}",
                f"Data points: {len(df)}",
                "",
                "| Date | Open | High | Low | Close | Volume |",
                "|------|------|------|-----|-------|--------|",
            ]

            # Show first 10 and last 5 rows if more than 20 rows
            if len(df) > 20:
                for idx in df.head(10).itertuples():
                    result.append(
                        f"| {idx.Index.strftime('%Y-%m-%d')} | "
                        f"{idx.open:.2f} | {idx.high:.2f} | {idx.low:.2f} | "
                        f"{idx.close:.2f} | {idx.volume:,.0f} |"
                    )
                result.append("| ... | ... | ... | ... | ... | ... |")
                for idx in df.tail(5).itertuples():
                    result.append(
                        f"| {idx.Index.strftime('%Y-%m-%d')} | "
                        f"{idx.open:.2f} | {idx.high:.2f} | {idx.low:.2f} | "
                        f"{idx.close:.2f} | {idx.volume:,.0f} |"
                    )
            else:
                for idx in df.itertuples():
                    result.append(
                        f"| {idx.Index.strftime('%Y-%m-%d')} | "
                        f"{idx.open:.2f} | {idx.high:.2f} | {idx.low:.2f} | "
                        f"{idx.close:.2f} | {idx.volume:,.0f} |"
                    )

            # Add summary statistics
            if "close" in df.columns and len(df) > 1:
                first_close = df["close"].iloc[0]
                last_close = df["close"].iloc[-1]
                pct_change = ((last_close - first_close) / first_close) * 100
                high = df["high"].max()
                low = df["low"].min()

                result.extend([
                    "",
                    "**Summary:**",
                    f"• Period Return: {pct_change:+.2f}%",
                    f"• Period High: {high:.2f}",
                    f"• Period Low: {low:.2f}",
                    f"• First Close: {first_close:.2f}",
                    f"• Last Close: {last_close:.2f}",
                ])

            return "\n".join(result)

        except ValueError as e:
            return f"❌ Invalid date format: {str(e)}. Use YYYY-MM-DD."
        except Exception as e:
            return f"❌ Error getting price history: {str(e)}"


class GetFXRateInput(BaseModel):
    """Input for FX rate lookup."""

    from_currency: str = Field(description="Source currency code (e.g., EUR, GBP)")
    to_currency: str = Field(description="Target currency code (e.g., USD, CHF)")
    as_of: Optional[str] = Field(
        default=None,
        description="Historical date in YYYY-MM-DD format (optional, defaults to current rate)"
    )


class GetFXRateTool(BaseTool):
    """Tool for getting exchange rates."""

    name: str = "get_fx_rate"
    description: str = """Get exchange rate between two currencies.

    Returns the current or historical exchange rate with staleness information.
    Useful for currency conversion and multi-currency portfolio analysis.

    Example: Get EUR to USD rate, or historical rate for 2024-01-15
    """
    args_schema: type[BaseModel] = GetFXRateInput
    market_data_service: Optional[MarketDataService] = None

    def __init__(self, market_data_service: MarketDataService):
        super().__init__()
        self.market_data_service = market_data_service

    def _run(
        self,
        from_currency: str,
        to_currency: str,
        as_of: Optional[str] = None,
    ) -> str:
        """Get exchange rate."""
        try:
            from_cur = Currency(from_currency.upper())
            to_cur = Currency(to_currency.upper())

            as_of_date = date.fromisoformat(as_of) if as_of else None

            result = self.market_data_service.get_fx_rate(
                from_cur, to_cur, as_of_date
            )

            if result.error:
                return f"❌ {result.error}"

            if result.rate is None:
                return f"❌ No exchange rate available for {from_currency} → {to_currency}"

            # Format response
            rate_str = f"{float(result.rate):.6f}"
            date_str = result.as_of_date.isoformat() if result.as_of_date else "current"
            staleness = "⚠️ Stale" if result.is_stale else "✅ Fresh"

            lines = [
                f"💱 **Exchange Rate: {from_currency.upper()} → {to_currency.upper()}**",
                "",
                f"• Rate: 1 {from_currency.upper()} = {rate_str} {to_currency.upper()}",
                f"• As of: {date_str}",
                f"• Status: {staleness}",
            ]

            if result.age_seconds is not None:
                age_mins = result.age_seconds / 60
                if age_mins < 60:
                    lines.append(f"• Age: {int(age_mins)} minutes")
                else:
                    lines.append(f"• Age: {age_mins / 60:.1f} hours")

            # Add inverse rate for convenience
            inverse = 1 / float(result.rate)
            lines.extend([
                "",
                f"**Inverse:** 1 {to_currency.upper()} = {inverse:.6f} {from_currency.upper()}",
            ])

            return "\n".join(lines)

        except ValueError as e:
            return f"❌ Invalid currency or date: {str(e)}"
        except Exception as e:
            return f"❌ Error getting FX rate: {str(e)}"


class GetBatchPricesInput(BaseModel):
    """Input for batch price lookup."""

    symbols: str = Field(
        description="Comma-separated list of symbols (e.g., 'AAPL,MSFT,GOOGL')"
    )


class GetBatchPricesTool(BaseTool):
    """Tool for getting current prices for multiple symbols."""

    name: str = "get_batch_prices"
    description: str = """Get current prices for multiple symbols at once.

    More efficient than calling get_current_price repeatedly.
    Returns prices with staleness information for each symbol.

    Example: Get prices for AAPL,MSFT,GOOGL,TSLA
    """
    args_schema: type[BaseModel] = GetBatchPricesInput
    market_data_service: Optional[MarketDataService] = None

    def __init__(self, market_data_service: MarketDataService):
        super().__init__()
        self.market_data_service = market_data_service

    def _run(self, symbols: str) -> str:
        """Get batch prices."""
        try:
            symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

            if not symbol_list:
                return "❌ No valid symbols provided."

            results = self.market_data_service.get_current_prices(symbol_list)

            lines = [
                f"💰 **Current Prices** ({len(symbol_list)} symbols)",
                "",
                "| Symbol | Price | Status |",
                "|--------|-------|--------|",
            ]

            success_count = 0
            for symbol in symbol_list:
                result = results.get(symbol)
                if result and result.price is not None:
                    status = "⚠️ Stale" if result.is_stale else "✅"
                    lines.append(f"| {symbol} | {float(result.price):.2f} | {status} |")
                    success_count += 1
                else:
                    error = result.error if result else "Not found"
                    lines.append(f"| {symbol} | N/A | ❌ {error} |")

            lines.extend([
                "",
                f"**Retrieved:** {success_count}/{len(symbol_list)} prices",
            ])

            return "\n".join(lines)

        except Exception as e:
            return f"❌ Error getting batch prices: {str(e)}"


class GetDataFreshnessTool(BaseTool):
    """Tool for checking data freshness status."""

    name: str = "get_data_freshness"
    description: str = """Check how fresh the market data is.

    Returns information about when prices and FX rates were last updated,
    and whether the data is considered stale.

    Useful for understanding data quality before making decisions.
    """
    market_data_service: Optional[MarketDataService] = None

    def __init__(self, market_data_service: MarketDataService):
        super().__init__()
        self.market_data_service = market_data_service

    def _run(self) -> str:
        """Check data freshness."""
        try:
            freshness = self.market_data_service.freshness

            # Determine overall status
            if freshness.is_stale:
                status = "⚠️ Data is STALE - consider refreshing"
            else:
                status = "✅ Data is FRESH"

            lines = [
                f"📊 **Data Freshness Status**",
                "",
                f"**Overall:** {status}",
                "",
                "**Price Data:**",
                f"• Last Update: {freshness.freshness_display}",
                f"• Symbols Tracked: {freshness.symbols_updated}",
            ]

            if freshness.prices_age_minutes is not None:
                lines.append(f"• Age: {freshness.prices_age_minutes:.1f} minutes")

            lines.extend([
                "",
                "**FX Data:**",
                f"• Pairs Tracked: {freshness.fx_pairs_updated}",
            ])

            if freshness.fx_age_minutes is not None:
                lines.append(f"• Age: {freshness.fx_age_minutes:.1f} minutes")

            if freshness.errors:
                lines.extend([
                    "",
                    "**Recent Errors:**",
                ] + [f"• {err}" for err in freshness.errors[:5]])

            return "\n".join(lines)

        except Exception as e:
            return f"❌ Error checking data freshness: {str(e)}"


class RefreshDataTool(BaseTool):
    """Tool for refreshing market data."""

    name: str = "refresh_data"
    description: str = """Refresh market data for all portfolio positions.

    Forces a fresh fetch of:
    - Current prices for all positions
    - FX rates for all currencies in the portfolio

    Use this when data is stale or before making important decisions.
    """
    market_data_service: Optional[MarketDataService] = None
    portfolio_manager: Optional[PortfolioManager] = None

    def __init__(
        self,
        market_data_service: MarketDataService,
        portfolio_manager: PortfolioManager,
    ):
        super().__init__()
        self.market_data_service = market_data_service
        self.portfolio_manager = portfolio_manager

    def _run(self) -> str:
        """Refresh market data."""
        try:
            if not self.portfolio_manager.current_portfolio:
                return "❌ No portfolio loaded."

            portfolio = self.portfolio_manager.current_portfolio
            result = self.market_data_service.refresh_all(portfolio)

            if result.success:
                status = "✅ Refresh completed successfully"
            else:
                status = "⚠️ Refresh completed with errors"

            lines = [
                f"🔄 **Data Refresh Results**",
                "",
                f"**Status:** {status}",
                f"**Duration:** {result.duration_seconds:.2f} seconds",
                "",
                "**Updated:**",
                f"• Prices: {result.symbols_updated} symbols",
                f"• FX Rates: {result.fx_pairs_updated} pairs",
            ]

            if result.errors:
                lines.extend([
                    "",
                    "**Errors:**",
                ] + [f"• {err}" for err in result.errors])

            return "\n".join(lines)

        except Exception as e:
            return f"❌ Error refreshing data: {str(e)}"


def create_market_data_tools(
    market_data_service: MarketDataService,
    portfolio_manager: PortfolioManager,
) -> List[BaseTool]:
    """Create all MarketDataService tools for the Analytics Agent."""
    return [
        GetPriceHistoryTool(market_data_service),
        GetFXRateTool(market_data_service),
        GetBatchPricesTool(market_data_service),
        GetDataFreshnessTool(market_data_service),
        RefreshDataTool(market_data_service, portfolio_manager),
    ]
