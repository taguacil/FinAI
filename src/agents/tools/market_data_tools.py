"""
MarketDataService tools for the Analytics Agent.

Exposes market data functionality including price history, FX rates,
batch pricing, and data freshness monitoring.
"""

from datetime import date
from typing import Dict, List, Optional

import pandas as pd
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from ...portfolio.manager import PortfolioManager
from ...portfolio.market_data_store import MarketDataStore
from ...portfolio.models import Currency
from ...services.market_data_service import MarketDataService


class GetPriceHistoryInput(BaseModel):
    """Input for price history lookup."""

    symbol: str = Field(description="Stock/instrument symbol (e.g., AAPL, TSLA)")
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")


class GetPriceHistoryTool(BaseTool):
    """Tool for getting historical price data from stored database."""

    name: str = "get_price_history"
    description: str = """Get historical price data for a symbol over a date range.

    Returns daily close prices from the stored database.
    Useful for analyzing price trends, calculating returns, and technical analysis.

    Example: Get AAPL prices from 2024-01-01 to 2024-06-30
    """
    args_schema: type[BaseModel] = GetPriceHistoryInput
    market_data_store: Optional[MarketDataStore] = None

    def __init__(self, market_data_store: MarketDataStore):
        super().__init__()
        self.market_data_store = market_data_store

    def _run(self, symbol: str, start_date: str, end_date: str) -> str:
        """Get historical prices from stored database."""
        try:
            start = date.fromisoformat(start_date)
            end = date.fromisoformat(end_date)
            symbol = symbol.upper().strip()

            # Get prices from MarketDataStore (stored database)
            raw_prices = self.market_data_store.get_prices(symbol, start, end)

            if not raw_prices:
                return f"❌ No price history found for {symbol} between {start_date} and {end_date}. Run fetch_and_update_prices first."

            # Convert to sorted list
            sorted_dates = sorted(raw_prices.keys())
            prices_list = [(d, float(raw_prices[d])) for d in sorted_dates]

            # Format as readable table
            result = [
                f"📈 **Price History for {symbol}**",
                f"Period: {start_date} to {end_date}",
                f"Data points: {len(prices_list)}",
                "",
                "| Date | Close |",
                "|------|-------|",
            ]

            # Show first 10 and last 5 rows if more than 20 rows
            if len(prices_list) > 20:
                for d, price in prices_list[:10]:
                    result.append(f"| {d.strftime('%Y-%m-%d')} | {price:.2f} |")
                result.append("| ... | ... |")
                for d, price in prices_list[-5:]:
                    result.append(f"| {d.strftime('%Y-%m-%d')} | {price:.2f} |")
            else:
                for d, price in prices_list:
                    result.append(f"| {d.strftime('%Y-%m-%d')} | {price:.2f} |")

            # Add summary statistics
            if len(prices_list) > 1:
                first_close = prices_list[0][1]
                last_close = prices_list[-1][1]
                pct_change = ((last_close - first_close) / first_close) * 100
                high = max(p[1] for p in prices_list)
                low = min(p[1] for p in prices_list)

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


class GetMovingAverageSignalInput(BaseModel):
    """Input for moving average signal calculation."""

    symbol: str = Field(description="Stock/instrument symbol (e.g., AAPL, TSLA)")
    short_period: int = Field(
        default=50,
        description="Short-term moving average period in days (default 50)"
    )
    long_period: int = Field(
        default=200,
        description="Long-term moving average period in days (default 200)"
    )


class GetMovingAverageSignalTool(BaseTool):
    """Tool for calculating moving average crossover signals."""

    name: str = "get_moving_average_signal"
    description: str = """Calculate moving average signal for a symbol.

    Computes short-term MA (default 50-day) and long-term MA (default 200-day),
    then returns the difference (short MA - long MA).

    Signal interpretation:
    - Positive value = BUY signal (Golden Cross territory)
    - Negative value = SELL signal (Death Cross territory)

    Example: Get MA signal for AAPL with 50-day and 200-day MAs
    """
    args_schema: type[BaseModel] = GetMovingAverageSignalInput
    market_data_store: Optional[MarketDataStore] = None

    def __init__(self, market_data_store: MarketDataStore):
        super().__init__()
        self.market_data_store = market_data_store

    def _run(
        self,
        symbol: str,
        short_period: int = 50,
        long_period: int = 200,
    ) -> str:
        """Calculate moving average signal using stored price data."""
        from datetime import timedelta

        try:
            symbol = symbol.upper().strip()

            # Need enough data for the long MA plus buffer
            end = date.today()
            start = end - timedelta(days=int(long_period * 2))

            # Get prices from MarketDataStore (stored database)
            prices_dict: Dict[date, float] = {}
            raw_prices = self.market_data_store.get_prices(symbol, start, end)

            if not raw_prices:
                return f"❌ No price history found for {symbol} in database. Run fetch_and_update_prices first."

            # Convert to sorted list of (date, price)
            for d, price in raw_prices.items():
                prices_dict[d] = float(price)

            # Sort by date and create series
            sorted_dates = sorted(prices_dict.keys())
            closes = pd.Series([prices_dict[d] for d in sorted_dates], index=sorted_dates)

            if len(closes) < long_period:
                return (
                    f"❌ Insufficient data for {symbol}: "
                    f"need {long_period} days, got {len(closes)}. "
                    f"Run fetch_and_update_prices to load more historical data."
                )

            # Calculate moving averages
            short_ma = closes.tail(short_period).mean()
            long_ma = closes.tail(long_period).mean()
            current_price = closes.iloc[-1]

            # Calculate signal
            ma_diff = short_ma - long_ma
            ma_diff_pct = (ma_diff / long_ma) * 100

            # Determine signal
            if ma_diff > 0:
                signal = "🟢 BUY"
                signal_desc = "Golden Cross territory"
            else:
                signal = "🔴 SELL"
                signal_desc = "Death Cross territory"

            # Price position relative to MAs
            price_vs_short = ((current_price - short_ma) / short_ma) * 100
            price_vs_long = ((current_price - long_ma) / long_ma) * 100

            # Price position indicators
            price_above_short = current_price > short_ma
            price_above_long = current_price > long_ma

            short_position = "ABOVE" if price_above_short else "BELOW"
            long_position = "ABOVE" if price_above_long else "BELOW"

            short_icon = "🟢" if price_above_short else "🔴"
            long_icon = "🟢" if price_above_long else "🔴"

            # Trend strength assessment
            if price_above_short and price_above_long and ma_diff > 0:
                trend = "🟢 Strong Uptrend"
            elif price_above_long and ma_diff > 0:
                trend = "🟡 Uptrend (pullback)"
            elif not price_above_short and not price_above_long and ma_diff < 0:
                trend = "🔴 Strong Downtrend"
            elif not price_above_long and ma_diff < 0:
                trend = "🟡 Downtrend (bounce)"
            elif ma_diff > 0:
                trend = "🟡 Bullish but weakening"
            else:
                trend = "🟡 Bearish but stabilizing"

            lines = [
                f"📊 **Moving Average Signal: {symbol}**",
                "",
                f"**MA Crossover Signal: {signal}** ({signal_desc})",
                f"**Trend Assessment: {trend}**",
                "",
                "**Moving Averages:**",
                f"• {short_period}-day MA: {short_ma:.2f}",
                f"• {long_period}-day MA: {long_ma:.2f}",
                f"• Difference (Short - Long): {ma_diff:+.2f} ({ma_diff_pct:+.2f}%)",
                "",
                "**Price Position:**",
                f"• Current Price: {current_price:.2f}",
                f"• {short_icon} {short_position} {short_period}-MA by {abs(price_vs_short):.2f}%",
                f"• {long_icon} {long_position} {long_period}-MA by {abs(price_vs_long):.2f}%",
            ]

            return "\n".join(lines)

        except Exception as e:
            return f"❌ Error calculating MA signal: {str(e)}"


def create_market_data_tools(
    market_data_service: MarketDataService,
    portfolio_manager: PortfolioManager,
) -> List[BaseTool]:
    """Create all MarketDataService tools for the Analytics Agent."""
    market_data_store = portfolio_manager.market_data_store
    return [
        GetPriceHistoryTool(market_data_store),
        GetFXRateTool(market_data_service),
        GetBatchPricesTool(market_data_service),
        GetDataFreshnessTool(market_data_service),
        RefreshDataTool(market_data_service, portfolio_manager),
        GetMovingAverageSignalTool(market_data_store),
    ]
