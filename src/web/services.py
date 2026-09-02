"""Backend glue for the FinAI web app.

Builds the existing portfolio backend once and exposes small, JSON-friendly
data assemblers that the FastAPI routes render. No business logic lives here —
this only adapts the in-process managers to the web layer.
"""

from __future__ import annotations

import json
import math
import os
from datetime import date, datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data_providers.manager import DataProviderManager
from src.portfolio.asset_classes import (
    CATEGORY_BY_VIEW_MODE,
    category_for_instrument_type,
)
from src.portfolio.manager import PortfolioManager
from src.portfolio.models import Currency, TransactionType
from src.portfolio.optimizer import (
    OptimizationMethod,
    OptimizationObjective,
    PortfolioOptimizer,
)
from src.portfolio.simulation_store import SimulationStore
from src.portfolio.storage import FileBasedStorage
from src.services.market_data_service import MarketDataService
from src.utils.metrics import FinancialMetricsCalculator

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR = os.environ.get("FINAI_DATA_DIR") or os.path.join(_PROJECT_ROOT, "data")

# Human-friendly labels for InstrumentType values, used for asset-class allocation.
_ASSET_CLASS_LABELS = {
    "stock": "Equities",
    "etf": "ETFs",
    "bond": "Bonds",
    "crypto": "Crypto",
    "cash": "Cash",
    "mutual_fund": "Mutual Funds",
    "option": "Options",
    "future": "Futures",
    "structured_product": "Structured Products",
}

# Asset-class view modes shared by the Dashboard and Analytics selectors.
ASSET_CLASS_VIEWS = [
    ("all", "All"),
    ("equities_only", "Equities"),
    ("fixed_income_only", "Fixed Income"),
    ("structured_only", "Structured"),
    ("other_only", "Other"),
]

# view_mode -> instrument category and instrument_type -> category both come
# from the canonical taxonomy in src.portfolio.asset_classes (imported above).
_CATEGORY_BY_VIEW_MODE = CATEGORY_BY_VIEW_MODE
_instrument_category = category_for_instrument_type


def _f(value: Any) -> Optional[float]:
    """Coerce Decimal/int/float to float; None stays None."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _num(value: Any) -> Optional[float]:
    """Like _f but also drops NaN/inf so the result is always JSON-safe."""
    v = _f(value)
    if v is None or not math.isfinite(v):
        return None
    return v


def _pct(value: Any) -> Optional[float]:
    """Decimal fraction (0.15) -> percent (15.0), JSON-safe."""
    v = _num(value)
    return v * 100 if v is not None else None


class AppContext:
    """Holds the portfolio backend for the lifetime of the web process."""

    def __init__(self, data_dir: str = _DATA_DIR, offline: bool = True):
        self.data_dir = data_dir
        storage = FileBasedStorage(data_dir)
        data_provider = DataProviderManager()
        # Cache-only by default: page loads must never block on live network
        # fetches (matches the app's "manual price control" design). Live
        # providers are stashed so an explicit refresh action can restore them.
        self._live_providers = list(data_provider.providers)
        if offline:
            data_provider.providers = []
        self.offline = offline
        self.data_provider = data_provider
        market_data_service = MarketDataService(data_provider)
        self.storage = storage
        self.manager = PortfolioManager(storage, market_data_service, data_dir=data_dir)
        # Give the metrics calculator the local price store so benchmark
        # comparison works from cached data even with no live providers.
        self.metrics = FinancialMetricsCalculator(
            data_provider, market_data_store=self.manager.market_data_store
        )
        # Persist web simulation runs so past backtests / projections can be
        # revisited (and re-rendered) without recomputation.
        self.sim_store = SimulationStore(data_dir=data_dir)

    def set_online(self, online: bool) -> None:
        """Toggle live data providers (used by explicit price-refresh actions)."""
        self.data_provider.providers = list(self._live_providers) if online else []
        self.offline = not online

    # -- portfolio selection -------------------------------------------------

    def list_portfolios(self) -> List[Dict[str, str]]:
        """Return [{id, name}] for the portfolio switcher."""
        out: List[Dict[str, str]] = []
        for pid in self.manager.list_portfolios():
            try:
                p = self.storage.load_portfolio(pid)
                out.append({"id": pid, "name": p.name if p else pid})
            except Exception:
                out.append({"id": pid, "name": pid})
        return out

    def ensure_loaded(self, portfolio_id: Optional[str]) -> Optional[str]:
        """Load the requested portfolio (or the first available). Returns active id."""
        available = self.manager.list_portfolios()
        if not available:
            return None
        target = portfolio_id if portfolio_id in available else available[0]
        current = self.manager.current_portfolio
        if not current or current.id != target:
            self.manager.load_portfolio(target)
            self._hydrate_prices_from_store()
        return target

    def _hydrate_prices_from_store(self) -> None:
        """Populate each position's current_price from the local price cache.

        The web app is cache-only, so nothing fetches live prices to fill
        ``position.current_price`` — leaving invested / P&L at zero even when
        the store holds prices. This reads the latest cached price (in the
        instrument's own currency) as a decoupled populate step; it is
        in-memory only and never persisted or fetched over the network.
        """
        portfolio = self.manager.current_portfolio
        store = self.manager.market_data_store
        if not portfolio or store is None:
            return
        for symbol, position in portfolio.positions.items():
            if position.quantity == 0:
                continue
            dps = getattr(position.instrument, "data_provider_symbol", None)
            latest = None
            for candidate in (symbol, dps):
                if not candidate:
                    continue
                try:
                    latest = store.get_latest_price(candidate)
                except Exception:  # noqa: BLE001
                    latest = None
                if latest:
                    break
            if latest and latest[1] is not None:
                position.current_price = latest[1]

    # -- data assemblers -----------------------------------------------------

    def _history_start(self) -> date:
        """Earliest transaction date, or one year ago as a floor."""
        txns = self.manager.get_transaction_history()
        floor = date.today() - timedelta(days=365)
        if not txns:
            return floor
        earliest = min(t["timestamp"].date() for t in txns)
        return min(earliest, floor) if earliest < floor else earliest

    def _ytd_return_pct(self, target_ccy: Optional["Currency"],
                        view_mode: str = "all") -> Optional[float]:
        """Year-to-date time-weighted return (%) in the given display currency.

        Mirrors the base-currency TWR that PortfolioManager.get_ytd_performance
        computes, but honours ``target_currency`` so the figure changes when the
        user switches display currency, and ``view_mode`` so a single-class view
        reports that class's YTD. Returns None if history is insufficient.
        """
        pm = self.manager
        try:
            today = date.today()
            ystart = date(today.year, 1, 1)
            ydf = pm.get_portfolio_history_filtered(
                ystart, today, view_mode=view_mode, target_currency=target_ccy
            )
            if (not isinstance(ydf, pd.DataFrame) or ydf.empty
                    or "total_value" not in ydf or len(ydf) < 2):
                return None
            if view_mode == "all":
                raw = pm.get_external_cash_flows_by_day(
                    ystart, today, target_currency=target_ccy)
            else:
                raw = pm.get_category_cash_flows_by_day(
                    ystart, today, view_mode=view_mode, target_currency=target_ccy)
            yflows = {d: float(v) for d, v in raw.items()}
            yreturns = self.metrics.calculate_returns_from_df(
                ydf, "total_value", yflows
            ) or []
            if not yreturns:
                return None
            twr = 1.0
            for r in yreturns:
                twr *= (1.0 + r)
            return (twr - 1.0) * 100.0
        except Exception:
            return None

    def dashboard(self, view_mode: str = "all") -> Dict[str, Any]:
        """Everything the dashboard page needs.

        ``view_mode`` optionally restricts the KPIs, equity curve, holdings table,
        by-holding allocation and transactions to a single asset class so it can
        be analysed on its own. The by-class allocation and per-class value bands
        always reflect the whole portfolio, to keep the overview available.
        """
        pm = self.manager
        portfolio = pm.current_portfolio
        if not portfolio:
            return {"empty": True}

        if view_mode not in {k for k, _ in ASSET_CLASS_VIEWS}:
            view_mode = "all"
        category = _CATEGORY_BY_VIEW_MODE.get(view_mode)
        views = [{"key": k, "label": lbl} for k, lbl in ASSET_CLASS_VIEWS]

        base_ccy = portfolio.base_currency.value
        positions = pm.get_position_summary()
        invested = sum((p["market_value"] or Decimal(0)) for p in positions)
        unrealized = sum((p["unrealized_pnl"] or Decimal(0)) for p in positions)
        net_worth = pm.get_portfolio_value()
        cash = net_worth - invested

        # cost basis for total return %
        cost_basis = invested - unrealized
        total_return_pct = float(unrealized / cost_basis * 100) if cost_basis else 0.0

        ytd = pm.get_ytd_performance()
        ytd_pct = (ytd.get("portfolio") or {}).get("ytd_pct")

        # equity curve — always show from Jan 1 2024 (or the first transaction,
        # whichever is later, to avoid a long flat-zero prefix).
        floor_2024 = date(2024, 1, 1)
        txns_all = pm.get_transaction_history()
        earliest = min((t["timestamp"].date() for t in txns_all), default=floor_2024)
        start = max(floor_2024, earliest)
        hist = pm.get_portfolio_history(start, date.today())
        curve = {"dates": [], "values": []}
        dates_idx = []
        if isinstance(hist, pd.DataFrame) and not hist.empty and "total_value" in hist:
            dates_idx = list(hist.index)
            curve["dates"] = [d.strftime("%Y-%m-%d") for d in dates_idx]
            curve["values"] = [float(v) for v in hist["total_value"].tolist()]

        # instrument type (asset class) per symbol, from the live position objects
        type_by_symbol = {
            sym: pos.instrument.instrument_type.value
            for sym, pos in portfolio.positions.items()
        }

        # per-asset-class value history (cumulating stack). One band per
        # instrument type present, plus cash — together they sum to net worth.
        class_history = {"dates": curve["dates"], "series": []}
        if dates_idx:
            types_present = sorted({
                pos.instrument.instrument_type.value
                for pos in portfolio.positions.values() if pos.quantity != 0
            })
            # One replay computes every class band; reindex each type column to
            # the master date axis (avoids a full history replay per class).
            by_type = pm.get_value_history_by_type(
                start, date.today(), types=types_present
            )
            if isinstance(by_type, pd.DataFrame) and not by_type.empty:
                by_type = by_type.reindex(dates_idx).fillna(0.0)
                for t in types_present:
                    if t not in by_type.columns:
                        continue
                    vals = [float(v) for v in by_type[t].tolist()]
                    if any(v > 0 for v in vals):
                        class_history["series"].append({
                            "label": _ASSET_CLASS_LABELS.get(t, t.title()),
                            "values": vals,
                        })
            # cash band from the full-history cash column
            if "cash_value" in hist:
                cash_vals = [float(v) for v in hist["cash_value"].reindex(dates_idx).fillna(0.0).tolist()]
                if any(v > 0 for v in cash_vals):
                    class_history["series"].append({"label": "Cash", "values": cash_vals})

        # positions (JSON-safe), sorted by value desc.
        # We surface BOTH the instrument's original currency (price/value as
        # traded) and the base-currency equivalent via the current FX rate.
        pos_rows = []
        for p in positions:
            fx = _f(p["fx_rate"]) or 1.0
            price_base = _f(p["current_price"])
            value_base = _f(p["market_value"])
            pos_rows.append(
                {
                    "symbol": p["symbol"],
                    "name": p["name"],
                    "asset_class": type_by_symbol.get(p["symbol"], "other"),
                    "quantity": _f(p["quantity"]),
                    "avg_cost": _f(p["average_cost"]),
                    # base-currency (portfolio) figures
                    "price": price_base,
                    "value": value_base,
                    "pnl": _f(p["unrealized_pnl"]),
                    "pnl_pct": _f(p["unrealized_pnl_percent"]),
                    # original-currency figures (as traded)
                    "currency": p["original_currency"],
                    "price_local": (price_base / fx) if price_base is not None and fx else price_base,
                    "value_local": (value_base / fx) if value_base is not None and fx else value_base,
                    "fx_rate": fx,
                    "is_fx": p["original_currency"] != base_ccy,
                    "has_price": p["has_current_price"],
                }
            )
        pos_rows.sort(key=lambda r: r["value"] or 0, reverse=True)

        # allocation by asset class (base currency), + residual cash
        class_totals: Dict[str, float] = {}
        for r in pos_rows:
            v = r["value"] or 0
            if v > 0:
                class_totals[r["asset_class"]] = class_totals.get(r["asset_class"], 0.0) + v
        if cash and cash > 0:
            class_totals["cash"] = class_totals.get("cash", 0.0) + float(cash)
        alloc_by_class = sorted(
            ({"label": _ASSET_CLASS_LABELS.get(k, k.title()), "value": v} for k, v in class_totals.items()),
            key=lambda a: a["value"],
            reverse=True,
        )

        # --- asset-class filter -------------------------------------------
        # When a single class is selected, the KPIs, equity curve, holdings
        # table, by-holding allocation and transactions narrow to that class;
        # cash is only shown in the "all" view (it isn't attributed to a class).
        if category is not None:
            view_rows = [r for r in pos_rows
                         if _instrument_category(r["asset_class"]) == category]
            invested_v = sum((r["value"] or 0) for r in view_rows)
            unrealized_v = sum((r["pnl"] or 0) for r in view_rows)
            cost_v = invested_v - unrealized_v
            kpis = {
                "net_worth": invested_v,
                "invested": invested_v,
                "cash": None,
                "unrealized_pnl": unrealized_v,
                "total_return_pct": (unrealized_v / cost_v * 100) if cost_v else 0.0,
                "ytd_pct": self._ytd_return_pct(None, view_mode=view_mode),
            }
            # equity curve = this class's market value over time
            cdf = pm.get_portfolio_history_filtered(start, date.today(), view_mode=view_mode)
            curve_out = {"dates": [], "values": []}
            if isinstance(cdf, pd.DataFrame) and not cdf.empty and "total_value" in cdf:
                curve_out["dates"] = [d.strftime("%Y-%m-%d") for d in cdf.index]
                curve_out["values"] = [float(v) for v in cdf["total_value"].tolist()]
            # Category per symbol for filtering transactions. Transaction records
            # can carry an inconsistent instrument_type for the same symbol (e.g.
            # a dividend booked as "stock" on a bond), so the live position's type
            # is authoritative; fall back to transactions only for sold-out names.
            cat_by_symbol = {
                t.instrument.symbol: _instrument_category(
                    t.instrument.instrument_type.value
                    if hasattr(t.instrument.instrument_type, "value")
                    else str(t.instrument.instrument_type))
                for t in portfolio.transactions
            }
            cat_by_symbol.update(
                {sym: _instrument_category(itype) for sym, itype in type_by_symbol.items()}
            )
            tx_filter = lambda sym: cat_by_symbol.get(sym) == category
        else:
            view_rows = pos_rows
            kpis = {
                "net_worth": _f(net_worth),
                "invested": _f(invested),
                "cash": _f(cash),
                "unrealized_pnl": _f(unrealized),
                "total_return_pct": total_return_pct,
                "ytd_pct": ytd_pct,
            }
            curve_out = curve
            tx_filter = None

        # allocation by holding (base currency), + residual cash (all view only)
        alloc = [{"label": r["symbol"], "value": r["value"] or 0}
                 for r in view_rows if (r["value"] or 0) > 0]
        if category is None and cash and cash > 0:
            alloc.append({"label": "Cash", "value": float(cash)})

        # recent transactions (longer list; the panel scrolls)
        txns = [t for t in pm.get_transaction_history()
                if tx_filter is None or tx_filter(t["symbol"])][:40]
        tx_rows = [
            {
                "date": t["timestamp"].strftime("%Y-%m-%d"),
                "symbol": t["symbol"],
                "type": t["type"],
                "quantity": _f(t["quantity"]),
                "price": _f(t["price"]),
                "total": _f(t["total_value"]),
                "currency": t["currency"],
            }
            for t in txns
        ]

        return {
            "empty": False,
            "base_currency": base_ccy,
            "view_mode": view_mode,
            "views": views,
            "kpis": kpis,
            "curve": curve_out,
            "class_history": class_history,
            "positions": view_rows,
            "allocation": alloc,
            "allocation_by_class": alloc_by_class,
            "transactions": tx_rows,
        }

    # -- analytics -----------------------------------------------------------

    # Asset-class views for the analytics selector; each maps to a
    # get_portfolio_history_filtered view_mode. "All assets" reads nicer here
    # than the shared "All" label.
    ANALYTICS_VIEWS = [(k, "All assets" if k == "all" else lbl)
                       for k, lbl in ASSET_CLASS_VIEWS]

    def analytics(self, days: int = 365, benchmark: str = "SPY",
                  start_date: Optional[date] = None, end_date: Optional[date] = None,
                  currency: Optional[str] = None,
                  view_mode: str = "all") -> Dict[str, Any]:
        """Performance, risk and (when data allows) benchmark analytics.

        An explicit start_date/end_date pair (from the calendar) takes
        precedence over the `days` preset. `currency` denominates the whole
        analysis in a chosen currency (defaults to the portfolio base
        currency); conversion uses cached historical FX rates. `view_mode`
        restricts the analysis to one asset class (with attributed cash) so a
        class can be analysed separately.
        """
        pm = self.manager
        portfolio = pm.current_portfolio
        if not portfolio:
            return {"empty": True}

        base_ccy = portfolio.base_currency.value
        # display currency (falls back to base); only override when it differs
        try:
            display_ccy = Currency(currency) if currency else portfolio.base_currency
        except ValueError:
            display_ccy = portfolio.base_currency
        target_ccy = display_ccy if display_ccy != portfolio.base_currency else None

        end = end_date or date.today()
        if start_date:
            start = start_date
        elif days and days > 0:
            start = end - timedelta(days=days)
        else:
            start = self._history_start()

        all_currencies = [c.value for c in Currency]
        valid_view_modes = {m for m, _ in self.ANALYTICS_VIEWS}
        if view_mode not in valid_view_modes:
            view_mode = "all"
        views = [{"key": k, "label": lbl} for k, lbl in self.ANALYTICS_VIEWS]

        try:
            df = pm.get_portfolio_history_filtered(
                start, end, view_mode=view_mode, target_currency=target_ccy
            )
        except Exception:
            df = pd.DataFrame()

        if not isinstance(df, pd.DataFrame) or df.empty or "total_value" not in df:
            return {"empty": False, "base_currency": base_ccy,
                    "display_currency": display_ccy.value, "currencies": all_currencies,
                    "no_history": True, "view_mode": view_mode, "views": views,
                    "period": {"days": days, "start": start.isoformat(),
                               "end": end.isoformat()}}

        # Cash flows for time-weighted returns. The "all" view uses external
        # deposits/withdrawals; a single-class view uses that class's buys/sells
        # (in the display currency) so the return reflects price/coupon
        # performance rather than capital deployed into the class.
        flows: Dict[date, float] = {}
        try:
            if view_mode == "all":
                raw = pm.get_external_cash_flows_by_day(start, end, target_currency=target_ccy)
            else:
                raw = pm.get_category_cash_flows_by_day(
                    start, end, view_mode=view_mode, target_currency=target_ccy)
            flows = {d: float(v) for d, v in raw.items()}
        except Exception:
            flows = {}

        returns = self.metrics.calculate_returns_from_df(df, "total_value", flows) or []
        try:
            m = self.metrics.calculate_metrics_from_df(df, "total_value", benchmark, flows) or {}
        except Exception:
            m = {}
        if m.get("error"):
            m = {}

        dates = [d.strftime("%Y-%m-%d") for d in df.index]
        values = [float(v) for v in df["total_value"].tolist()]

        # cumulative portfolio return (%) from the daily TWR series
        port_cum, acc = [], 1.0
        for r in returns:
            acc *= (1.0 + r)
            port_cum.append((acc - 1.0) * 100)
        # calculate_returns_from_df emits a return only for days where the prior
        # value was > 0, so for a class acquired partway through the window the
        # return series is shorter than df and does NOT map onto dates[1:].
        # Rebuild the exact positional indices it used so the cumulative curve
        # (and the benchmark below) line up with the correct dates.
        raw_vals = df["total_value"].tolist()
        ret_idx = [i for i in range(1, len(raw_vals)) if float(raw_vals[i - 1]) > 0]
        n = min(len(port_cum), len(ret_idx))
        port_cum = port_cum[:n]
        cum_dates = [dates[i] for i in ret_idx[:n]]

        # benchmark cumulative return (%), aligned to the portfolio dates.
        # The curve renders whenever we have stored benchmark prices; the
        # comparison metrics (beta/alpha/…) require >=2 aligned points.
        bench_available = bool(m.get("benchmark_available"))
        bench_cum = None
        if m.get("benchmark_prices"):
            try:
                s = pd.Series(m["benchmark_prices"])
                s.index = pd.to_datetime(list(s.index))
                s = s.sort_index()
                last_real = s.index.max()
                # ffill fills weekends/holidays within the covered range, but we
                # must NOT fabricate a flat tail past the last real observation
                # (that produced a misleading straight benchmark line).
                s = s.reindex(df.index, method="ffill").where(df.index <= last_real)
                clean = s.dropna()
                base0 = float(clean.iloc[0]) if not clean.empty else None
                if base0:
                    full = [((float(v) / base0) - 1.0) * 100 if pd.notna(v) else None for v in s]
                    # Align the benchmark curve to the same return dates as the
                    # portfolio curve (see ret_idx above) so both share an x-axis.
                    bench_cum = [full[i] for i in ret_idx[:n]]
            except Exception:
                bench_cum = None

        # returns distribution (%)
        rp = [r * 100 for r in returns]
        dist = {
            "returns": [round(x, 4) for x in rp],
            "best": max(rp) if rp else None,
            "worst": min(rp) if rp else None,
            "avg": (sum(rp) / len(rp)) if rp else None,
            "positive_days": sum(1 for x in rp if x > 0),
            "total_days": len(rp),
        }

        # YTD in the *display* currency. pm.get_ytd_performance() only computes
        # it in the base currency, so it would report an identical number in
        # every currency view; recompute the TWR from a currency-aware history.
        ytd_pct = self._ytd_return_pct(target_ccy, view_mode=view_mode)

        metrics = {
            "total_return": _pct(m.get("total_return")),
            "annualized_return": _pct(m.get("annualized_return")),
            "volatility": _pct(m.get("volatility")),
            "sharpe": _num(m.get("sharpe_ratio")),
            "sortino": _num(m.get("sortino_ratio")),
            "max_drawdown": _pct(m.get("max_drawdown")),
            "max_dd_duration": m.get("max_drawdown_duration"),
            "calmar": _num(m.get("calmar_ratio")),
            "var_5pct": _pct(m.get("var_5pct")),
            "cvar_5pct": _pct(m.get("cvar_5pct")),
            "ytd_pct": _num(ytd_pct),
            "days_analyzed": m.get("days_analyzed"),
        }
        bench_metrics = None
        if bench_available:
            bench_metrics = {
                "symbol": benchmark,
                "beta": _num(m.get("beta")),
                "alpha": _pct(m.get("alpha")),
                "information_ratio": _num(m.get("information_ratio")),
                "benchmark_return": _pct(m.get("benchmark_return")),
                "benchmark_volatility": _pct(m.get("benchmark_volatility")),
            }

        return {
            "empty": False,
            "no_history": False,
            "base_currency": base_ccy,
            "display_currency": display_ccy.value,
            "currencies": all_currencies,
            "view_mode": view_mode,
            "views": views,
            "benchmark_symbol": benchmark,
            "offline": self.offline,
            "period": {"days": days, "start": start.isoformat(), "end": end.isoformat()},
            "curve": {"dates": dates, "values": values},
            "cumulative": {"dates": cum_dates, "portfolio": port_cum, "benchmark": bench_cum},
            "distribution": dist,
            "metrics": metrics,
            "benchmark": bench_metrics,
        }


    # -- optimize ------------------------------------------------------------

    def optimize(
        self,
        run: bool,
        lookback_days: int = 252,
        risk_free_rate: float = 0.04,
        objective: str = "max_sharpe",
        include_cash: bool = True,
        scope: str = "all",
        selected_symbols: Optional[List[str]] = None,
        candidate_symbols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """HRP & Markowitz target weights, risk/return picture, rebalancing trades.

        scope controls which held positions are optimized:
          - "all":  optimize every tradable holding (default).
          - "lock": keep `selected_symbols` at current weight, optimize the rest.
          - "only": optimize only `selected_symbols` (their combined weight is
                    held fixed); everything else is kept at current weight.
        candidate_symbols are instruments not currently held (drawn from the
        locally stored universe) that the optimizer may allocate to (BUY).
        """
        pm = self.manager
        portfolio = pm.current_portfolio
        if not portfolio:
            return {"empty": True}

        base_ccy = portfolio.base_currency.value
        selected_symbols = [s.upper() for s in (selected_symbols or [])]
        candidate_symbols = [s.upper() for s in (candidate_symbols or [])]

        # Pickers for the form: held symbols, and the stored universe not held.
        held_symbols = [s for s, p in portfolio.positions.items() if p.quantity != 0]
        try:
            universe = set(pm.market_data_store.get_symbols())
        except Exception:
            universe = set()
        candidate_universe = sorted(universe - set(held_symbols))

        params = {
            "lookback_days": lookback_days,
            "risk_free_rate": risk_free_rate,
            "objective": objective,
            "include_cash": include_cash,
            "scope": scope,
            "selected_symbols": selected_symbols,
            "candidate_symbols": candidate_symbols,
            "holdings": sorted(held_symbols),
            "candidate_universe": candidate_universe,
        }
        if not run:
            return {"empty": False, "ran": False, "base_currency": base_ccy, "params": params}

        positions = {s: p for s, p in portfolio.positions.items() if p.quantity != 0}
        if not positions:
            return {"empty": False, "ran": True, "base_currency": base_ccy, "params": params,
                    "error": "No positions to optimize."}

        # Translate the scope choice into the optimizer's lock list.
        if scope == "lock":
            locked_symbols = [s for s in selected_symbols if s in positions] or None
        elif scope == "only" and selected_symbols:
            locked_symbols = [s for s in positions if s not in selected_symbols] or None
        else:
            locked_symbols = None

        total_value = Decimal(0)
        for p in positions.values():
            mv = p.market_value if p.market_value else (p.quantity * p.average_cost)
            total_value += mv or Decimal(0)

        obj = (OptimizationObjective.MIN_VOLATILITY
               if objective == "min_volatility" else OptimizationObjective.MAX_SHARPE)
        cash_balances = portfolio.cash_balances if include_cash else None

        try:
            optimizer = PortfolioOptimizer(
                pm.data_manager,
                base_currency=portfolio.base_currency,
                storage=pm.storage,
                portfolio_id=portfolio.id,
            )
            results = optimizer.compare_methods(
                positions=positions,
                locked_symbols=locked_symbols,
                lookback_days=lookback_days,
                risk_free_rate=risk_free_rate,
                total_portfolio_value=total_value,
                cash_balances=cash_balances,
                objective=obj,
                include_cash=include_cash,
                candidate_symbols=candidate_symbols or None,
            )
        except Exception as exc:  # noqa: BLE001 — surface any optimizer failure to the UI
            return {"empty": False, "ran": True, "base_currency": base_ccy, "params": params,
                    "error": f"Optimization failed: {exc}"}

        hrp = results.get(OptimizationMethod.HRP)
        mk = results.get(OptimizationMethod.MARKOWITZ)
        if not hrp or not mk:
            return {"empty": False, "ran": True, "base_currency": base_ccy, "params": params,
                    "error": "Optimizer returned no result."}

        current_w = hrp.current_weights or {}
        symbols = sorted(
            set(current_w) | set(hrp.weights or {}) | set(mk.weights or {}),
            key=lambda s: max((hrp.weights or {}).get(s, 0), (mk.weights or {}).get(s, 0)),
            reverse=True,
        )
        weight_rows = [
            {
                "symbol": s,
                "current": (current_w.get(s, 0.0)) * 100,
                "hrp": (hrp.weights or {}).get(s, 0.0) * 100,
                "markowitz": (mk.weights or {}).get(s, 0.0) * 100,
            }
            for s in symbols
        ]
        cur_cash = max(0.0, 1.0 - sum((current_w or {}).values()))
        weight_rows.append({
            "symbol": "Cash", "current": cur_cash * 100,
            "hrp": (hrp.cash_weight or 0.0) * 100, "markowitz": (mk.cash_weight or 0.0) * 100,
            "is_cash": True,
        })

        def method_metrics(r) -> Dict[str, Any]:
            return {
                "expected_return": _pct(r.expected_annual_return),
                "volatility": _pct(r.annual_volatility),
                "sharpe": _num(r.sharpe_ratio),
                "cash_weight": _pct(r.cash_weight),
                "warnings": list(r.warnings or []),
            }

        assets = [
            {"label": am.symbol, "x": _num(am.volatility * 100), "y": _num(am.expected_return * 100),
             "weight": _num(am.current_weight * 100)}
            for am in (hrp.asset_metrics or [])
            if am.volatility is not None and am.expected_return is not None
        ]
        portfolios = []
        for name, r in (("HRP", hrp), ("Markowitz", mk)):
            x, y = _num((r.annual_volatility or 0) * 100), _num((r.expected_annual_return or 0) * 100)
            if x is not None and y is not None:
                portfolios.append({"name": name, "x": x, "y": y})

        trades = [
            {
                "symbol": t.symbol, "action": t.action, "shares": _num(t.shares),
                "value_base": _num(t.estimated_value), "value_native": _num(t.estimated_value_native),
                "currency": t.currency, "current_pct": _num(t.current_weight * 100),
                "target_pct": _num(t.target_weight * 100),
            }
            for t in (hrp.rebalancing_trades or [])
        ]
        buy_total = sum(t["value_base"] for t in trades if t["action"] == "BUY" and t["value_base"])
        sell_total = sum(t["value_base"] for t in trades if t["action"] == "SELL" and t["value_base"])

        return {
            "empty": False,
            "ran": True,
            "base_currency": base_ccy,
            "params": params,
            "total_value": _num(total_value),
            "weights": weight_rows,
            "metrics": {"hrp": method_metrics(hrp), "markowitz": method_metrics(mk)},
            "scatter": {"assets": assets, "portfolios": portfolios,
                        "risk_free": risk_free_rate * 100},
            "trades": trades,
            "trade_summary": {"buy": buy_total, "sell": sell_total, "net": buy_total - sell_total},
        }


    # -- simulate: backtest --------------------------------------------------

    def default_universe(self) -> List[str]:
        """Portfolio symbols (provider symbol where set) for the backtest form."""
        pm = self.manager
        portfolio = pm.current_portfolio
        if not portfolio:
            return []
        out = []
        for sym, pos in portfolio.positions.items():
            if pos.quantity == 0:
                continue
            dps = getattr(pos.instrument, "data_provider_symbol", None)
            out.append(dps or sym)
        return out

    def backtest(
        self,
        run: bool,
        symbols: List[str],
        start: date,
        end: date,
        initial_capital: float = 100_000.0,
        rebalance_frequency: str = "monthly",
        strategy_specs: Optional[List[str]] = None,
        benchmark_symbol: str = "SPY",
        lookback_days: int = 252,
        transaction_cost_bps: float = 0.0,
        risk_free_rate: float = 0.04,
    ) -> Dict[str, Any]:
        pm = self.manager
        portfolio = pm.current_portfolio
        if not portfolio:
            return {"empty": True}
        base_ccy = portfolio.base_currency.value
        strategy_specs = strategy_specs or ["hrp", "equal_weight", "buy_and_hold"]
        # Available holdings (symbol + name) and strategies drive the pickers.
        available_universe = []
        for sym, pos in portfolio.positions.items():
            if pos.quantity == 0:
                continue
            dps = getattr(pos.instrument, "data_provider_symbol", None) or sym
            available_universe.append({"symbol": dps, "name": pos.instrument.name or sym})
        available_universe.sort(key=lambda a: a["symbol"])
        params = {
            "symbols": symbols, "start": start.isoformat(), "end": end.isoformat(),
            "initial_capital": initial_capital, "rebalance_frequency": rebalance_frequency,
            "strategy_specs": strategy_specs, "benchmark_symbol": benchmark_symbol,
            "lookback_days": lookback_days, "transaction_cost_bps": transaction_cost_bps,
            "risk_free_rate": risk_free_rate,
            "available_universe": available_universe,
            "available_strategies": ["hrp", "max_sharpe", "min_volatility",
                                     "equal_weight", "buy_and_hold"],
        }
        if not run:
            return {"empty": False, "ran": False, "base_currency": base_ccy, "params": params}
        if not symbols:
            return {"empty": False, "ran": True, "base_currency": base_ccy, "params": params,
                    "error": "Enter at least one symbol."}

        try:
            from src.portfolio.backtester import Backtester, BacktestConfig
            from src.portfolio.strategies import build_strategies

            config = BacktestConfig(
                symbols=symbols, start_date=start, end_date=end,
                initial_capital=float(initial_capital),
                rebalance_frequency=rebalance_frequency,
                transaction_cost_bps=float(transaction_cost_bps),
                benchmark_symbol=benchmark_symbol,
                base_currency=portfolio.base_currency,
                risk_free_rate=float(risk_free_rate),
                lookback_days=int(lookback_days),
            )
            strategies = build_strategies(strategy_specs, lookback_days=lookback_days,
                                          risk_free_rate=risk_free_rate)
            # Currency awareness (best-effort): map known symbols to their currency.
            price_currencies = {}
            for sym, pos in portfolio.positions.items():
                price_currencies[sym.upper()] = pos.instrument.currency
            fx_rate_on = getattr(pm.data_manager, "get_historical_fx_rate_on", None)
            backtester = Backtester(
                market_data_store=pm.market_data_store,
                metrics_calculator=self.metrics,
                fx_rate_on=fx_rate_on,
                price_currencies=price_currencies or None,
            )
            result = backtester.run(config, strategies)
        except Exception as exc:  # noqa: BLE001
            return {"empty": False, "ran": True, "base_currency": base_ccy, "params": params,
                    "error": f"Backtest failed: {exc}"}

        # master date axis from the first strategy (all share the same grid)
        master_dates: List[str] = []
        series, metrics_rows, dd_series = [], [], []
        for s in result.strategies:
            ec = s.equity_curve
            if not isinstance(ec, pd.DataFrame) or ec.empty or "total_value" not in ec:
                continue
            dts = [d.strftime("%Y-%m-%d") for d in ec.index]
            vals = [float(v) for v in ec["total_value"].tolist()]
            if not master_dates:
                master_dates = dts
            series.append({"name": s.name, "values": vals})
            # drawdown (%)
            peak, dd = float("-inf"), []
            for v in vals:
                peak = max(peak, v)
                dd.append(((v - peak) / peak * 100) if peak > 0 else 0.0)
            dd_series.append({"name": s.name, "values": dd})
            m = s.metrics or {}
            metrics_rows.append({
                "name": s.name,
                "final_value": _num(s.final_value),
                "total_return": _pct(m.get("total_return")),
                "cagr": _pct(m.get("annualized_return")),
                "volatility": _pct(m.get("volatility")),
                "sharpe": _num(m.get("sharpe_ratio")),
                "sortino": _num(m.get("sortino_ratio")),
                "max_drawdown": _pct(m.get("max_drawdown")),
            })

        benchmark = None
        if isinstance(result.benchmark_curve, pd.DataFrame) and not result.benchmark_curve.empty:
            bc = result.benchmark_curve
            try:
                aligned = bc["total_value"].reindex(
                    pd.to_datetime(master_dates), method="ffill"
                ) if master_dates else bc["total_value"]
                benchmark = {"name": benchmark_symbol,
                             "values": [(_num(v)) for v in aligned.tolist()]}
            except Exception:
                benchmark = None

        out = {
            "empty": False, "ran": True, "base_currency": base_ccy, "params": params,
            "window": {"start": result.price_start.isoformat() if result.price_start else None,
                       "end": result.price_end.isoformat() if result.price_end else None},
            "warnings": list(result.warnings or []),
            "equity": {"dates": master_dates, "series": series, "benchmark": benchmark},
            "drawdown": {"dates": master_dates, "series": dd_series},
            "metrics": metrics_rows,
        }
        best = max(metrics_rows, key=lambda m: (m.get("total_return") or float("-inf")),
                   default=None)
        headline = (f"{best['name']} {best['total_return']:+.1f}%"
                    if best and best.get("total_return") is not None else "—")
        self._save_simulation(
            kind="backtest",
            label=f"{len(series)} strategies · {', '.join(symbols)}",
            headline=headline, params=params, result=out,
        )
        return out

    # -- simulate: Monte Carlo ----------------------------------------------

    # Strategies whose expected return / volatility can drive a Monte Carlo run.
    _MC_STRATEGIES = {
        "max_sharpe": ("Max Sharpe", OptimizationObjective.MAX_SHARPE, OptimizationMethod.MARKOWITZ),
        "min_volatility": ("Min Volatility", OptimizationObjective.MIN_VOLATILITY, OptimizationMethod.MARKOWITZ),
        "hrp": ("HRP", OptimizationObjective.MAX_SHARPE, OptimizationMethod.HRP),
    }

    @staticmethod
    def mc_strategies() -> List[Dict[str, str]]:
        return [{"key": k, "label": v[0]} for k, v in AppContext._MC_STRATEGIES.items()]

    def _strategy_assumptions(self, strategy: str, lookback_days: int,
                              risk_free_rate: float) -> Dict[str, float]:
        """Expected annual return / volatility for a strategy, via the optimizer."""
        pm = self.manager
        portfolio = pm.current_portfolio
        label, objective, method = self._MC_STRATEGIES[strategy]
        positions = {s: p for s, p in portfolio.positions.items() if p.quantity != 0}
        if len(positions) < 2:
            raise ValueError("Need at least 2 tradable positions to derive a strategy.")
        total_value = Decimal(0)
        for p in positions.values():
            mv = p.market_value if p.market_value else (p.quantity * p.average_cost)
            total_value += mv or Decimal(0)
        optimizer = PortfolioOptimizer(
            pm.data_manager, base_currency=portfolio.base_currency,
            storage=pm.storage, portfolio_id=portfolio.id,
        )
        results = optimizer.compare_methods(
            positions=positions, lookback_days=lookback_days,
            risk_free_rate=risk_free_rate, total_portfolio_value=total_value,
            cash_balances=portfolio.cash_balances, objective=objective, include_cash=True,
        )
        r = results.get(method)
        if not r or r.expected_annual_return is None or r.annual_volatility is None:
            raise ValueError(f"Optimizer produced no {label} result for this window.")
        return {"label": label, "mu": float(r.expected_annual_return),
                "sigma": float(r.annual_volatility), "sharpe": _f(r.sharpe_ratio)}

    def monte_carlo(
        self,
        run: bool,
        scenario: str = "likely",
        projection_years: float = 5.0,
        monte_carlo_runs: int = 1000,
        monthly_deposit: float = 0.0,
        monthly_withdrawal: float = 0.0,
        strategy: Optional[str] = None,
        lookback_days: int = 252,
        risk_free_rate: float = 0.04,
    ) -> Dict[str, Any]:
        pm = self.manager
        portfolio = pm.current_portfolio
        if not portfolio:
            return {"empty": True}
        base_ccy = portfolio.base_currency.value
        strategy = strategy if strategy in self._MC_STRATEGIES else None
        params = {
            "scenario": scenario, "projection_years": projection_years,
            "monte_carlo_runs": monte_carlo_runs,
            "monthly_deposit": monthly_deposit, "monthly_withdrawal": monthly_withdrawal,
            "strategy": strategy, "lookback_days": lookback_days,
            "risk_free_rate": risk_free_rate,
        }
        if not run:
            return {"empty": False, "ran": False, "base_currency": base_ccy, "params": params,
                    "strategies": self.mc_strategies()}

        try:
            from src.portfolio.scenarios import (
                MarketAssumptions, PortfolioScenarioEngine,
                ScenarioConfiguration, ScenarioType,
            )

            value = pm.get_portfolio_value()
            if not value or value <= 0:
                return {"empty": False, "ran": True, "base_currency": base_ccy, "params": params,
                        "strategies": self.mc_strategies(),
                        "error": "Portfolio has no value to project."}
            engine = PortfolioScenarioEngine(random_seed=42)
            snapshot = pm.create_current_snapshot()

            basis = None
            if strategy:
                # Drive the projection from a strategy's optimized return/vol.
                # (The GBM engine is single-asset, so only mu/sigma matter.)
                a = self._strategy_assumptions(strategy, int(lookback_days), float(risk_free_rate))
                config = ScenarioConfiguration(
                    scenario_type=ScenarioType.CUSTOM,
                    name=f"{a['label']} strategy",
                    description=f"Monte Carlo driven by the {a['label']} optimized portfolio.",
                    market_assumptions=MarketAssumptions(
                        expected_return=a["mu"], volatility=a["sigma"],
                        risk_free_rate=float(risk_free_rate),
                    ),
                )
                basis = {"kind": "strategy", "label": a["label"],
                         "mu": a["mu"], "sigma": a["sigma"], "sharpe": a["sharpe"]}
            else:
                scenarios = engine.create_predefined_scenarios(float(value))
                config = scenarios.get(scenario) or scenarios.get("likely")
                ma = config.market_assumptions
                basis = {"kind": "scenario", "label": scenario.capitalize(),
                         "mu": float(ma.expected_return), "sigma": float(ma.volatility),
                         "sharpe": None}
            config.projection_years = float(projection_years)
            config.monte_carlo_runs = int(monte_carlo_runs)
            config.recurring_deposits = float(monthly_deposit)
            config.recurring_withdrawals = float(monthly_withdrawal)
            config.confidence_intervals = [0.05, 0.25, 0.5, 0.75, 0.95]
            result = engine.run_scenario_simulation(snapshot, config)
        except Exception as exc:  # noqa: BLE001
            return {"empty": False, "ran": True, "base_currency": base_ccy, "params": params,
                    "strategies": self.mc_strategies(),
                    "error": f"Simulation failed: {exc}"}

        def clean_list(xs):
            return [(_num(x)) for x in (xs or [])]

        pcts = result.percentiles or {}
        dates = [d.isoformat() for d in (result.dates or [])]
        bands = {
            "p5": clean_list(pcts.get(0.05)),
            "p25": clean_list(pcts.get(0.25)),
            "p50": clean_list(pcts.get(0.5) or result.mean_trajectory),
            "p75": clean_list(pcts.get(0.75)),
            "p95": clean_list(pcts.get(0.95)),
        }
        stats = result.get_summary_stats() or {}
        summary = {k: _num(v) for k, v in stats.items()}
        summary["probability_of_loss"] = _num(result.probability_of_loss)
        summary["probability_of_doubling"] = _num(result.probability_of_doubling)

        out = {
            "empty": False, "ran": True, "base_currency": base_ccy, "params": params,
            "strategies": self.mc_strategies(),
            "basis": basis,
            "start_value": _num(result.start_value),
            "dates": dates,
            "bands": bands,
            "final_values": clean_list(result.final_values),
            "summary": summary,
        }
        median = summary.get("median_final_value")
        headline = (f"median {median:,.0f} {base_ccy}"
                    if median is not None else "—")
        self._save_simulation(
            kind="montecarlo",
            label=f"{basis['label']} · {projection_years:g}y · {monte_carlo_runs} runs"
            if basis else f"{projection_years:g}y · {monte_carlo_runs} runs",
            headline=headline, params=params, result=out, random_seed=42,
        )
        return out

    # -- simulate: history --------------------------------------------------

    _SIM_TOOLS = {"backtest": "web_backtest", "montecarlo": "web_monte_carlo"}

    def _save_simulation(self, kind: str, label: str, headline: str,
                         params: Dict[str, Any], result: Dict[str, Any],
                         random_seed: Optional[int] = None) -> None:
        """Persist a web simulation run (best-effort; never breaks a page load)."""
        portfolio = self.manager.current_portfolio
        try:
            self.sim_store.save(
                tool=self._SIM_TOOLS.get(kind, kind),
                inputs={"kind": kind, "label": label, "headline": headline, "params": params},
                output=json.dumps(result, default=str),
                portfolio_id=portfolio.id if portfolio else None,
                portfolio_name=portfolio.name if portfolio else None,
                random_seed=random_seed,
            )
        except Exception:  # noqa: BLE001 — persistence must not break the run
            pass

    def simulation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Recent web simulation runs (newest first) for the current portfolio."""
        portfolio = self.manager.current_portfolio
        pid = portfolio.id if portfolio else None
        rows: List[Dict[str, Any]] = []
        for rec in self.sim_store.list(portfolio_id=pid, limit=limit):
            inp = rec.get("inputs") or {}
            if inp.get("kind") not in self._SIM_TOOLS:
                continue
            rows.append({
                "id": rec.get("id"),
                "created_at": rec.get("created_at"),
                "kind": inp.get("kind"),
                "label": inp.get("label") or "",
                "headline": inp.get("headline") or "",
            })
        return rows

    def get_simulation(self, sim_id: str) -> Optional[Dict[str, Any]]:
        """Load a saved run and parse its structured result for re-rendering."""
        rec = self.sim_store.get(sim_id)
        if not rec:
            return None
        inp = rec.get("inputs") or {}
        kind = inp.get("kind")
        if kind not in self._SIM_TOOLS:
            return None
        try:
            result = json.loads(rec.get("output") or "null")
        except (ValueError, TypeError):
            result = None
        return {
            "id": rec.get("id"),
            "created_at": rec.get("created_at"),
            "kind": kind,
            "label": inp.get("label") or "",
            "headline": inp.get("headline") or "",
            "result": result,
        }

    def delete_simulation(self, sim_id: str) -> Dict[str, Any]:
        """Delete a saved run."""
        ok = self.sim_store.delete(sim_id)
        return {"ok": ok, "error": None if ok else "Run not found."}


    # -- data & settings -----------------------------------------------------

    def data_page(self) -> Dict[str, Any]:
        """Market-data overview, per-instrument prices and app/portfolio settings."""
        pm = self.manager
        portfolio = pm.current_portfolio
        if not portfolio:
            return {"empty": True}
        base_ccy = portfolio.base_currency.value

        store = pm.market_data_store
        try:
            price_count = store.get_price_count()
        except Exception:
            price_count = None
        try:
            symbols_tracked = len(store.get_symbols())
        except Exception:
            symbols_tracked = None

        # Map each position to the symbol its prices are stored under, so we can
        # report the date of the *actual* most recent stored price (a far more
        # honest "freshness" signal than a wall-clock last-refresh timestamp).
        dps_by_symbol = {
            sym: (getattr(pos.instrument, "data_provider_symbol", None) or sym)
            for sym, pos in portfolio.positions.items()
        }

        def _latest_price_date(symbol: str) -> Optional[date]:
            for candidate in {symbol, dps_by_symbol.get(symbol, symbol)}:
                try:
                    latest = store.get_latest_price(candidate)
                except Exception:
                    latest = None
                if latest:
                    return latest[0]
            return None

        rows = []
        latest_overall: Optional[date] = None
        for p in pm.get_position_summary():
            fx = _f(p.get("fx_rate")) or 1.0
            price_base = _f(p.get("current_price"))
            last_date = _latest_price_date(p["symbol"])
            if last_date and (latest_overall is None or last_date > latest_overall):
                latest_overall = last_date
            rows.append({
                "symbol": p["symbol"],
                "name": p["name"],
                "currency": p.get("original_currency"),
                "price_local": (price_base / fx) if price_base is not None and fx else price_base,
                "price_base": price_base,
                "is_fx": p.get("original_currency") != base_ccy,
                "has_price": p.get("has_current_price"),
                "last_price_date": last_date.isoformat() if last_date else None,
            })
        rows.sort(key=lambda r: (r["has_price"], r["symbol"]))

        # "Data as of" = newest stored price across the book; stale if we have
        # nothing within the last week (calendar-based, not refresh-based).
        data_as_of = latest_overall.isoformat() if latest_overall else None
        is_stale = (latest_overall is None) or (
            (date.today() - latest_overall).days > 7
        )

        return {
            "empty": False,
            "base_currency": base_ccy,
            "portfolio_name": portfolio.name,
            "portfolio_id": portfolio.id,
            "created_at": portfolio.created_at.isoformat() if getattr(portfolio, "created_at", None) else None,
            "offline": self.offline,
            "stats": {
                "net_worth": _num(pm.get_portfolio_value()),
                "positions": len(rows),
                "price_count": price_count,
                "symbols_tracked": symbols_tracked,
                "data_as_of": data_as_of,
                "is_stale": is_stale,
            },
            "prices": rows,
            "currencies": [c.value for c in Currency],
            "transactions": self.transactions(),
            "transaction_types": self.transaction_types(),
        }

    def set_price(self, symbol: str, price: float, currency: Optional[str] = None) -> Dict[str, Any]:
        """Manually set today's price for a held instrument."""
        pm = self.manager
        if not pm.current_portfolio:
            return {"ok": False, "error": "No portfolio loaded."}
        try:
            ccy = Currency(currency) if currency else None
            ok = pm.set_position_price(symbol, Decimal(str(price)), currency=ccy)
            return {"ok": bool(ok), "error": None if ok else f"Could not set price for {symbol}."}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    # -- transactions --------------------------------------------------------

    def transactions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Transaction list (most recent first) for the CRUD table."""
        pm = self.manager
        if not pm.current_portfolio:
            return []
        rows = pm.get_transaction_history()
        out = []
        for t in rows:
            ts = t["timestamp"]
            out.append({
                "id": t["id"],
                "date": ts.date().isoformat() if hasattr(ts, "date") else str(ts),
                "symbol": t["symbol"],
                "name": t["name"],
                "type": t["type"],
                "quantity": float(t["quantity"]),
                "price": float(t["price"]),
                "total_value": float(t["total_value"]),
                "currency": t["currency"],
                "notes": t["notes"] or "",
            })
        return out[:limit] if limit else out

    @staticmethod
    def transaction_types() -> List[str]:
        return [t.value for t in TransactionType]

    def add_transaction(self, symbol: str, txn_type: str, quantity: float,
                        price: float, date_str: Optional[str] = None,
                        currency: Optional[str] = None,
                        notes: Optional[str] = None) -> Dict[str, Any]:
        """Add a transaction from the web form."""
        pm = self.manager
        if not pm.current_portfolio:
            return {"ok": False, "error": "No portfolio loaded."}
        try:
            tt = TransactionType(txn_type.lower())
            ts = datetime.strptime(date_str, "%Y-%m-%d") if date_str else None
            ccy = Currency(currency) if currency else None
            ok = pm.add_transaction(
                symbol=symbol.upper().strip(), transaction_type=tt,
                quantity=Decimal(str(quantity)), price=Decimal(str(price)),
                timestamp=ts, currency=ccy, notes=notes or None,
            )
            return {"ok": bool(ok), "error": None if ok else "Could not add transaction."}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def modify_transaction(self, transaction_id: str, quantity: Optional[float] = None,
                          price: Optional[float] = None, date_str: Optional[str] = None,
                          notes: Optional[str] = None) -> Dict[str, Any]:
        """Edit an existing transaction from the web form."""
        pm = self.manager
        if not pm.current_portfolio:
            return {"ok": False, "error": "No portfolio loaded."}
        try:
            ts = datetime.strptime(date_str, "%Y-%m-%d") if date_str else None
            ok = pm.modify_transaction(
                transaction_id,
                quantity=Decimal(str(quantity)) if quantity is not None else None,
                price=Decimal(str(price)) if price is not None else None,
                timestamp=ts,
                notes=notes,
            )
            return {"ok": bool(ok), "error": None if ok else "Transaction not found."}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def delete_transaction(self, transaction_id: str) -> Dict[str, Any]:
        """Delete a transaction from the web form."""
        pm = self.manager
        if not pm.current_portfolio:
            return {"ok": False, "error": "No portfolio loaded."}
        try:
            ok = pm.delete_transaction(transaction_id)
            return {"ok": bool(ok), "error": None if ok else "Transaction not found."}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}

    def _ensure_benchmark(self, symbol: str = "SPY") -> None:
        """Populate the benchmark price series into the local store.

        This is the explicit fetch step for the benchmark — kept separate from
        analytics/backtest, which only ever read from the store. Must be called
        while providers are online (see refresh_prices).
        """
        store = self.manager.market_data_store
        if store is None:
            return
        start = self._history_start()
        end = date.today()

        def _fetch(sym: str, a: date, b: date):
            out = []
            for p in self.data_provider.get_historical_prices(sym, a, b):
                if p.close_price:
                    out.append((p.date, Decimal(str(p.close_price))))
            return out

        store.ensure_prices(symbol, start, end, data_provider=_fetch)

    def _ensure_fx_rates(self) -> None:
        """Seed the persistent FX cache for every currency pair the portfolio
        can need, across the full history window.

        Like _ensure_benchmark, this is the explicit fetch step: analytics only
        ever *reads* FX rates from the cache, so if the cache is incomplete a
        currency-converted history silently falls back to the current rate (or,
        worse, drops positions). Seeding here — during a historical refresh —
        keeps currency conversion seamless for any display currency the UI
        offers, offline, afterwards.
        """
        pm = self.manager
        portfolio = pm.current_portfolio
        dm = getattr(pm, "data_manager", None)
        if portfolio is None or dm is None:
            return
        if not hasattr(dm, "get_historical_fx_rates_range"):
            return

        # Currencies in play = base + every position/cash native currency +
        # every display currency the UI exposes (so switching stays seamless).
        currencies = {portfolio.base_currency}
        for pos in portfolio.positions.values():
            if getattr(pos, "currency", None):
                currencies.add(pos.currency)
        try:
            currencies.update(portfolio.cash_balances.keys())
        except Exception:
            pass
        currencies.update(Currency)

        start = self._history_start()
        end = date.today()

        # get_historical_fx_rates_range stores the pair and the cache inverts on
        # read, so one unordered pair suffices.
        seen: set = set()
        for a in currencies:
            for b in currencies:
                if a == b:
                    continue
                key = frozenset((a.value, b.value))
                if key in seen:
                    continue
                seen.add(key)
                try:
                    dm.get_historical_fx_rates_range(a, b, start, end)
                except Exception:
                    continue

    def refresh_prices(self, historical: bool = False) -> Dict[str, Any]:
        """Fetch live prices (temporarily going online), then return to cache-only."""
        pm = self.manager
        if not pm.current_portfolio:
            return {"ok": False, "error": "No portfolio loaded."}
        was_offline = self.offline
        try:
            self.set_online(True)
            if historical:
                res = pm.update_market_data()
                # Seed the benchmark series into the store as part of the same
                # explicit fetch, so analytics/backtest can read it offline.
                self._ensure_benchmark("SPY")
                # Seed historical FX rates too, so currency-converted analytics
                # stay accurate and seamless offline for any display currency.
                self._ensure_fx_rates()
            else:
                res = pm.update_current_prices()
            updated = sum(1 for v in (res or {}).values() if v)
            total = len(res or {})
            return {"ok": True, "updated": updated, "total": total, "error": None}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        finally:
            if was_offline:
                self.set_online(False)


@lru_cache(maxsize=1)
def get_context() -> AppContext:
    """Process-wide singleton (managers are relatively expensive to build)."""
    return AppContext()
