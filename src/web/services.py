"""Backend glue for the FinAI web app.

Builds the existing portfolio backend once and exposes small, JSON-friendly
data assemblers that the FastAPI routes render. No business logic lives here —
this only adapts the in-process managers to the web layer.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data_providers.manager import DataProviderManager
from src.portfolio.manager import PortfolioManager
from src.portfolio.storage import FileBasedStorage
from src.services.market_data_service import MarketDataService
from src.utils.metrics import FinancialMetricsCalculator

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_DIR = os.environ.get("FINAI_DATA_DIR") or os.path.join(_PROJECT_ROOT, "data")


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
        self.metrics = FinancialMetricsCalculator(data_provider)

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
        return target

    # -- data assemblers -----------------------------------------------------

    def _history_start(self) -> date:
        """Earliest transaction date, or one year ago as a floor."""
        txns = self.manager.get_transaction_history()
        floor = date.today() - timedelta(days=365)
        if not txns:
            return floor
        earliest = min(t["timestamp"].date() for t in txns)
        return min(earliest, floor) if earliest < floor else earliest

    def dashboard(self) -> Dict[str, Any]:
        """Everything the dashboard page needs."""
        pm = self.manager
        portfolio = pm.current_portfolio
        if not portfolio:
            return {"empty": True}

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

        # equity curve
        start = self._history_start()
        hist = pm.get_portfolio_history(start, date.today())
        curve = {"dates": [], "values": []}
        if isinstance(hist, pd.DataFrame) and not hist.empty and "total_value" in hist:
            curve["dates"] = [d.strftime("%Y-%m-%d") for d in hist.index]
            curve["values"] = [float(v) for v in hist["total_value"].tolist()]

        # positions (JSON-safe), sorted by value desc
        pos_rows = sorted(
            (
                {
                    "symbol": p["symbol"],
                    "name": p["name"],
                    "quantity": _f(p["quantity"]),
                    "avg_cost": _f(p["average_cost"]),
                    "price": _f(p["current_price"]),
                    "value": _f(p["market_value"]),
                    "pnl": _f(p["unrealized_pnl"]),
                    "pnl_pct": _f(p["unrealized_pnl_percent"]),
                    "currency": p["original_currency"],
                    "has_price": p["has_current_price"],
                }
                for p in positions
            ),
            key=lambda r: r["value"] or 0,
            reverse=True,
        )

        # allocation (positions + cash)
        alloc = [{"label": r["symbol"], "value": r["value"] or 0} for r in pos_rows if (r["value"] or 0) > 0]
        if cash and cash > 0:
            alloc.append({"label": "Cash", "value": float(cash)})

        # recent transactions
        txns = pm.get_transaction_history()[:12]
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
            "kpis": {
                "net_worth": _f(net_worth),
                "invested": _f(invested),
                "cash": _f(cash),
                "unrealized_pnl": _f(unrealized),
                "total_return_pct": total_return_pct,
                "ytd_pct": ytd_pct,
            },
            "curve": curve,
            "positions": pos_rows,
            "allocation": alloc,
            "transactions": tx_rows,
        }


@lru_cache(maxsize=1)
def get_context() -> AppContext:
    """Process-wide singleton (managers are relatively expensive to build)."""
    return AppContext()
