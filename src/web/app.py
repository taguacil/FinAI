"""FinAI FastAPI web app.

A purpose-built dark fintech frontend that reuses the existing portfolio
backend in-process (no MCP, no separate service). Server-rendered with Jinja2 +
Tailwind (CDN) + Alpine.js; charts via Plotly.js.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.web.services import get_context


def _parse_date(value: Optional[str], default: date) -> date:
    if not value:
        return default
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return default


def _csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip().upper() for v in value.split(",") if v.strip()]

_HERE = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(_HERE, "templates"))

app = FastAPI(title="FinAI", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=os.path.join(_HERE, "static")), name="static")

# Consolidated information architecture (7 Streamlit tabs -> 5 pages).
NAV = [
    {"key": "dashboard", "label": "Dashboard", "href": "/", "icon": "layout-dashboard"},
    {"key": "analytics", "label": "Analytics", "href": "/analytics", "icon": "line-chart"},
    {"key": "optimize", "label": "Optimize", "href": "/optimize", "icon": "scale"},
    {"key": "simulate", "label": "Simulate", "href": "/simulate", "icon": "flask-conical"},
    {"key": "data", "label": "Data & Settings", "href": "/data", "icon": "database"},
]


def _base_context(request: Request, active: str, portfolio: Optional[str]):
    ctx = get_context()
    active_id = ctx.ensure_loaded(portfolio)
    portfolios = ctx.list_portfolios()
    active_name = next((p["name"] for p in portfolios if p["id"] == active_id), None)
    return ctx, {
        "request": request,
        "nav": NAV,
        "active": active,
        "portfolios": portfolios,
        "active_portfolio_id": active_id,
        "active_portfolio_name": active_name,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, portfolio: Optional[str] = None):
    ctx, base = _base_context(request, "dashboard", portfolio)
    base["data"] = ctx.dashboard()
    return templates.TemplateResponse("dashboard.html", base)


@app.get("/api/dashboard")
def api_dashboard(portfolio: Optional[str] = None):
    ctx = get_context()
    ctx.ensure_loaded(portfolio)
    return JSONResponse(ctx.dashboard())


@app.get("/analytics", response_class=HTMLResponse)
def analytics(request: Request, portfolio: Optional[str] = None,
              days: int = 365, benchmark: str = "SPY"):
    ctx, base = _base_context(request, "analytics", portfolio)
    base["data"] = ctx.analytics(days=days, benchmark=benchmark or "SPY")
    base["days"] = days
    return templates.TemplateResponse("analytics.html", base)


@app.get("/optimize", response_class=HTMLResponse)
def optimize(request: Request, portfolio: Optional[str] = None, run: int = 0,
             lookback_days: int = 252, risk_free_rate: float = 0.04,
             objective: str = "max_sharpe", include_cash: int = 1):
    ctx, base = _base_context(request, "optimize", portfolio)
    base["data"] = ctx.optimize(
        run=bool(run), lookback_days=lookback_days, risk_free_rate=risk_free_rate,
        objective=objective, include_cash=bool(include_cash),
    )
    return templates.TemplateResponse("optimize.html", base)


@app.get("/simulate", response_class=HTMLResponse)
def simulate(
    request: Request, portfolio: Optional[str] = None, tab: str = "backtest",
    # backtest params
    bt_run: int = 0, symbols: Optional[str] = None,
    start: Optional[str] = None, end: Optional[str] = None,
    initial_capital: float = 100_000.0, rebalance: str = "monthly",
    strategies: Optional[str] = None, bt_benchmark: str = "SPY",
    lookback_days: int = 252, cost_bps: float = 0.0, rf: float = 0.04,
    # monte carlo params
    mc_run: int = 0, scenario: str = "likely", years: float = 5.0,
    runs: int = 1000, deposit: float = 0.0, withdrawal: float = 0.0,
):
    ctx, base = _base_context(request, "simulate", portfolio)
    today = date.today()
    universe = _csv(symbols) or ctx.default_universe()
    strat_specs = _csv(strategies) or ["hrp", "equal_weight", "buy_and_hold"]
    # strategies come in as UPPER from _csv; normalize to spec keys
    strat_specs = [s.lower() for s in strat_specs]
    base["tab"] = "montecarlo" if tab == "montecarlo" else "backtest"
    base["backtest"] = ctx.backtest(
        run=bool(bt_run), symbols=universe,
        start=_parse_date(start, today - timedelta(days=730)),
        end=_parse_date(end, today),
        initial_capital=initial_capital, rebalance_frequency=rebalance,
        strategy_specs=strat_specs, benchmark_symbol=bt_benchmark or "SPY",
        lookback_days=lookback_days, transaction_cost_bps=cost_bps, risk_free_rate=rf,
    )
    base["montecarlo"] = ctx.monte_carlo(
        run=bool(mc_run), scenario=scenario, projection_years=years,
        monte_carlo_runs=runs, monthly_deposit=deposit, monthly_withdrawal=withdrawal,
    )
    return templates.TemplateResponse("simulate.html", base)


@app.get("/data", response_class=HTMLResponse)
def data(request: Request, portfolio: Optional[str] = None,
         msg: Optional[str] = None, err: Optional[str] = None):
    ctx, base = _base_context(request, "data", portfolio)
    base["data"] = ctx.data_page()
    base["flash"] = {"msg": msg, "err": err}
    return templates.TemplateResponse("data.html", base)


@app.post("/data/set-price")
def data_set_price(portfolio: Optional[str] = Form(None), symbol: str = Form(...),
                   price: float = Form(...), currency: Optional[str] = Form(None)):
    ctx = get_context()
    ctx.ensure_loaded(portfolio)
    res = ctx.set_price(symbol, price, currency or None)
    q = f"?portfolio={portfolio or ''}"
    q += f"&msg=Price+set+for+{symbol}" if res["ok"] else f"&err={(res.get('error') or 'Failed').replace(' ', '+')}"
    return RedirectResponse(url=f"/data{q}", status_code=303)


@app.post("/data/refresh")
def data_refresh(portfolio: Optional[str] = Form(None), historical: int = Form(0)):
    ctx = get_context()
    ctx.ensure_loaded(portfolio)
    res = ctx.refresh_prices(historical=bool(historical))
    q = f"?portfolio={portfolio or ''}"
    if res["ok"]:
        q += f"&msg=Updated+{res.get('updated', 0)}+of+{res.get('total', 0)}+symbols"
    else:
        q += f"&err={(res.get('error') or 'Failed').replace(' ', '+')}"
    return RedirectResponse(url=f"/data{q}", status_code=303)
