"""FinAI FastAPI web app.

A purpose-built dark fintech frontend that reuses the existing portfolio
backend in-process (no MCP, no separate service). Server-rendered with Jinja2 +
Tailwind (CDN) + Alpine.js; charts via Plotly.js.
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.web.services import get_context

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


# --- Pages to be filled in next (nav is live so the shell is complete) ------

def _placeholder(request: Request, key: str, portfolio: Optional[str], blurb: str):
    _ctx, base = _base_context(request, key, portfolio)
    base["title"] = next(n["label"] for n in NAV if n["key"] == key)
    base["blurb"] = blurb
    return templates.TemplateResponse("placeholder.html", base)


@app.get("/analytics", response_class=HTMLResponse)
def analytics(request: Request, portfolio: Optional[str] = None):
    return _placeholder(request, "analytics", portfolio,
                        "Performance, risk metrics, benchmark comparison and returns distribution.")


@app.get("/optimize", response_class=HTMLResponse)
def optimize(request: Request, portfolio: Optional[str] = None):
    return _placeholder(request, "optimize", portfolio,
                        "HRP & Markowitz target weights, efficient frontier and suggested rebalancing trades.")


@app.get("/simulate", response_class=HTMLResponse)
def simulate(request: Request, portfolio: Optional[str] = None):
    return _placeholder(request, "simulate", portfolio,
                        "Strategy backtesting and Monte Carlo projections in one place.")


@app.get("/data", response_class=HTMLResponse)
def data(request: Request, portfolio: Optional[str] = None):
    return _placeholder(request, "data", portfolio,
                        "Market data & price management, plus portfolio and app settings.")
