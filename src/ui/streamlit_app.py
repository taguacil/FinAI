"""
Streamlit UI for the Portfolio Tracker with AI Agent.
"""

import os
import sys
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pypdf import PdfReader

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.agents.portfolio_agent import PortfolioAgent
from src.data_providers.manager import DataProviderManager
from src.portfolio.manager import PortfolioManager
from src.portfolio.models import Currency, TransactionType
from src.portfolio.storage import FileBasedStorage
from src.utils.metrics import FinancialMetricsCalculator


class PortfolioTrackerUI:
    """Streamlit UI for Portfolio Tracker."""

    def __init__(self):
        """Initialize the UI."""
        self.setup_page_config()
        self.initialize_components()

    def setup_page_config(self):
        """Configure Streamlit page."""
        st.set_page_config(
            page_title="AI Portfolio Tracker",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    @st.cache_resource
    def initialize_components(_self):
        """Initialize portfolio components with caching."""
        try:
            storage = FileBasedStorage()
            data_manager = DataProviderManager()
            portfolio_manager = PortfolioManager(storage, data_manager)
            metrics_calculator = FinancialMetricsCalculator(data_manager)

            agent = PortfolioAgent(
                portfolio_manager=portfolio_manager,
                data_manager=data_manager,
                metrics_calculator=metrics_calculator,
            )

            return portfolio_manager, agent, metrics_calculator
        except Exception as e:
            st.error(f"Error initializing components: {e}")
            return None, None, None

    def run(self):
        """Run the Streamlit app."""
        # Initialize components
        portfolio_manager, agent, metrics_calculator = self.initialize_components()

        if not all([portfolio_manager, agent, metrics_calculator]):
            st.error("Failed to initialize application components.")
            return

        # Initialize session state
        self.init_session_state()

        # Ensure previously selected portfolio remains loaded across reruns
        try:
            sel_id = st.session_state.get("selected_portfolio")
            if sel_id:
                if (
                    not portfolio_manager.current_portfolio
                    or portfolio_manager.current_portfolio.id != sel_id
                ):
                    loaded = portfolio_manager.load_portfolio(sel_id)
                    if loaded:
                        st.session_state.portfolio_loaded = True
        except Exception:
            pass

        # Header
        st.title("🤖 AI Portfolio Tracker")
        st.markdown(
            "*Your intelligent financial companion for portfolio management and investment advice*"
        )

        # Sidebar
        self.render_sidebar(portfolio_manager)

        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            ["💬 AI Chat", "📊 Portfolio", "📈 Analytics", "⚙️ Settings"]
        )

        with tab1:
            self.render_chat_interface(agent)

        with tab2:
            self.render_portfolio_overview(portfolio_manager)

        with tab3:
            self.render_analytics(portfolio_manager, metrics_calculator)

        with tab4:
            self.render_settings()

    def init_session_state(self):
        """Initialize session state variables."""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "portfolio_loaded" not in st.session_state:
            st.session_state.portfolio_loaded = False
        if "selected_portfolio" not in st.session_state:
            st.session_state.selected_portfolio = None

    def render_sidebar(self, portfolio_manager):
        """Render the sidebar with portfolio management."""
        st.sidebar.header("📁 Portfolio Management")

        # Portfolio selection (show by name instead of ID)
        portfolio_ids = portfolio_manager.list_portfolios()
        if portfolio_ids:
            id_to_name: Dict[str, str] = {}
            options: List[str] = ["None"]
            try:
                for pid in portfolio_ids:
                    p = portfolio_manager.storage.load_portfolio(pid)
                    display_name = p.name if p else pid
                    id_to_name[pid] = display_name
                    options.append(display_name)
            except Exception:
                # Fallback to IDs only if loading fails
                id_to_name = {pid: pid for pid in portfolio_ids}
                options = ["None"] + portfolio_ids

            # Compute default index based on selected portfolio id
            default_index = 0
            if st.session_state.selected_portfolio:
                sel_id = st.session_state.selected_portfolio
                sel_name = id_to_name.get(sel_id)
                if sel_name and sel_name in options:
                    default_index = options.index(sel_name)

            chosen_name = st.sidebar.selectbox(
                "Select Portfolio:", options, index=default_index
            )

            if chosen_name != "None":
                # Map back to id
                # Find the first id with this name (names are usually unique)
                chosen_id = None
                for pid, nm in id_to_name.items():
                    if nm == chosen_name:
                        chosen_id = pid
                        break
                # If different choice, or not currently loaded, load it
                if chosen_id and (
                    chosen_id != st.session_state.selected_portfolio
                    or not portfolio_manager.current_portfolio
                    or portfolio_manager.current_portfolio.id != chosen_id
                ):
                    portfolio = portfolio_manager.load_portfolio(chosen_id)
                    if portfolio:
                        st.session_state.portfolio_loaded = True
                        st.session_state.selected_portfolio = chosen_id
                        st.sidebar.success(f"Loaded: {portfolio.name}")
                    else:
                        st.sidebar.error("Failed to load portfolio")

        # Create new portfolio
        st.sidebar.subheader("Create New Portfolio")
        with st.sidebar.form("create_portfolio"):
            new_name = st.text_input("Portfolio Name")
            base_currency = st.selectbox("Base Currency", [c.value for c in Currency])

            if st.form_submit_button("Create Portfolio"):
                if new_name:
                    try:
                        portfolio = portfolio_manager.create_portfolio(
                            new_name, Currency(base_currency)
                        )
                        st.session_state.portfolio_loaded = True
                        st.session_state.selected_portfolio = portfolio.id
                        st.sidebar.success(f"Created portfolio: {new_name}")
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Error creating portfolio: {e}")
                else:
                    st.sidebar.error("Please enter a portfolio name")

        # Portfolio info
        if st.session_state.portfolio_loaded and portfolio_manager.current_portfolio:
            st.sidebar.subheader("Current Portfolio")
            portfolio = portfolio_manager.current_portfolio
            st.sidebar.info(
                f"""
            **Name:** {portfolio.name}
            **Base Currency:** {portfolio.base_currency.value}
            **Created:** {portfolio.created_at.strftime('%Y-%m-%d')}
            **Positions:** {len(portfolio.positions)}
            """
            )

            # Quick actions
            st.sidebar.subheader("🔄 Data Management")

            # Combined update: current prices + snapshots since last
            if st.sidebar.button(
                "🔄 Update Portfolio (Prices + Snapshots)", type="primary"
            ):
                with st.spinner("Updating prices and snapshots..."):
                    price_results = portfolio_manager.update_current_prices()
                    success_count = sum(price_results.values())
                    total_count = len(price_results)
                    snapshots = portfolio_manager.create_snapshots_since_last()
                    st.sidebar.success(
                        f"Prices: {success_count}/{total_count} updated | Snapshots created: {len(snapshots)}"
                    )
                    st.rerun()

    def render_chat_interface(self, agent):
        """Render the AI chat interface."""
        st.header("💬 Chat with AI Financial Advisor")

        # Chat history
        chat_container = st.container()

        with chat_container:
            if not st.session_state.chat_history:
                # Initialize conversation
                initial_message = agent.initialize_conversation()
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": initial_message}
                )

            # Display chat history
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])

        # Chat input
        user_input = st.chat_input(
            "Ask me about your portfolio, investments, or market conditions..."
        )

        if user_input:
            # Add user message
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            # Get AI response
            with st.spinner("Thinking..."):
                try:
                    response = agent.chat(user_input)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": error_msg}
                    )

            st.rerun()

        # Model selection + PDF + Quick action buttons
        st.subheader("Quick Actions")
        col_pdf, col0, col1, col2, col3 = st.columns([2, 2, 1, 1, 1])

        # PDF Uploader
        with col_pdf:
            uploaded_pdf = st.file_uploader(
                "Attach PDF (datasheet)", type=["pdf"], accept_multiple_files=False
            )
            if uploaded_pdf is not None:
                st.caption(
                    f"Selected: {uploaded_pdf.name} ({uploaded_pdf.size/1024:.1f} KB)"
                )
                pdf_bytes = uploaded_pdf.read()
                if st.button("Attach PDF to Chat"):
                    try:
                        import io

                        reader = PdfReader(io.BytesIO(pdf_bytes))
                        texts = []
                        for page in reader.pages:
                            try:
                                t = page.extract_text() or ""
                                if t:
                                    texts.append(t)
                            except Exception:
                                continue
                        content = "\n\n".join(texts).strip()
                        if not content:
                            st.warning("No text extracted from the PDF.")
                        else:
                            if len(content) > 100_000:
                                content = content[:100_000] + "\n\n...[truncated]"
                            st.session_state.chat_history.append(
                                {
                                    "role": "user",
                                    "content": f"📄 Attached PDF: {uploaded_pdf.name}\n\n{content}",
                                }
                            )
                            st.success("PDF content attached to chat.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Failed to read PDF: {e}")
                if st.button("Analyze PDF Now"):
                    try:
                        import io

                        reader = PdfReader(io.BytesIO(pdf_bytes))
                        texts = []
                        for page in reader.pages:
                            try:
                                t = page.extract_text() or ""
                                if t:
                                    texts.append(t)
                            except Exception:
                                continue
                        content = "\n\n".join(texts).strip()
                        if not content:
                            st.warning("No text extracted from the PDF.")
                        else:
                            if len(content) > 80_000:
                                content = content[:80_000] + "\n\n...[truncated]"
                            prompt = (
                                f"Please analyze the attached PDF datasheet '{uploaded_pdf.name}'. "
                                f"Summarize key points, risks, and any financial metrics or terms.\n\n"
                                f"Extracted text follows:\n{content}"
                            )
                            st.session_state.chat_history.append(
                                {"role": "user", "content": prompt}
                            )
                            with st.spinner("Analyzing PDF..."):
                                response = agent.chat(prompt)
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": response}
                            )
                            st.rerun()
                    except Exception as e:
                        st.error(f"Failed to analyze PDF: {e}")

        with col0:
            st.markdown("**AI Model**")
            # Single dropdown with all supported models and their providers
            all_models = [
                (
                    "Azure GPT-4.1",
                    {
                        "provider": "azure-openai",
                        "endpoint": "https://kallamai.openai.azure.com/",
                        "model": "gpt-4.1",
                    },
                ),
                (
                    "Azure GPT-4.1 Mini",
                    {
                        "provider": "azure-openai",
                        "endpoint": "https://kallamai.openai.azure.com/",
                        "model": "gpt-4.1-mini",
                    },
                ),
                (
                    "Azure o4-mini",
                    {
                        "provider": "azure-openai",
                        "endpoint": "https://kallamai.openai.azure.com/",
                        "model": "o4-mini",
                    },
                ),
                (
                    "Azure GPT-5",
                    {
                        "provider": "azure-openai",
                        "endpoint": "https://kallamai.openai.azure.com/",
                        "model": "gpt-5",
                    },
                ),
                (
                    "Azure GPT-5 Mini",
                    {
                        "provider": "azure-openai",
                        "endpoint": "https://kallamai.openai.azure.com/",
                        "model": "gpt-5-mini",
                    },
                ),
                (
                    "Claude Sonnet 4 (thinking)",
                    {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
                ),
                (
                    "Gemini 2.0 Flash Lite",
                    {"provider": "vertex-ai", "model": "gemini-2.0-flash-lite-001"},
                ),
                (
                    "Gemini 2.5 Pro (thinking)",
                    {"provider": "vertex-ai", "model": "gemini-2.5-pro"},
                ),
            ]
            model_choice = st.selectbox("Model", all_models, format_func=lambda x: x[0])
            if st.button("Apply Model"):
                try:
                    meta = model_choice[1]
                    provider_key = meta.get("provider")
                    if provider_key == "azure-openai":
                        azure_key = os.getenv("AZURE_OPENAI_API_KEY", "")
                        agent.set_llm_config(
                            provider="azure-openai",
                            azure_endpoint=meta.get("endpoint"),
                            azure_api_key=azure_key,
                            azure_model=meta.get("model"),
                        )
                    elif provider_key == "anthropic":
                        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
                        agent.set_llm_config(
                            provider="anthropic",
                            anthropic_api_key=anthropic_key,
                            anthropic_model=meta.get("model"),
                        )
                    else:
                        project = os.getenv(
                            "GOOGLE_VERTEX_PROJECT", "mystic-fountain-415918"
                        )
                        location = os.getenv("GOOGLE_VERTEX_LOCATION", "us-central1")
                        agent.set_llm_config(
                            provider="vertex-ai",
                            vertex_project=project,
                            vertex_location=location,
                            vertex_model=meta.get("model"),
                        )
                    st.success(f"Model set to {model_choice[0]}")
                except Exception as e:
                    st.error(f"Failed to set model: {e}")

        with col1:
            if st.button("📊 Portfolio Summary"):
                if st.session_state.portfolio_loaded:
                    response = agent.chat("Please show me my current portfolio summary")
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                    st.rerun()
                else:
                    st.warning("Please load a portfolio first")

        with col2:
            if st.button("📈 Performance Analysis"):
                if st.session_state.portfolio_loaded:
                    response = agent.analyze_portfolio_performance()
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                    st.rerun()
                else:
                    st.warning("Please load a portfolio first")

        with col3:
            if st.button("🧹 Clear Chat"):
                st.session_state.chat_history = []
                agent.clear_conversation()
                st.rerun()

    def render_portfolio_overview(self, portfolio_manager):
        """Render portfolio overview with transactions."""
        st.header("📊 Portfolio Overview")

        if not st.session_state.portfolio_loaded:
            st.warning("Please load or create a portfolio to view details.")
            return

        # Portfolio summary
        portfolio = portfolio_manager.current_portfolio

        total_value = portfolio_manager.get_portfolio_value()
        positions = portfolio_manager.get_position_summary()

        # Prepare YTD reference prices from local snapshots (no network)
        ytd_start = date(date.today().year, 1, 1)
        try:
            ytd_snaps = portfolio_manager.storage.load_snapshots(
                portfolio.id, ytd_start, date.today()
            )
            # Reference price: first available snapshot in YTD where the symbol appears with a price
            ref_prices_by_symbol: Dict[str, Decimal] = {}
            for snap in ytd_snaps:
                for sym, snap_pos in snap.positions.items():
                    if (
                        sym not in ref_prices_by_symbol
                        and snap_pos.current_price is not None
                    ):
                        ref_prices_by_symbol[sym] = snap_pos.current_price

            # Current price: last available snapshot (scan from most recent backwards to fill gaps)
            curr_prices_by_symbol: Dict[str, Decimal] = {}
            for snap in reversed(ytd_snaps):
                for sym, snap_pos in snap.positions.items():
                    if (
                        sym not in curr_prices_by_symbol
                        and snap_pos.current_price is not None
                    ):
                        curr_prices_by_symbol[sym] = snap_pos.current_price

            # Fallback to latest snapshot overall if YTD list is empty
            if not ytd_snaps:
                latest_snap = portfolio_manager.storage.get_latest_snapshot(
                    portfolio.id
                )
                if latest_snap:
                    for sym, snap_pos in latest_snap.positions.items():
                        if snap_pos.current_price is not None:
                            curr_prices_by_symbol[sym] = snap_pos.current_price
        except Exception:
            ref_prices_by_symbol = {}
            curr_prices_by_symbol = {}

        # Enable FX conversion for accurate base-currency views
        fetch_live = True

        # Check data freshness
        positions_with_prices = [
            pos for pos in positions if pos.get("current_price") is not None
        ]
        positions_without_prices = [
            pos for pos in positions if pos.get("current_price") is None
        ]

        # Show data freshness warning
        if positions_without_prices:
            st.warning(
                f"⚠️ {len(positions_without_prices)} positions have no current prices. Use 'Update Current Prices' to fetch latest data."
            )

        if positions_with_prices:
            # Find the most recent price update
            latest_update = max(
                (
                    pos.get("last_updated")
                    for pos in positions_with_prices
                    if pos.get("last_updated")
                ),
                default=None,
            )
            if latest_update:
                st.info(
                    f"📅 Latest price update: {latest_update.strftime('%Y-%m-%d %H:%M')}"
                )

        # Show latest snapshot date
        latest_snapshot = portfolio_manager.storage.get_latest_snapshot(portfolio.id)
        if latest_snapshot:
            st.info(f"🗓️ Latest snapshot: {latest_snapshot.date.isoformat()}")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Value", f"${total_value:,.2f}")

        with col2:
            total_positions = len(positions)
            st.metric("Positions", total_positions)

        with col3:
            # Sum cash across currencies in base currency
            cash_total_base = Decimal("0")
            if portfolio.cash_balances:
                for curr, amt in portfolio.cash_balances.items():
                    curr_code = getattr(curr, "value", str(curr))
                    cash_total_base += self._convert_to_base(
                        portfolio_manager,
                        Decimal(str(amt)),
                        curr_code,
                        portfolio.base_currency.value,
                        allow_fetch=True,
                    )
            st.metric("Cash (base)", f"${cash_total_base:,.2f}")

        with col4:
            # Calculate total P&L
            total_pnl = sum(
                float(pos.get("unrealized_pnl", 0) or 0) for pos in positions
            )
            st.metric("Unrealized P&L", f"${total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")

        # Additional metrics: YTD Performance (TWR) and Unrealized P&L (%)
        ytd_col1, ytd_col2 = st.columns(2)

        # YTD performance (time-weighted): remove external cash flows
        try:
            portfolio_id = portfolio_manager.current_portfolio.id
            start_of_year = date(date.today().year, 1, 1)
            today = date.today()
            ytd_snaps = portfolio_manager.storage.load_snapshots(
                portfolio_id, start_of_year, today
            )

            if len(ytd_snaps) >= 2:
                # Build external cash flows by day (respect live fetch toggle)
                if fetch_live:
                    flows_dec = portfolio_manager.get_external_cash_flows_by_day(
                        start_of_year, today
                    )
                else:
                    # Base-only approximation without FX calls
                    flows_dec = {}
                    base = portfolio_manager.current_portfolio.base_currency
                    for txn in portfolio_manager.current_portfolio.transactions:
                        d = txn.timestamp.date()
                        if d < start_of_year or d > today:
                            continue
                        if txn.transaction_type not in [
                            TransactionType.DEPOSIT,
                            TransactionType.WITHDRAWAL,
                        ]:
                            continue
                        if txn.currency == base:
                            amt = txn.total_value
                            if txn.transaction_type == TransactionType.WITHDRAWAL:
                                amt = -amt
                            flows_dec[d] = flows_dec.get(d, Decimal("0")) + amt

                flows_float = {d: float(v) for d, v in flows_dec.items()}

                # Compute TWR using the calculator's return function
                calc = FinancialMetricsCalculator(portfolio_manager.data_manager)
                daily_returns = calc.calculate_returns(ytd_snaps, flows_float)

                if daily_returns:
                    # Geometric aggregation for period return
                    twr = 1.0
                    for r in daily_returns:
                        twr *= 1.0 + r
                    ytd_perf_pct = (twr - 1.0) * 100.0
                    ytd_col1.metric("YTD Performance (TWR)", f"{ytd_perf_pct:.2f}%")
                else:
                    ytd_col1.metric("YTD Performance (TWR)", "N/A")
            else:
                ytd_col1.metric("YTD Performance (TWR)", "N/A")
        except Exception:
            ytd_col1.metric("YTD Performance (TWR)", "N/A")

        # Unrealized P&L percentage from current positions
        try:
            total_cost_basis = sum(
                float(pos.get("cost_basis", 0) or 0) for pos in positions
            )
            if total_cost_basis > 0:
                unrealized_pct = total_pnl / total_cost_basis * 100.0
                ytd_col2.metric("Unrealized P&L (%)", f"{unrealized_pct:.2f}%")
            else:
                ytd_col2.metric("Unrealized P&L (%)", "N/A")
        except Exception:
            ytd_col2.metric("Unrealized P&L (%)", "N/A")

        # Positions by category (card layout)
        if positions:
            st.subheader("📈 Current Positions")

            portfolio = portfolio_manager.current_portfolio
            base_currency = portfolio.base_currency

            # Build enriched rows with ISIN, base market value, category, and YTD PnL
            enriched = []
            start_of_year = date(date.today().year, 1, 1)

            for pos in positions:
                symbol = pos["symbol"]
                instrument = (
                    portfolio.positions.get(symbol).instrument
                    if symbol in portfolio.positions
                    else None
                )
                isin = instrument.isin if instrument else None
                currency_code = pos.get("currency")

                # Market value in base currency
                mv = pos.get("market_value") or Decimal("0")
                mv_base = self._convert_to_base(
                    portfolio_manager,
                    mv,
                    currency_code,
                    base_currency.value,
                    allow_fetch=True,
                )

                # Convert unrealized PnL to base currency
                unreal_val_native = pos.get("unrealized_pnl") or Decimal("0")
                unreal_val_base = self._convert_to_base(
                    portfolio_manager,
                    unreal_val_native,
                    currency_code,
                    base_currency.value,
                    allow_fetch=True,
                )

                # Category classification
                category = self._classify_position(pos, instrument)

                # YTD market-only (local snapshots): compare current price to first YTD snapshot price
                ytd_market_pnl_native = None
                ytd_market_pct = None
                try:
                    if symbol in ref_prices_by_symbol:
                        ref_price = Decimal(str(ref_prices_by_symbol[symbol]))
                        qty_dec = Decimal(str(pos.get("quantity") or 0))
                        # Prefer latest snapshot price if available; fallback to UI position price
                        curr_px_val = (
                            curr_prices_by_symbol.get(symbol)
                            if symbol in curr_prices_by_symbol
                            else pos.get("current_price")
                        )
                        curr_px = Decimal(str(curr_px_val or 0))
                        if qty_dec > 0 and ref_price and curr_px:
                            ytd_market_pnl_native = (curr_px - ref_price) * qty_dec
                            base_val_start_native = ref_price * qty_dec
                            if base_val_start_native != 0:
                                ytd_market_pct = (
                                    ytd_market_pnl_native / base_val_start_native
                                ) * 100
                except Exception:
                    pass

                # Total buy price (cost basis) in native and base
                total_buy_native = pos.get("cost_basis")
                total_buy_base = (
                    self._convert_to_base(
                        portfolio_manager,
                        (
                            total_buy_native
                            if total_buy_native is not None
                            else Decimal("0")
                        ),
                        currency_code,
                        base_currency.value,
                        allow_fetch=False,
                    )
                    if total_buy_native is not None
                    else None
                )

                # Latest purchase date for this position
                latest_buy_date = None
                try:
                    buys = [
                        t.timestamp.date()
                        for t in portfolio.transactions
                        if t.instrument.symbol == symbol
                        and t.transaction_type == TransactionType.BUY
                    ]
                    if buys:
                        latest_buy_date = max(buys).isoformat()
                except Exception:
                    latest_buy_date = None

                enriched.append(
                    {
                        **pos,
                        "isin": isin or "-",
                        "currency": currency_code,
                        "market_value_base": mv_base,
                        "unrealized_pnl_base": unreal_val_base,
                        "category": category,
                        "ytd_market_pnl": ytd_market_pnl_native,
                        "ytd_market_pnl_percent": ytd_market_pct,
                        "total_buy_price": total_buy_native,
                        "total_buy_price_base": total_buy_base,
                        "latest_buy_date": latest_buy_date,
                    }
                )

            # No YTD aggregate metrics in overview to avoid external data fetches

            # Render by category using cards
            categories = [
                "Short Term",
                "Bonds",
                "Equities",
                "Alternatives",
                "Miscellaneous",
            ]
            for cat in categories:
                group = [e for e in enriched if e.get("category") == cat]
                # Include cash under Short Term as its own items
                cash_items = []
                if cat == "Short Term" and portfolio.cash_balances:
                    for curr, amt in portfolio.cash_balances.items():
                        curr_code = getattr(curr, "value", str(curr))
                        # FX summary for cash (compat with older cached manager)
                        fx_summary = self._get_cash_fx_summary(portfolio_manager).get(
                            curr
                        )
                        amt_base = self._convert_to_base(
                            portfolio_manager,
                            Decimal(str(amt)),
                            curr_code,
                            base_currency.value,
                            allow_fetch=True,
                        )
                        # Compute FX P&L percent if base cost available (non-base currency only)
                        is_base_cur = curr_code == base_currency.value
                        fx_pnl_base = (
                            (
                                fx_summary.get("fx_unrealized_pnl_base")
                                if fx_summary
                                else None
                            )
                            if not is_base_cur
                            else None
                        )
                        base_cost = fx_summary.get("base_cost") if fx_summary else None
                        fx_pnl_pct = (
                            (fx_pnl_base / base_cost * 100)
                            if (not is_base_cur)
                            and fx_summary
                            and base_cost not in (None, Decimal("0"))
                            else None
                        )
                        # Robust YTD FX using manager method (respects purchase dates)
                        ytd_fx_base = None
                        ytd_fx_pct = None
                        try:
                            if hasattr(portfolio_manager, "get_cash_ytd_fx_summary"):
                                ysum = portfolio_manager.get_cash_ytd_fx_summary().get(
                                    curr
                                )
                                if ysum:
                                    ytd_fx_base = ysum.get("ytd_fx_pnl_base")
                                    ytd_fx_pct = ysum.get("ytd_fx_percent")
                        except Exception:
                            ytd_fx_base = None
                            ytd_fx_pct = None

                        cash_items.append(
                            {
                                "name": f"Cash ({curr_code})",
                                "isin": "-",
                                "currency": curr_code,
                                "instrument_type": "cash",
                                "quantity": None,
                                "current_price": None,
                                "market_value_base": amt_base,
                                "market_value": Decimal(str(amt)),
                                "unrealized_pnl_base": fx_pnl_base,
                                "unrealized_pnl_percent": fx_pnl_pct,
                                "ytd_fx_pnl_base": ytd_fx_base,
                                "ytd_fx_percent": ytd_fx_pct,
                                "ytd_unrealized_pnl": None,
                                "ytd_unrealized_pnl_percent": None,
                            }
                        )

                if not group and not cash_items:
                    continue

                st.markdown(f"### {cat}")
                items = group + cash_items
                # Display in rows of 3 cards
                for i in range(0, len(items), 3):
                    cols = st.columns(3)
                    for col, item in zip(cols, items[i : i + 3]):
                        with col:
                            self._render_position_card(item, base_currency.value)

            # Allocation charts
            st.subheader("📊 Allocation")
            col_a, col_b = st.columns(2)
            with col_a:
                self.plot_allocation_by_category(
                    enriched,
                    base_currency.value,
                    portfolio_manager,
                    allow_fetch=fetch_live,
                )
            with col_b:
                self.plot_allocation_by_currency(
                    enriched,
                    base_currency.value,
                    portfolio_manager,
                    allow_fetch=fetch_live,
                )

        # Add transaction form
        st.subheader("➕ Add Transaction")
        self.render_transaction_form(portfolio_manager)

        # Recent transactions
        st.subheader("📝 Recent Transactions")
        transactions = portfolio_manager.get_transaction_history()

        if transactions:
            df_transactions = pd.DataFrame(transactions)
            df_transactions["timestamp"] = pd.to_datetime(
                df_transactions["timestamp"]
            ).dt.strftime("%Y-%m-%d %H:%M")

            # Format monetary columns
            for col in ["quantity", "price", "total_value"]:
                if col in df_transactions.columns:
                    df_transactions[col] = df_transactions[col].apply(
                        lambda x: f"{float(x):,.2f}"
                    )

            st.dataframe(df_transactions, use_container_width=True)
        else:
            st.info("No recent transactions found.")

    def plot_allocation_by_category(
        self,
        positions,
        base_currency_code: str,
        portfolio_manager,
        allow_fetch: bool = False,
    ):
        """Plot allocation by category using base-currency market values."""
        if not positions:
            return

        totals: Dict[str, float] = {}
        for pos in positions:
            category = pos.get("category", "Miscellaneous")
            mv_base = pos.get("market_value_base") or Decimal("0")
            totals[category] = totals.get(category, 0.0) + float(mv_base)

        # Include cash balances as part of Short Term
        cash_total_base = Decimal("0")
        portfolio = portfolio_manager.current_portfolio
        if portfolio and portfolio.cash_balances:
            for curr, amt in portfolio.cash_balances.items():
                curr_code = getattr(curr, "value", str(curr))
                cash_total_base += self._convert_to_base(
                    portfolio_manager,
                    Decimal(str(amt)),
                    curr_code,
                    base_currency_code,
                    allow_fetch=allow_fetch,
                )
        if cash_total_base > 0:
            totals["Short Term"] = totals.get("Short Term", 0.0) + float(
                cash_total_base
            )

        labels = list(totals.keys())
        values = [totals[k] for k in labels]

        if not values or sum(values) <= 0:
            return

        fig = px.pie(values=values, names=labels, title="Allocation by Category")
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label+value",
            texttemplate=f"%{{label}}<br>%{{value:,.2f}} {base_currency_code}<br>%{{percent}}",
            hovertemplate=f"%{{label}}<br>Value: %{{value:,.2f}} {base_currency_code}<br>Share: %{{percent}}<extra></extra>",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    def plot_allocation_by_currency(
        self,
        positions,
        base_currency_code: str,
        portfolio_manager,
        allow_fetch: bool = False,
    ):
        """Plot allocation by instrument currency (converted to base for weights)."""
        if not positions:
            return

        totals: Dict[str, float] = {}
        for pos in positions:
            currency = pos.get("currency", base_currency_code)
            if allow_fetch:
                mv_base = pos.get("market_value_base") or Decimal("0")
                totals[currency] = totals.get(currency, 0.0) + float(mv_base)
            else:
                # Without fetching FX, only include base currency positions
                if currency == base_currency_code:
                    mv_base = pos.get("market_value_base") or Decimal("0")
                    totals[currency] = totals.get(currency, 0.0) + float(mv_base)

        # Include cash balances by currency
        portfolio = portfolio_manager.current_portfolio
        if portfolio and portfolio.cash_balances:
            for curr, amt in portfolio.cash_balances.items():
                curr_code = getattr(curr, "value", str(curr))
                if allow_fetch:
                    amt_base = self._convert_to_base(
                        portfolio_manager,
                        Decimal(str(amt)),
                        curr_code,
                        base_currency_code,
                        allow_fetch=allow_fetch,
                    )
                    totals[curr_code] = totals.get(curr_code, 0.0) + float(amt_base)
                else:
                    if curr_code == base_currency_code:
                        totals[curr_code] = totals.get(curr_code, 0.0) + float(
                            Decimal(str(amt))
                        )

        labels = list(totals.keys())
        values = [totals[k] for k in labels]

        if not values or sum(values) <= 0:
            return

        title = f"Allocation by Currency (in {base_currency_code}{' (base-only without FX)' if not allow_fetch else ''})"
        fig = px.pie(values=values, names=labels, title=title)
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label+value",
            texttemplate=f"%{{label}}<br>%{{value:,.2f}} {base_currency_code}<br>%{{percent}}",
            hovertemplate=f"%{{label}}<br>Value: %{{value:,.2f}} {base_currency_code}<br>Share: %{{percent}}<extra></extra>",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    def _render_position_card(self, item: Dict, base_currency_code: str) -> None:
        """Render a compact, color-coded card (HTML) for a position."""
        name = item.get("name", "-")
        isin = item.get("isin", "-")
        currency = item.get("currency", "-")
        qty = item.get("quantity")
        price = item.get("current_price")
        mv_base = item.get("market_value_base")
        mv_native = item.get("market_value")
        pnl_base = item.get("unrealized_pnl_base")
        pnl_pct = item.get("unrealized_pnl_percent")
        ytd_mkt_native = item.get("ytd_market_pnl")
        ytd_mkt_pct = item.get("ytd_market_pnl_percent")
        total_buy = item.get("total_buy_price")
        total_buy_base = item.get("total_buy_price_base")
        avg_cost = item.get("average_cost")

        def fmt_money(val):
            return f"{float(val):,.2f}" if val is not None else "-"

        def fmt_signed(val):
            return f"{float(val):+.2f}" if val is not None else "-"

        def colored(val):
            if val is None:
                return "-"
            color = "#23a55a" if float(val) >= 0 else "#f23f43"
            return f"<span style='color:{color};'>{fmt_signed(val)}</span>"

        qty_price_html = ""
        if qty is not None and price is not None:
            qty_price_html = f"<div style='margin-bottom:6px;'>{fmt_money(qty)} @ {fmt_money(price)} {currency}</div>"

        lines = [
            f"<div style='font-weight:600; margin-bottom:2px;'>{name}</div>",
            f"<div style='color:#666; font-size:12px; margin-bottom:6px;'>ISIN: {isin}</div>",
            qty_price_html,
            # For cash, prefer showing native currency value prominently, with base in parentheses
            (
                f"<div><span style='color:#666;'>Value:</span> <strong>{fmt_money(mv_native)} {currency}</strong>"
                f" <span style='color:#999;'>({fmt_money(mv_base)} {base_currency_code})</span></div>"
                if (item.get("instrument_type") == "cash" and mv_native is not None)
                else f"<div><span style='color:#666;'>Market Value:</span> <strong>{fmt_money(mv_base)} {base_currency_code}</strong></div>"
            ),
        ]
        # Show PnL line only when present and meaningful
        show_pnl = True
        if item.get("instrument_type") == "cash":
            # For cash, hide FX PnL when currency equals base (no FX exposure)
            if currency == base_currency_code:
                show_pnl = False
        if show_pnl and pnl_base is not None:
            lines.append(
                f"<div><span style='color:#666;'>Unrealized PnL:</span> {colored(pnl_base)} {base_currency_code} ({fmt_signed(pnl_pct)}%)</div>"
            )
        # YTD Market/Fx: for cash, compute from Jan 1 FX; for positions, use snapshots (if present)
        if item.get("instrument_type") == "cash":
            ytd_fx_base = item.get("ytd_fx_pnl_base")
            ytd_fx_pct = item.get("ytd_fx_percent")
            if ytd_fx_base is not None and currency != base_currency_code:
                ytd_line = (
                    f"<div style='font-size:13px; margin-top:4px;'><em>YTD FX:</em> "
                    f"{colored(ytd_fx_base)} {base_currency_code} ({fmt_signed(ytd_fx_pct)}%)</div>"
                )
            else:
                ytd_line = f"<div style='font-size:13px; margin-top:4px;'><em>YTD FX:</em> N/A</div>"
        else:
            # Positions: show snapshot-based market YTD in native
            if ytd_mkt_native is not None:
                ytd_line = (
                    f"<div style='font-size:13px; margin-top:4px;'><em>YTD Market:</em> "
                    f"{colored(ytd_mkt_native)} {currency} ({fmt_signed(ytd_mkt_pct)}%)</div>"
                )
            else:
                ytd_line = f"<div style='font-size:13px; margin-top:4px;'><em>YTD Market:</em> N/A</div>"
        lines.append(ytd_line)
        if total_buy is not None:
            qty_str = fmt_money(qty) if qty is not None else "-"
            avg_str = fmt_money(avg_cost) if avg_cost is not None else "-"
            lines.append(
                f"<div style='margin-top:4px;'><span style='color:#666;'>Total Buy:</span> {qty_str} @ {avg_str} {currency}"
                f" = {fmt_money(total_buy_base)} {base_currency_code}"
                f" <span style='color:#999;'>(native: {fmt_money(total_buy)} {currency})</span></div>"
            )
        # Purchase date
        latest_buy = item.get("latest_buy_date")
        if latest_buy:
            lines.append(
                f"<div style='color:#666; font-size:12px; margin-top:2px;'>Last Buy: {latest_buy}</div>"
            )

        card_html = (
            "<div style='border:1px solid #e6e6e6; border-radius:8px; padding:10px; margin-bottom:8px; font-size:14px;'>"
            + "".join([l for l in lines if l])
            + "</div>"
        )
        st.markdown(card_html, unsafe_allow_html=True)

    def _convert_to_base(
        self,
        portfolio_manager,
        amount: Decimal,
        from_currency_code: str,
        base_currency_code: str,
        allow_fetch: bool = False,
    ) -> Decimal:
        """Convert an amount from a currency code to portfolio base currency using provider rates."""
        if amount is None:
            return Decimal("0")
        if not from_currency_code or from_currency_code == base_currency_code:
            return Decimal(str(amount))
        if not allow_fetch:
            # Avoid network calls; return native amount (caller decides how to handle)
            return Decimal(str(amount))
        try:
            from src.portfolio.models import Currency

            # Try real-time FX first
            rate = portfolio_manager.data_manager.get_exchange_rate(
                Currency(from_currency_code), Currency(base_currency_code)
            )
            if rate:
                return Decimal(str(amount)) * rate

            # Fallback to historical FX for today (handles cases where live quote is missing)
            hist_rate = portfolio_manager.data_manager.get_historical_fx_rate_on(
                date.today(), Currency(from_currency_code), Currency(base_currency_code)
            )
            if hist_rate:
                return Decimal(str(amount)) * hist_rate
        except Exception:
            pass
        return Decimal(str(amount))

    def _get_cash_fx_summary(self, portfolio_manager) -> Dict:
        """Safely get cash FX summary even if the manager instance is from an older cache.

        Tries PortfolioManager.get_cash_fx_summary(); if missing, computes locally.
        Returns a dict keyed by Currency with fields similar to manager method.
        """
        # Try direct method if available
        if hasattr(portfolio_manager, "get_cash_fx_summary"):
            try:
                return portfolio_manager.get_cash_fx_summary()
            except Exception:
                pass

        # Local computation fallback
        portfolio = portfolio_manager.current_portfolio
        if not portfolio:
            return {}
        from src.portfolio.models import Currency as Cur
        from src.portfolio.models import TransactionType

        base = portfolio.base_currency
        foreign_balance: Dict[Cur, Decimal] = {}
        base_cost: Dict[Cur, Decimal] = {}

        for txn in sorted(portfolio.transactions, key=lambda t: t.timestamp):
            if txn.transaction_type not in [
                TransactionType.DEPOSIT,
                TransactionType.WITHDRAWAL,
            ]:
                continue
            cur = txn.currency
            amt = txn.total_value
            if cur not in foreign_balance:
                foreign_balance[cur] = Decimal("0")
                base_cost[cur] = Decimal("0")
            if cur == base:
                fx = Decimal("1")
            else:
                fx = (
                    portfolio_manager.data_manager.get_historical_fx_rate_on(
                        txn.timestamp.date(), cur, base
                    )
                    or portfolio_manager.data_manager.get_exchange_rate(cur, base)
                    or Decimal("1")
                )
            if txn.transaction_type == TransactionType.DEPOSIT:
                foreign_balance[cur] += amt
                base_cost[cur] += amt * fx
            else:
                existing_bal = foreign_balance[cur]
                existing_cost = base_cost[cur]
                avg_rate = (existing_cost / existing_bal) if existing_bal else fx
                foreign_balance[cur] = existing_bal - amt
                base_cost[cur] = existing_cost - (amt * avg_rate)
                if foreign_balance[cur].copy_abs() < Decimal("0.0000001"):
                    foreign_balance[cur] = Decimal("0")
                if base_cost[cur].copy_abs() < Decimal("0.0000001"):
                    base_cost[cur] = Decimal("0")

        result: Dict[Cur, Dict[str, Decimal]] = {}
        currencies = set(portfolio.cash_balances.keys()) | set(foreign_balance.keys())
        for cur in currencies:
            amt_foreign = portfolio.cash_balances.get(cur, Decimal("0"))
            cost_base = base_cost.get(cur, Decimal("0"))
            rate = (
                Decimal("1")
                if cur == base
                else (
                    portfolio_manager.data_manager.get_exchange_rate(cur, base)
                    or portfolio_manager.data_manager.get_historical_fx_rate_on(
                        date.today(), cur, base
                    )
                    or Decimal("1")
                )
            )
            current_value_base = amt_foreign * rate
            avg_cost_rate = (cost_base / amt_foreign) if amt_foreign else Decimal("0")
            fx_unrealized = current_value_base - cost_base
            result[cur] = {
                "foreign_amount": amt_foreign,
                "base_cost": cost_base,
                "current_rate": rate,
                "current_value_base": current_value_base,
                "fx_unrealized_pnl_base": fx_unrealized,
                "avg_cost_rate": avg_cost_rate,
            }
        return result

    def _classify_position(self, pos: Dict, instrument) -> str:
        """Classify a position into user-defined categories."""
        itype = pos.get("instrument_type", "").lower()
        name = (pos.get("name") or "").lower()
        symbol = (pos.get("symbol") or "").upper()

        # Short-term: cash or treasury bills by name hints
        if itype == "cash":
            return "Short Term"
        if itype == "bond":
            if any(hint in name for hint in ["treasury", "t-bill", "bill", "tbill"]):
                return "Short Term"
            return "Bonds"

        # Alternatives: crypto or gold
        if itype == "crypto":
            return "Alternatives"
        if any(hint in name for hint in ["gold", "bullion"]) or symbol in {
            "GLD",
            "IAU",
            "PHYS",
        }:
            return "Alternatives"

        # Equities default for stocks and ETFs
        if itype in {"stock", "etf"}:
            return "Equities"

        return "Miscellaneous"

    def _get_reference_price_for_date(
        self, portfolio_manager, symbol: str, ref_date: date
    ) -> Optional[Decimal]:
        """Get historical close price for the given date, with short lookahead if missing."""
        try:
            # Exact date
            prices = portfolio_manager.data_manager.get_historical_prices(
                symbol, ref_date, ref_date
            )
            if prices:
                pd0 = prices[0]
                return (
                    pd0.close_price or pd0.open_price or pd0.high_price or pd0.low_price
                )
            # Look ahead a few business days (first available of the year)
            end = ref_date + timedelta(days=5)
            prices = portfolio_manager.data_manager.get_historical_prices(
                symbol, ref_date, end
            )
            if prices:
                pd0 = prices[0]
                return (
                    pd0.close_price or pd0.open_price or pd0.high_price or pd0.low_price
                )
        except Exception:
            return None
        return None

    def render_transaction_form(self, portfolio_manager):
        """Render transaction input form."""
        with st.form("add_transaction"):
            col1, col2, col3 = st.columns(3)

            with col1:
                symbol = st.text_input("Symbol (optional)", placeholder="e.g., AAPL")
                isin = st.text_input(
                    "ISIN (optional)", placeholder="e.g., US0378331005"
                )
                transaction_type = st.selectbox(
                    "Type", ["buy", "sell", "dividend", "deposit", "withdrawal", "fees"]
                )

            with col2:
                quantity = st.number_input("Quantity/Amount", min_value=0.0, step=0.01)
                price = st.number_input("Price", min_value=0.0, step=0.01)

            with col3:
                trade_date = st.date_input("Date", value=date.today())
                from src.portfolio.models import Currency

                currency_code = st.selectbox(
                    "Currency",
                    [c.value for c in Currency],
                    index=[c.value for c in Currency].index(
                        portfolio_manager.current_portfolio.base_currency.value
                    ),
                )

            notes = st.text_area("Notes (optional)")

            if st.form_submit_button("Add Transaction"):
                if transaction_type in {"deposit", "withdrawal", "fees"}:
                    # For cash movements: ignore symbol/isin, use CASH and amount in price field
                    if price > 0:
                        try:
                            timestamp = datetime.combine(
                                trade_date, datetime.now().time()
                            )
                            from src.portfolio.models import Currency, TransactionType

                            txn_type_map = {
                                "deposit": TransactionType.DEPOSIT,
                                "withdrawal": TransactionType.WITHDRAWAL,
                                "fees": TransactionType.FEES,
                            }
                            success = portfolio_manager.add_transaction(
                                symbol="CASH",
                                transaction_type=txn_type_map[transaction_type],
                                quantity=Decimal("1"),
                                price=Decimal(str(price)),
                                timestamp=timestamp,
                                notes=notes if notes else None,
                                isin=None,
                                currency=Currency(currency_code),
                            )
                            if success:
                                st.success(
                                    f"Added {transaction_type} of {price:.2f} {currency_code}"
                                )
                                st.rerun()
                            else:
                                st.error("Failed to add cash transaction")
                        except Exception as e:
                            st.error(f"Error adding cash transaction: {e}")
                    else:
                        st.error("Please enter a positive amount for cash transactions")
                elif (symbol or isin) and quantity > 0 and price > 0:
                    try:
                        # Use selected trade date at current time
                        timestamp = datetime.combine(trade_date, datetime.now().time())

                        # Map transaction types
                        from src.portfolio.models import TransactionType

                        txn_type_map = {
                            "buy": TransactionType.BUY,
                            "sell": TransactionType.SELL,
                            "dividend": TransactionType.DIVIDEND,
                            "deposit": TransactionType.DEPOSIT,
                            "withdrawal": TransactionType.WITHDRAWAL,
                        }

                        success = portfolio_manager.add_transaction(
                            symbol=(symbol.upper() if symbol else isin.upper()),
                            transaction_type=txn_type_map[transaction_type],
                            quantity=Decimal(str(quantity)),
                            price=Decimal(str(price)),
                            timestamp=timestamp,
                            notes=notes if notes else None,
                            isin=(isin.upper() if isin else None),
                            # For non-cash trades, let instrument currency be used
                            currency=None,
                        )

                        if success:
                            label = symbol.upper() if symbol else isin.upper()
                            st.success(
                                f"Added {transaction_type} transaction: {quantity} {label} @ ${price}"
                            )
                            st.rerun()
                        else:
                            st.error("Failed to add transaction")

                    except Exception as e:
                        st.error(f"Error adding transaction: {e}")
                else:
                    st.error(
                        "Please provide a Symbol or ISIN, and ensure quantity/price are greater than 0"
                    )

    def render_analytics(self, portfolio_manager, metrics_calculator):
        """Render portfolio analytics and charts."""
        st.header("📈 Portfolio Analytics")

        if not st.session_state.portfolio_loaded:
            st.warning("Please load a portfolio to view analytics.")
            return

        # Analysis date range selection
        col1, col2, col3 = st.columns(3)
        with col1:
            default_range = (date.today() - timedelta(days=365), date.today())
            date_range = st.date_input("Analysis Range", value=default_range)
        with col2:
            benchmark = st.text_input("Benchmark Symbol", value="SPY")
        with col3:
            display_currency_code = st.selectbox(
                "Display Currency",
                [c.value for c in Currency],
                index=[c.value for c in Currency].index(
                    portfolio_manager.current_portfolio.base_currency.value
                ),
            )

        # Normalize date range
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = default_range

        if start_date > end_date:
            st.error("Start date must be on or before end date.")
            return
        snapshots = portfolio_manager.storage.load_snapshots(
            portfolio_manager.current_portfolio.id, start_date, end_date
        )

        if len(snapshots) < 2:
            st.warning(
                "Insufficient data for selected period. Please add more historical data or create snapshots."
            )
            return

        # Show snapshot data freshness
        if snapshots:
            latest_snapshot_date = max(snapshot.date for snapshot in snapshots)
            days_since_latest = (date.today() - latest_snapshot_date).days
            if days_since_latest > 1:
                st.warning(
                    f"⚠️ Latest snapshot is {days_since_latest} days old. Use 'Create Snapshots Since Last' to update."
                )
            else:
                st.success(
                    f"✅ Latest snapshot: {latest_snapshot_date.strftime('%Y-%m-%d')}"
                )

        # Calculate metrics using proper time-weighted methods from metrics calculator
        with st.spinner("Calculating time-weighted metrics..."):
            # Get external cash flows for the period
            external_cash_flows = portfolio_manager.get_external_cash_flows_by_day(
                start_date, end_date
            )

            # Convert cash flows to float for metrics calculator
            cash_flows_float = {d: float(v) for d, v in external_cash_flows.items()}

            # Calculate time-weighted returns using the metrics calculator
            portfolio_returns_twr = metrics_calculator.calculate_time_weighted_return(
                snapshots, cash_flows_float
            )

            if not portfolio_returns_twr:
                st.error("Could not calculate time-weighted returns")
                return

            # Calculate comprehensive metrics using the metrics calculator
            comprehensive_metrics = metrics_calculator.calculate_portfolio_metrics(
                snapshots, benchmark, cash_flows_float
            )

            if "error" in comprehensive_metrics:
                st.error(comprehensive_metrics["error"])
                return

            # Extract key metrics
            metrics = {
                "total_return_twr": comprehensive_metrics.get(
                    "total_return", 0.0
                ),  # Fixed: now uses actual total return
                "annualized_return_twr": comprehensive_metrics.get(
                    "annualized_return", 0.0
                ),  # Fixed: now uses actual annualized return
                "time_weighted_annualized_return": comprehensive_metrics.get(
                    "time_weighted_annualized_return", 0.0
                ),
                "modified_dietz_return": comprehensive_metrics.get(
                    "modified_dietz_return", 0.0
                ),
                "volatility": comprehensive_metrics.get("volatility", 0.0),
                "sharpe_ratio": comprehensive_metrics.get("sharpe_ratio", 0.0),
                "sortino_ratio": comprehensive_metrics.get("sortino_ratio", 0.0),
                "max_drawdown": comprehensive_metrics.get("max_drawdown", 0.0),
                "max_drawdown_duration": comprehensive_metrics.get(
                    "max_drawdown_duration", 0
                ),
                "var_5pct": comprehensive_metrics.get("var_5pct", 0.0),
                "cvar_5pct": comprehensive_metrics.get("cvar_5pct", 0.0),
                "calmar_ratio": comprehensive_metrics.get("calmar_ratio", 0.0),
                "beta": comprehensive_metrics.get("beta", 0.0),
                "alpha": comprehensive_metrics.get("alpha", 0.0),
                "information_ratio": comprehensive_metrics.get(
                    "information_ratio", 0.0
                ),
                "benchmark_return": comprehensive_metrics.get("benchmark_return", 0.0),
                "benchmark_volatility": comprehensive_metrics.get(
                    "benchmark_volatility", 0.0
                ),
                "benchmark_available": comprehensive_metrics.get(
                    "benchmark_available", False
                ),
            }

        # Display key metrics with emphasis on time-weighted returns
        st.subheader("📊 Performance Metrics (Time-Weighted)")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Return (TWR)", f"{metrics.get('total_return_twr', 0)*100:.2f}%"
            )
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")

        with col2:
            st.metric(
                "Annualized Return (TWR)",
                f"{metrics.get('annualized_return_twr', 0)*100:.2f}%",
            )
            st.metric("Volatility", f"{metrics.get('volatility', 0)*100:.2f}%")

        with col3:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.2f}%")
            st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.3f}")

        with col4:
            if metrics.get("benchmark_available"):
                st.metric("Beta", f"{metrics.get('beta', 0):.3f}")
                st.metric("Alpha", f"{metrics.get('alpha', 0)*100:.2f}%")
            else:
                st.info("Benchmark data not available")

        # Additional return metrics for comparison
        st.subheader("📈 Return Calculation Methods Comparison")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Period Total Return (TWR)",
                f"{metrics.get('total_return_twr', 0)*100:.2f}%",
            )
            st.caption("Total return for the selected period using geometric linking")

        with col2:
            st.metric(
                "Period Annualized Return (TWR)",
                f"{metrics.get('annualized_return_twr', 0)*100:.2f}%",
            )
            st.caption(
                "Annualized return for the selected period using geometric linking"
            )

        with col3:
            st.metric(
                "Time-Weighted Annualized Return",
                f"{metrics.get('time_weighted_annualized_return', 0)*100:.2f}%",
            )
            st.caption("Annualized return using dedicated TWR methodology")

        # Additional time-weighted metrics
        st.subheader("🔄 Time-Weighted Return Details")
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Modified Dietz Return",
                f"{metrics.get('modified_dietz_return', 0)*100:.2f}%",
            )
            st.metric("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.3f}")

        with col2:
            st.metric("Value at Risk (5%)", f"{metrics.get('var_5pct', 0)*100:.3f}%")
            st.metric("Conditional VaR (5%)", f"{metrics.get('cvar_5pct', 0)*100:.3f}%")

        # Comparison between TWR and MWR methodologies
        st.subheader("📊 Return Methodology Comparison")

        # Calculate money-weighted returns for comparison
        try:
            portfolio_returns_mwr = metrics_calculator.calculate_money_weighted_return(
                snapshots, cash_flows_float
            )
            mwr_annualized = (
                metrics_calculator.calculate_annualized_money_weighted_return(
                    snapshots, cash_flows_float
                )
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Time-Weighted Return (TWR)",
                    f"{metrics.get('annualized_return_twr', 0)*100:.2f}%",
                )
                st.caption(
                    "Eliminates cash flow impact - measures pure investment performance"
                )

            with col2:
                st.metric("Money-Weighted Return (MWR)", f"{mwr_annualized*100:.2f}%")
                st.caption(
                    "Includes cash flow timing - shows investor's actual experience"
                )

            with col3:
                difference = metrics.get("annualized_return_twr", 0) - mwr_annualized
                st.metric("TWR vs MWR Difference", f"{difference*100:.2f}%")
                if abs(difference) > 0.01:  # 1% threshold
                    if difference > 0:
                        st.caption("TWR > MWR: Cash flows helped performance")
                    else:
                        st.caption("TWR < MWR: Cash flows hurt performance")
                else:
                    st.caption("TWR ≈ MWR: Minimal cash flow impact")

        except Exception as e:
            st.warning(
                f"Could not calculate money-weighted returns for comparison: {e}"
            )
            st.info(
                "Time-weighted returns are shown above. Money-weighted returns require additional data processing."
            )

        # YTD Time-Weighted Performance
        try:
            ytd_start = date(date.today().year, 1, 1)
            if end_date < ytd_start:
                ytd_start = end_date  # guard
            ytd_snaps = portfolio_manager.storage.load_snapshots(
                portfolio_manager.current_portfolio.id, ytd_start, end_date
            )

            if len(ytd_snaps) >= 2:
                ytd_flows = portfolio_manager.get_external_cash_flows_by_day(
                    ytd_start, end_date
                )
                ytd_flows_f = {d: float(v) for d, v in ytd_flows.items()}

                # Calculate YTD time-weighted return using the same method as Portfolio tab
                daily_returns = metrics_calculator.calculate_time_weighted_return(
                    ytd_snaps, ytd_flows_f
                )

                if daily_returns:
                    # Geometric aggregation for period return (same as Portfolio tab)
                    twr = 1.0
                    for r in daily_returns:
                        twr *= 1.0 + r
                    ytd_perf_pct = (twr - 1.0) * 100.0
                    st.metric("YTD Performance (TWR)", f"{ytd_perf_pct:.2f}%")
                else:
                    st.metric("YTD Performance (TWR)", "N/A")
            else:
                st.metric("YTD Performance (TWR)", "N/A")
        except Exception:
            st.metric("YTD Performance (TWR)", "N/A")

        # Prepare benchmark series (aligned to snapshot dates)
        bench_map: Dict[date, float] = {}
        try:
            price_data = portfolio_manager.data_manager.get_historical_prices(
                benchmark, start_date, end_date
            )
            for pd_item in price_data:
                if pd_item.close_price:
                    bench_map[pd_item.date] = float(pd_item.close_price)
        except Exception:
            bench_map = {}

        # Align benchmark prices to snapshot dates with forward-fill
        bench_prices_aligned: List[Optional[float]] = []
        last_px: Optional[float] = None
        for s in snapshots:
            px = bench_map.get(s.date, last_px)
            bench_prices_aligned.append(px)
            if px is not None:
                last_px = px

        # Portfolio value chart with category overlays and benchmark in selected currency
        self.plot_portfolio_and_categories(
            snapshots,
            display_currency_code,
            portfolio_manager,
            start_date,
            end_date,
            benchmark_symbol=benchmark,
            benchmark_prices_aligned=bench_prices_aligned,
            portfolio_returns_twr=portfolio_returns_twr,  # Pass TWR returns for plotting
        )

        # Risk metrics
        st.subheader("⚠️ Risk Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Value at Risk (5%)", f"{metrics.get('var_5pct', 0)*100:.3f}%")
            st.metric("Conditional VaR (5%)", f"{metrics.get('cvar_5pct', 0)*100:.3f}%")

        with col2:
            st.metric("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.3f}")
            if metrics.get("benchmark_available"):
                st.metric(
                    "Information Ratio", f"{metrics.get('information_ratio', 0):.3f}"
                )

        # Add explanation of time-weighted methodology
        st.info(
            "💡 **Time-Weighted Returns (TWR)**: All performance metrics above use time-weighted returns, "
            "which eliminate the impact of external cash flows (deposits/withdrawals) to measure pure investment performance. "
            "This applies to both the overall portfolio and individual asset categories, providing consistent and "
            "comparable performance measurement across all components."
        )

        # Daily time-weighted returns chart
        if portfolio_returns_twr:
            st.subheader("📈 Daily Time-Weighted Returns")

            # Convert returns to percentages for better readability
            daily_returns_pct = [r * 100 for r in portfolio_returns_twr]

            # Create dates list from snapshots (skip first date since returns start from second snapshot)
            dates = [s.date for s in snapshots]

            fig3 = go.Figure()
            fig3.add_trace(
                go.Scatter(
                    x=dates[1:],  # Skip first date since returns start from second snapshot
                    y=daily_returns_pct,
                    mode="lines+markers",
                    name="Daily TWR",
                    line=dict(color="#1f77b4", width=1),
                    marker=dict(size=3),
                )
            )

            # Add zero line for reference
            fig3.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                annotation_text="Zero Return Line",
            )

            fig3.update_layout(
                title="Daily Time-Weighted Returns Over Time",
                xaxis_title="Date",
                yaxis_title="Daily Return (%)",
                height=300,
                showlegend=True,
            )
            fig3.update_xaxes(range=[start_date, end_date])
            st.plotly_chart(fig3, use_container_width=True)

            # Summary statistics for daily returns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Day", f"{max(daily_returns_pct):.2f}%")
            with col2:
                st.metric("Worst Day", f"{min(daily_returns_pct):.2f}%")
            with col3:
                st.metric("Avg Daily Return", f"{np.mean(daily_returns_pct):.2f}%")
            with col4:
                st.metric(
                    "Positive Days",
                    f"{sum(1 for r in daily_returns_pct if r > 0)}/{len(daily_returns_pct)}",
                )

    def plot_portfolio_and_categories(
        self,
        snapshots,
        display_currency_code: str,
        portfolio_manager,
        start_date: date,
        end_date: date,
        benchmark_symbol: Optional[str] = None,
        benchmark_prices_aligned: Optional[List[Optional[float]]] = None,
        portfolio_returns_twr: Optional[float] = None,
    ):
        """Plot portfolio and category series in selected currency, plus cumulative returns."""
        if len(snapshots) < 2:
            return

        # FX rate cache and last-known fallback per pair
        fx_cache: Dict[tuple, Optional[Decimal]] = {}
        fx_last_rate: Dict[tuple, Optional[Decimal]] = {}

        def get_rate(day: date, from_code: str, to_code: str) -> Optional[Decimal]:
            if from_code == to_code:
                return Decimal("1")
            key = (day, from_code, to_code)
            if key in fx_cache:
                return fx_cache[key]
            from src.portfolio.models import Currency as Cur

            try:
                rate = portfolio_manager.data_manager.get_historical_fx_rate_on(
                    day, Cur(from_code), Cur(to_code)
                )
            except Exception:
                rate = None
            fx_cache[key] = rate
            pair_key = (from_code, to_code)
            if rate is not None:
                fx_last_rate[pair_key] = rate
            else:
                # Fallback to last known rate for this pair if available
                last = fx_last_rate.get(pair_key)
                if last is not None:
                    rate = last
                    fx_cache[key] = rate
            return rate

        # Category classifier (reuse UI logic)
        def classify(instrument) -> str:
            itype = (
                instrument.instrument_type.value
                if hasattr(instrument.instrument_type, "value")
                else str(instrument.instrument_type)
            )
            itype = itype.lower()
            name = (instrument.name or "").lower()
            symbol = (instrument.symbol or "").upper()
            if itype == "cash":
                return "Short Term"
            if itype == "bond":
                if any(h in name for h in ["treasury", "t-bill", "bill", "tbill"]):
                    return "Short Term"
                return "Bonds"
            if itype == "crypto":
                return "Alternatives"
            if any(h in name for h in ["gold", "bullion"]) or symbol in {
                "GLD",
                "IAU",
                "PHYS",
            }:
                return "Alternatives"
            if itype in {"stock", "etf"}:
                return "Equities"
            return "Miscellaneous"

        dates: List[date] = [s.date for s in snapshots]
        # Portfolio line in display currency
        portfolio_values: List[float] = []
        base_code = (
            snapshots[0].base_currency.value
            if hasattr(snapshots[0].base_currency, "value")
            else str(snapshots[0].base_currency)
        )
        for s in snapshots:
            val = Decimal(str(s.total_value))
            rate = get_rate(s.date, base_code, display_currency_code)
            if rate is not None:
                val = val * rate
            portfolio_values.append(float(val))

        # Category lines in display currency
        categories = [
            "Short Term",
            "Bonds",
            "Equities",
            "Alternatives",
            "Miscellaneous",
        ]
        cat_series: Dict[str, List[float]] = {c: [] for c in categories}

        for s in snapshots:
            # init per-snapshot sums
            sums = {c: Decimal("0") for c in categories}
            for pos in s.positions.values():
                if pos.market_value is None:
                    continue
                instr = pos.instrument
                cat = classify(instr)
                from_code = (
                    instr.currency.value
                    if hasattr(instr.currency, "value")
                    else str(instr.currency)
                )
                val = Decimal(str(pos.market_value))
                # convert directly to display currency
                rate = get_rate(s.date, from_code, display_currency_code)
                if rate is not None:
                    val = val * rate
                sums[cat] += val
            # append floats
            for c in categories:
                cat_series[c].append(float(sums[c]))

        # Plot portfolio and categories
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=portfolio_values,
                mode="lines",
                name="Portfolio",
                line=dict(width=2),
            )
        )
        color_map = {
            "Short Term": "#1f77b4",
            "Bonds": "#ff7f0e",
            "Equities": "#2ca02c",
            "Alternatives": "#d62728",
            "Miscellaneous": "#9467bd",
        }
        for c in categories:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=cat_series[c],
                    mode="lines",
                    name=c,
                    line=dict(color=color_map.get(c)),
                )
            )

        # Add benchmark to value chart (scaled to start portfolio value for comparability)
        if benchmark_symbol and benchmark_prices_aligned:
            # Create scaled series: benchmark normalized to its first non-None, then scaled to portfolio initial value
            first_bench = next(
                (p for p in benchmark_prices_aligned if p is not None), None
            )
            if first_bench and portfolio_values:
                scaled = []
                for p in benchmark_prices_aligned:
                    if p is None:
                        scaled.append(None)
                    else:
                        scaled.append(portfolio_values[0] * (p / first_bench))
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=scaled,
                        mode="lines",
                        name=f"{benchmark_symbol} (scaled)",
                        line=dict(color="#555", dash="dash"),
                    )
                )

        fig.update_layout(
            title=f"Portfolio Value Over Time (in {display_currency_code}) - Time-Weighted Analysis",
            xaxis_title="Date",
            yaxis_title=f"Value ({display_currency_code})",
            height=420,
            showlegend=True,
        )
        fig.update_xaxes(range=[start_date, end_date])
        st.plotly_chart(fig, use_container_width=True)

        # Cumulative returns for portfolio and categories (using time-weighted returns)
        def to_cum_returns_twr(returns: List[float]) -> List[Optional[float]]:
            """Calculate cumulative returns from time-weighted daily returns."""
            if not returns:
                return []

            # Calculate cumulative return using geometric linking: (1+r1)*(1+r2)*...*(1+rn) - 1
            cumulative = []
            running_product = 1.0

            for daily_return in returns:
                running_product *= 1 + daily_return
                cumulative_return = (running_product - 1.0) * 100.0
                cumulative.append(cumulative_return)

            return cumulative

        def to_cum_returns(values: List[float]) -> List[Optional[float]]:
            """Calculate cumulative returns from absolute values (for categories)."""
            if not values:
                return []
            # Find first non-zero starting point
            start_idx = None
            for i, v in enumerate(values):
                if abs(v) > 1e-12:
                    start_idx = i
                    break
            if start_idx is None:
                return [None for _ in values]
            base = values[start_idx]
            if abs(base) <= 1e-12:
                return [None for _ in values]
            out: List[Optional[float]] = [None] * start_idx
            for v in values[start_idx:]:
                out.append(((v / base) - 1.0) * 100.0)
            return out

        # Portfolio cumulative returns using TWR
        portfolio_cum_returns = (
            to_cum_returns_twr(portfolio_returns_twr) if portfolio_returns_twr else []
        )

        fig2 = go.Figure()
        if portfolio_cum_returns:
            fig2.add_trace(
                go.Scatter(
                    x=dates[1:],  # Skip first date since returns start from second snapshot
                    y=portfolio_cum_returns,
                    mode="lines",
                    name="Portfolio (TWR)",
                    line=dict(width=2, color="#000"),
                )
            )

        # Category cumulative returns (using time-weighted returns for consistency)
        for c in categories:
            # Calculate daily returns for this category using TWR methodology
            cat_returns = []
            for i in range(1, len(cat_series[c])):
                prev_val = cat_series[c][i - 1]
                curr_val = cat_series[c][i]

                # Apply TWR formula: (V_t - V_{t-1} - External_CF_t) / V_{t-1}
                # For categories, we assume no external cash flows (they're internal portfolio movements)
                if prev_val > 0:
                    daily_return = (curr_val - prev_val) / prev_val
                    cat_returns.append(daily_return)
                else:
                    cat_returns.append(0.0)

            # Calculate cumulative TWR returns for the category
            cat_cum_returns = to_cum_returns_twr(cat_returns) if cat_returns else []

            if cat_cum_returns:
                fig2.add_trace(
                    go.Scatter(
                        x=dates[1:],  # Skip first date since returns start from second snapshot
                        y=cat_cum_returns,
                        mode="lines",
                        name=f"{c} (TWR)",
                        line=dict(color=color_map.get(c)),
                    )
                )

        # Add benchmark cumulative returns
        if benchmark_symbol and benchmark_prices_aligned:
            first_bench = next(
                (p for p in benchmark_prices_aligned if p is not None), None
            )
            if first_bench:
                bench_cum = []
                for p in benchmark_prices_aligned:
                    if p is None:
                        bench_cum.append(None)
                    else:
                        bench_cum.append(((p / first_bench) - 1.0) * 100.0)
                fig2.add_trace(
                    go.Scatter(
                        x=dates,
                        y=bench_cum,
                        mode="lines",
                        name=f"{benchmark_symbol} (cum %)",
                        line=dict(color="#888", dash="dot"),
                    )
                )

        fig2.update_layout(
            title=f"Cumulative Returns - All Series (Time-Weighted) in {display_currency_code}",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            height=320,
            showlegend=True,
        )
        fig2.update_xaxes(range=[start_date, end_date])
        st.plotly_chart(fig2, use_container_width=True)

    def render_settings(self):
        """Render settings and configuration."""
        st.header("⚙️ Settings")

        st.subheader("📊 Data Providers")
        st.write("Current data providers:")
        st.write("- Yahoo Finance (Free)")
        st.write("- Alpha Vantage (API Key required)")

        st.info(
            "💡 **Data Fetching Policy**: Prices and market data are only fetched when you explicitly click 'Update Current Prices' in the sidebar. This helps control API usage and ensures you only get fresh data when needed."
        )

        st.subheader("💾 Data Management")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Clear Cache"):
                st.success("Cache cleared")

        with col2:
            if st.button("Export Data"):
                st.info("Export functionality would be implemented here")

        st.subheader("📊 Snapshot Management")
        st.info(
            "Update portfolio snapshots with current market data and create historical snapshots."
        )

        # Get current portfolio
        portfolio_manager = self.initialize_components()[0]
        if (
            portfolio_manager
            and hasattr(portfolio_manager, "current_portfolio")
            and portfolio_manager.current_portfolio
        ):
            portfolio_id = portfolio_manager.current_portfolio.id
            portfolio_name = portfolio_manager.current_portfolio.name

            st.write(f"**Current Portfolio:** {portfolio_name}")

            # Days to create snapshots for (support up to 1 year)
            days_to_snapshot = st.slider(
                "Days of historical snapshots to create", 30, 365, 365
            )

            if st.button("🔄 Update Portfolio Snapshots", type="primary"):
                with st.spinner("Updating snapshots..."):
                    try:
                        from src.utils.initializer import PortfolioInitializer

                        initializer = PortfolioInitializer()

                        # Update snapshots
                        result = initializer.update_portfolio_snapshots(
                            portfolio_id, days_to_snapshot
                        )

                        if result.get("success"):
                            st.success(f"✅ Successfully updated snapshots!")
                            st.write(
                                f"**Snapshots created:** {result['snapshots_created']}"
                            )
                            st.write(
                                f"**Failed snapshots:** {result['failed_snapshots']}"
                            )
                            st.write(
                                f"**Current portfolio value:** ${result['current_value']:,.2f}"
                            )

                            # Show price update results
                            price_results = result.get("price_update_results", {})
                            if price_results:
                                st.write("**Price update results:**")
                                for symbol, success in price_results.items():
                                    status = "✅" if success else "❌"
                                    st.write(f"  {status} {symbol}")
                        else:
                            st.error(
                                f"❌ Failed to update snapshots: {result.get('error', 'Unknown error')}"
                            )

                    except Exception as e:
                        st.error(f"❌ Error updating snapshots: {e}")
        else:
            st.warning(
                "No portfolio loaded. Please select a portfolio in the sidebar first."
            )

        st.subheader("ℹ️ About")
        st.markdown(
            """
        **AI Portfolio Tracker** v1.0

        A comprehensive portfolio management system with AI-powered investment advice.

        Features:
        - Multi-currency portfolio tracking
        - Real-time price updates
        - Advanced financial metrics
        - AI-powered investment advice
        - Transaction management
        - Performance analytics

        Built with: Python, Streamlit, LangChain, Plotly
        """
        )


def main():
    """Main function to run the Streamlit app."""
    app = PortfolioTrackerUI()
    app.run()


if __name__ == "__main__":
    main()
