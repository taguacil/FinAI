"""
Streamlit UI for the Portfolio Tracker with AI Agent.
"""

import logging
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
from src.services.market_data_service import MarketDataService
from src.utils.logging_config import setup_logging
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
            # Setup logging
            setup_logging(log_level="INFO", app_name="portfolio-tracker-ui")

            storage = FileBasedStorage()
            data_provider = DataProviderManager()

            # Create MarketDataService wrapping the DataProviderManager
            market_data_service = MarketDataService(data_provider)

            # Pass MarketDataService to PortfolioManager
            portfolio_manager = PortfolioManager(storage, market_data_service)

            # Metrics calculator uses the underlying DataProviderManager
            metrics_calculator = FinancialMetricsCalculator(data_provider)

            agent = PortfolioAgent(
                portfolio_manager=portfolio_manager,
                data_manager=market_data_service,
                metrics_calculator=metrics_calculator,
            )

            return portfolio_manager, agent, metrics_calculator, market_data_service
        except Exception as e:
            st.error(f"Error initializing components: {e}")
            return None, None, None, None

    def run(self):
        """Run the Streamlit app."""
        # Initialize components
        result = self.initialize_components()
        portfolio_manager, agent, metrics_calculator, market_data_service = result

        if not all([portfolio_manager, agent, metrics_calculator, market_data_service]):
            st.error("Failed to initialize application components.")
            return

        # Store market_data_service in session state for access in other methods
        if "market_data_service" not in st.session_state:
            st.session_state.market_data_service = market_data_service

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
        self.render_sidebar(portfolio_manager, market_data_service)

        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["💬 AI Chat", "📊 Portfolio", "📈 Analytics", "🔮 Scenarios", "⚙️ Settings"]
        )

        with tab1:
            self.render_chat_interface(agent)

        with tab2:
            self.render_portfolio_overview(portfolio_manager, market_data_service)

        with tab3:
            self.render_analytics(portfolio_manager, metrics_calculator)

        with tab4:
            self.render_scenarios(portfolio_manager, metrics_calculator)

        with tab5:
            self.render_settings()

    def init_session_state(self):
        """Initialize session state variables."""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "portfolio_loaded" not in st.session_state:
            st.session_state.portfolio_loaded = False
        if "selected_portfolio" not in st.session_state:
            st.session_state.selected_portfolio = None

    def render_sidebar(self, portfolio_manager, market_data_service):
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

            # Data freshness indicator (compact)
            if market_data_service is not None:
                freshness = market_data_service.freshness
                freshness_text = freshness.freshness_display

                if freshness.is_stale:
                    st.sidebar.caption(f"📡 Data: ⚠️ {freshness_text}")
                else:
                    st.sidebar.caption(f"📡 Data: ✅ {freshness_text}")

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

        # Quick Actions - prominent buttons at the top
        st.markdown("---")
        action_col1, action_col2, action_col3, action_col4 = st.columns(4)

        with action_col1:
            if st.button("📊 Portfolio Summary", use_container_width=True):
                if st.session_state.portfolio_loaded:
                    response = agent.chat("Please show me my current portfolio summary")
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                    st.rerun()
                else:
                    st.warning("Please load a portfolio first")

        with action_col2:
            if st.button("📈 Performance Analysis", use_container_width=True):
                if st.session_state.portfolio_loaded:
                    response = agent.analyze_portfolio_performance()
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                    st.rerun()
                else:
                    st.warning("Please load a portfolio first")

        with action_col3:
            if st.button("💡 Investment Ideas", use_container_width=True):
                if st.session_state.portfolio_loaded:
                    response = agent.chat("Based on my current portfolio, what investment opportunities should I consider?")
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response}
                    )
                    st.rerun()
                else:
                    st.warning("Please load a portfolio first")

        with action_col4:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                agent.clear_conversation()
                st.rerun()

        # Advanced Options - collapsible section
        with st.expander("⚙️ Advanced Options", expanded=False):
            adv_col1, adv_col2 = st.columns(2)

            # PDF Analysis
            with adv_col1:
                st.markdown("**📄 Document Analysis**")
                uploaded_pdf = st.file_uploader(
                    "Upload PDF (datasheet, prospectus, etc.)",
                    type=["pdf"],
                    accept_multiple_files=False,
                    key="pdf_uploader"
                )
                if uploaded_pdf is not None:
                    st.caption(
                        f"Selected: {uploaded_pdf.name} ({uploaded_pdf.size/1024:.1f} KB)"
                    )
                    pdf_bytes = uploaded_pdf.read()

                    pdf_btn_col1, pdf_btn_col2 = st.columns(2)
                    with pdf_btn_col1:
                        if st.button("📎 Attach to Chat", use_container_width=True):
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

                    with pdf_btn_col2:
                        if st.button("🔍 Analyze Now", use_container_width=True):
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

            # AI Model Selection
            with adv_col2:
                st.markdown("**🤖 AI Model Configuration**")
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
                model_choice = st.selectbox(
                    "Select Model",
                    all_models,
                    format_func=lambda x: x[0],
                    key="model_selector"
                )
                if st.button("Apply Model", use_container_width=True):
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

    def render_portfolio_overview(self, portfolio_manager, market_data_service):
        """Render portfolio overview with transactions."""
        st.header("📊 Portfolio Overview")

        if not st.session_state.portfolio_loaded:
            st.warning("Please load or create a portfolio to view details.")
            return

        # Portfolio summary
        portfolio = portfolio_manager.current_portfolio

        total_value = portfolio_manager.get_portfolio_value()
        positions = portfolio_manager.get_position_summary()

        # Initialize variables (will be populated if positions exist)
        enriched = []
        base_currency = portfolio.base_currency
        start_of_year = date(date.today().year, 1, 1)

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
        positions_with_current_prices = [
            pos for pos in positions if pos.get("has_current_price", False)
        ]
        positions_without_any_price = [
            pos for pos in positions
            if not pos.get("has_current_price", False)
        ]

        if positions_without_any_price:
            st.warning(
                f"⚠️ {len(positions_without_any_price)} positions have no price data. Use 'Update Current Prices' to fetch latest data."
            )

        if positions_with_current_prices:
            # Find the most recent price update
            latest_update = max(
                (
                    pos.get("last_updated")
                    for pos in positions_with_current_prices
                    if pos.get("last_updated")
                ),
                default=None,
            )
            if latest_update:
                st.info(
                    f"📅 Latest price update: {latest_update.strftime('%Y-%m-%d %H:%M')}"
                )

        # Data status row
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            latest_snapshot = portfolio_manager.storage.get_latest_snapshot(portfolio.id)
            if latest_snapshot:
                st.caption(f"🗓️ Latest snapshot: {latest_snapshot.date.isoformat()}")

        with status_col2:
            if market_data_service is not None:
                freshness = market_data_service.freshness
                freshness_text = freshness.freshness_display
                if freshness.is_stale:
                    st.caption(f"📡 Prices: ⚠️ {freshness_text}")
                else:
                    st.caption(f"📡 Prices: ✅ {freshness_text}")

        # Data update controls
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
        with col1:
            days_to_update = st.selectbox(
                "Period",
                [7, 14, 30, 60, 90, 180, 365],
                index=2,  # Default to 30 days
                help="Number of days back to update with fresh market data"
            )

        with col2:
            if st.button("📈 Update Snapshots", help="Fetch prices and create/update historical snapshots", type="primary"):
                with st.spinner(f"Updating snapshots for last {days_to_update} days..."):
                    try:
                        end_date = date.today()
                        start_date = end_date - timedelta(days=days_to_update)
                        logging.info(f"Updating snapshots from {start_date} to {end_date}")
                        refreshed = portfolio_manager.create_snapshots_for_range(start_date, end_date, save=True)
                        # Update freshness tracking
                        if market_data_service and portfolio_manager.current_portfolio:
                            market_data_service.refresh_all(portfolio_manager.current_portfolio)
                        st.success(f"✅ Updated {len(refreshed)} snapshots")
                        st.rerun()
                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        logging.error(f"Failed to update market data: {error_details}")
                        st.error(f"❌ Failed to update: {str(e)}")

        with col3:
            if st.button("💰 Quick Refresh", help="Update current prices only (faster, no snapshots)"):
                with st.spinner("Refreshing prices..."):
                    if market_data_service and portfolio_manager.current_portfolio:
                        result = market_data_service.refresh_all(portfolio_manager.current_portfolio)
                        st.success(f"✅ {result.symbols_updated} prices updated")
                        st.rerun()
                    else:
                        price_results = portfolio_manager.update_current_prices()
                        success_count = sum(price_results.values())
                        st.success(f"✅ {success_count} prices updated")
                        st.rerun()

        with col4:
            st.caption("💡 **Snapshots**: Updates historical data for analytics. **Quick Refresh**: Just updates current prices.")

        # Key metrics - styled container
        st.markdown("### 📈 Portfolio Summary")

        # First row: Core metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.metric("💰 Total Value", f"${total_value:,.2f}")

        with metric_col2:
            total_positions = len(positions)
            st.metric("📋 Positions", total_positions)

        with metric_col3:
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
            st.metric("💵 Cash (base)", f"${cash_total_base:,.2f}")

        with metric_col4:
            # Calculate total P&L
            total_pnl = sum(
                float(pos.get("unrealized_pnl", 0) or 0) for pos in positions
            )
            # Color-coded P&L metric - the delta parameter automatically colors positive green and negative red
            pnl_icon = "📈" if total_pnl >= 0 else "📉"
            st.metric(f"{pnl_icon} Unrealized P&L", f"${total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")

        # Additional metrics: YTD Performance (TWR) and Unrealized P&L (%)
        ytd_col1, ytd_col2 = st.columns(2)

        # YTD performance (time-weighted): remove external cash flows
        try:
            portfolio_id = portfolio_manager.current_portfolio.id
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

        # Positions by category (table layout)
        if positions:
            st.subheader("📈 Current Positions")

            # Build enriched rows with ISIN, base market value, category, and YTD PnL
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

            # Render by category using separate tables
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

                # Render compact table for this category
                self._render_category_table(items, base_currency.value, portfolio, portfolio_manager)

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

        # Transactions Section
        st.markdown("---")
        st.subheader("📝 Transactions")

        trans_tab1, trans_tab2 = st.tabs(["Recent Transactions", "➕ Add New"])

        with trans_tab1:
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

        with trans_tab2:
            self.render_transaction_form(portfolio_manager)

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

    def _convert_snapshots_to_currency(
        self, snapshots, target_currency_code: str, portfolio_manager
    ):
        """Convert portfolio snapshots to a different currency."""
        if not snapshots:
            return snapshots

        from src.portfolio.models import Currency, PortfolioSnapshot

        converted_snapshots = []

        for snapshot in snapshots:
            # Get the base currency of the snapshot
            base_currency_code = (
                snapshot.base_currency.value
                if hasattr(snapshot.base_currency, "value")
                else str(snapshot.base_currency)
            )

            # Skip conversion if currencies are the same
            if base_currency_code == target_currency_code:
                converted_snapshots.append(snapshot)
                continue

            # Get exchange rate for this date
            try:
                from_currency = Currency(base_currency_code)
                to_currency = Currency(target_currency_code)

                # Try to get historical rate for the snapshot date
                rate = portfolio_manager.data_manager.get_historical_fx_rate_on(
                    snapshot.date, from_currency, to_currency
                )

                # Fallback to current rate if historical rate not available
                if not rate:
                    rate = portfolio_manager.data_manager.get_exchange_rate(
                        from_currency, to_currency
                    )

                # Default to 1.0 if no rate available
                if not rate:
                    rate = Decimal("1.0")

                # Convert all monetary values
                # Use model_copy to avoid Pydantic validation issues with existing Position objects
                converted_snapshot = snapshot.model_copy(update={
                    'total_value': snapshot.total_value * rate,
                    'cash_balance': snapshot.cash_balance * rate,
                    'positions_value': snapshot.positions_value * rate,
                    'base_currency': Currency(target_currency_code),
                    'cash_balances': {Currency(target_currency_code): snapshot.cash_balance * rate},
                    'total_cost_basis': snapshot.total_cost_basis * rate,
                    'total_unrealized_pnl': snapshot.total_unrealized_pnl * rate,
                    # positions and total_unrealized_pnl_percent stay the same
                })

                converted_snapshots.append(converted_snapshot)

            except Exception as e:
                # If conversion fails, use original snapshot
                st.warning(f"Failed to convert snapshot for {snapshot.date}: {e}")
                converted_snapshots.append(snapshot)

        return converted_snapshots

    def _convert_cash_flows_to_currency(
        self, cash_flows: dict, from_currency_code: str, target_currency_code: str, portfolio_manager
    ):
        """Convert cash flows from one currency to another."""
        if from_currency_code == target_currency_code:
            return cash_flows

        if not cash_flows:
            return cash_flows

        from src.portfolio.models import Currency

        converted_cash_flows = {}

        for flow_date, flow_amount in cash_flows.items():
            try:
                from_currency = Currency(from_currency_code)
                to_currency = Currency(target_currency_code)

                # Get exchange rate for the cash flow date
                rate = portfolio_manager.data_manager.get_historical_fx_rate_on(
                    flow_date, from_currency, to_currency
                )

                # Fallback to current rate if historical rate not available
                if not rate:
                    rate = portfolio_manager.data_manager.get_exchange_rate(
                        from_currency, to_currency
                    )

                # Default to 1.0 if no rate available
                if not rate:
                    rate = Decimal("1.0")

                converted_cash_flows[flow_date] = flow_amount * float(rate)

            except Exception as e:
                # If conversion fails, use original amount
                st.warning(f"Failed to convert cash flow for {flow_date}: {e}")
                converted_cash_flows[flow_date] = flow_amount

        return converted_cash_flows

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

        # Short-term: cash only
        if itype == "cash":
            return "Short Term"
        if itype == "bond":
            # All bonds go to Bonds category, regardless of type
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

        # Analysis Configuration Panel
        with st.container():
            st.markdown("#### ⚙️ Analysis Configuration")
            config_col1, config_col2, config_col3, config_col4 = st.columns([2, 1, 1, 1])

            with config_col1:
                default_range = (date.today() - timedelta(days=365), date.today())
                date_range = st.date_input("📅 Analysis Period", value=default_range)

            with config_col2:
                benchmark = st.text_input("📊 Benchmark", value="SPY", help="Compare against a market index")

            with config_col3:
                display_currency_code = st.selectbox(
                    "💱 Currency",
                    [c.value for c in Currency],
                    index=[c.value for c in Currency].index(
                        portfolio_manager.current_portfolio.base_currency.value
                    ),
                    help="Display all values in this currency"
                )

            with config_col4:
                # Quick period selectors
                st.markdown("**Quick Select**")
                period_options = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365, "2Y": 730}
                if st.selectbox(
                    "Period",
                    list(period_options.keys()),
                    index=3,  # 1Y
                    key="quick_period",
                    label_visibility="collapsed"
                ):
                    # This is just for display; the actual date_range input is used
                    pass

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
                # Check if this is due to portfolio starting from zero
                nonzero_snapshots = [s for s in snapshots if float(s.total_value) > 0]
                if len(nonzero_snapshots) < 2:
                    st.warning("⚠️ Cannot calculate time-weighted returns: Portfolio needs at least 2 snapshots with positive values.")
                    st.info("💡 This typically happens when:")
                    st.info("   • Portfolio is just starting (only recent snapshots have value)")
                    st.info("   • Not enough historical data has been captured")
                    st.info("   • Portfolio values are zero or negative")

                    # Show snapshot values for context
                    st.write("**Snapshot values:**")
                    for i, snapshot in enumerate(snapshots[:5]):  # Show first 5 snapshots
                        st.write(f"  {snapshot.date}: ${float(snapshot.total_value):,.2f}")
                    if len(snapshots) > 5:
                        st.write(f"  ... and {len(snapshots) - 5} more snapshots")
                else:
                    st.error(f"Could not calculate time-weighted returns for technical reasons.")
                    st.error(f"Snapshots: {len(snapshots)}, Date range: {snapshots[0].date} to {snapshots[-1].date}")
                    st.error(f"Cash flows: {len(cash_flows_float)} entries")

                return

            # Convert snapshots to display currency for accurate metrics
            display_currency_snapshots = self._convert_snapshots_to_currency(
                snapshots, display_currency_code, portfolio_manager
            )

            # Convert cash flows to display currency
            display_currency_cash_flows = self._convert_cash_flows_to_currency(
                cash_flows_float, snapshots[0].base_currency.value, display_currency_code, portfolio_manager
            )

            # Calculate comprehensive metrics using the metrics calculator
            comprehensive_metrics = metrics_calculator.calculate_portfolio_metrics(
                display_currency_snapshots, benchmark, display_currency_cash_flows
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

        # Display unified metrics with descriptions
        st.markdown("---")
        st.subheader("📊 Performance Summary")

        # Core Performance Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Return",
                f"{metrics.get('total_return_twr', 0)*100:.2f}%"
            )
            st.caption("📏 Total portfolio return for the selected period")

            # YTD Performance
            try:
                ytd_start = date(date.today().year, 1, 1)
                if end_date < ytd_start:
                    ytd_start = end_date
                ytd_snaps = portfolio_manager.storage.load_snapshots(
                    portfolio_manager.current_portfolio.id, ytd_start, end_date
                )

                if len(ytd_snaps) >= 2:
                    # Convert YTD snapshots to display currency for accurate YTD performance
                    ytd_display_currency_snapshots = self._convert_snapshots_to_currency(
                        ytd_snaps, display_currency_code, portfolio_manager
                    )

                    ytd_flows = portfolio_manager.get_external_cash_flows_by_day(
                        ytd_start, end_date
                    )
                    ytd_flows_f = {d: float(v) for d, v in ytd_flows.items()}

                    # Convert YTD cash flows to display currency
                    ytd_display_currency_flows = self._convert_cash_flows_to_currency(
                        ytd_flows_f, ytd_snaps[0].base_currency.value, display_currency_code, portfolio_manager
                    )

                    daily_returns = metrics_calculator.calculate_time_weighted_return(
                        ytd_display_currency_snapshots, ytd_display_currency_flows
                    )

                    if daily_returns:
                        twr = 1.0
                        for r in daily_returns:
                            twr *= 1.0 + r
                        ytd_perf_pct = (twr - 1.0) * 100.0
                        st.metric("YTD Performance", f"{ytd_perf_pct:.2f}%", delta=f"{ytd_perf_pct:.2f}%")
                    else:
                        st.metric("YTD Performance", "N/A")
                else:
                    st.metric("YTD Performance", "N/A")
                st.caption("📅 Year-to-date performance")
            except Exception:
                st.metric("YTD Performance", "N/A")
                st.caption("📅 Year-to-date performance")

        with col2:
            st.metric(
                "Annualized Return",
                f"{metrics.get('annualized_return_twr', 0)*100:.2f}%",
            )
            st.caption("📈 Annual equivalent return rate")

            st.metric("Volatility", f"{metrics.get('volatility', 0)*100:.2f}%")
            st.caption("📊 Standard deviation of returns (risk measure)")

        with col3:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.2f}%")
            st.caption("📉 Largest peak-to-trough decline")

            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")
            st.caption("⚖️ Risk-adjusted return (higher is better)")

        with col4:
            st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.3f}")
            st.caption("📉 Downside risk-adjusted return")

            st.metric("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.3f}")
            st.caption("📈 Return vs max drawdown ratio")

        # Risk Metrics Section
        st.markdown("---")
        st.subheader("⚠️ Risk Analysis")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Value at Risk (5%)", f"{metrics.get('var_5pct', 0)*100:.3f}%")
            st.caption("📉 Maximum expected loss in 95% of cases")

        with col2:
            st.metric("Conditional VaR (5%)", f"{metrics.get('cvar_5pct', 0)*100:.3f}%")
            st.caption("💥 Average loss when VaR threshold is exceeded")

        with col3:
            st.metric(
                "Modified Dietz Return",
                f"{metrics.get('modified_dietz_return', 0)*100:.2f}%",
            )
            st.caption("🔄 Alternative return calculation method")

        # Benchmark Comparison (if available)
        st.markdown("---")
        if metrics.get("benchmark_available"):
            st.subheader(f"📊 Benchmark Comparison ({benchmark})")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Beta", f"{metrics.get('beta', 0):.3f}")
                st.caption("📈 Portfolio sensitivity to market movements")

            with col2:
                st.metric("Alpha", f"{metrics.get('alpha', 0)*100:.2f}%")
                st.caption("✨ Excess return vs benchmark (skill-based)")

            with col3:
                st.metric("Information Ratio", f"{metrics.get('information_ratio', 0):.3f}")
                st.caption("🎯 Active return per unit of tracking error")

            with col4:
                st.metric("Benchmark Return", f"{metrics.get('benchmark_return', 0)*100:.2f}%")
                st.caption(f"📊 {benchmark} return for comparison")
        else:
            st.info(f"💡 Benchmark data for {benchmark} is not available for comparison metrics")

        # Time-Weighted vs Money-Weighted Return Comparison
        with st.expander("🔍 Advanced: Return Methodology Comparison"):
            st.markdown("**Understanding Different Return Calculation Methods**")

            try:
                                # Calculate MWR using IRR (Internal Rate of Return) for accurate money-weighted return
                # IRR expects deposits as negative values (investor outflows)
                irr_cash_flows = {d: -v for d, v in cash_flows_float.items()} if cash_flows_float else {}
                irr_annual = metrics_calculator.calculate_internal_rate_of_return(
                    snapshots, irr_cash_flows
                )

                # Convert annual IRR to period return for comparison with TWR
                start_date_calc = snapshots[0].date
                end_date_calc = snapshots[-1].date
                days = (end_date_calc - start_date_calc).days
                years = days / 365.25

                if years > 0 and irr_annual != 0:
                    # Convert annual IRR to period return: (1 + annual_irr)^years - 1
                    mwr_period_pct = ((1 + irr_annual) ** years - 1) * 100.0
                else:
                    mwr_period_pct = 0.0

                # Use the same TWR calculation as YTD (total return for period, not annualized)
                twr_period_pct = metrics.get('total_return_twr', 0) * 100

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Time-Weighted Return",
                        f"{twr_period_pct:.2f}%",
                    )
                    st.caption("🎯 Pure investment performance (eliminates cash flow timing)")

                with col2:
                    st.metric("Money-Weighted Return", f"{mwr_period_pct:.2f}%")
                    st.caption("👤 Your actual experience (includes cash flow timing)")

                with col3:
                    difference_pct = twr_period_pct - mwr_period_pct
                    st.metric("Difference", f"{difference_pct:.2f}%")
                    if abs(difference_pct) > 1.0:  # 1% threshold
                        if difference_pct > 0:
                            st.caption("✅ Good timing: Added money during gains")
                        else:
                            st.caption("⚠️ Poor timing: Added money during losses")
                    else:
                        st.caption("➖ Minimal impact from cash flow timing")

            except Exception as e:
                st.warning(f"Could not calculate money-weighted returns: {e}")
                st.info("💡 Money-weighted returns require additional data processing.")

        # Use benchmark prices from metrics calculation (avoids duplicate fetch)
        bench_map: Dict[date, float] = comprehensive_metrics.get("benchmark_prices", {})

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
            display_currency_snapshots,  # Use converted snapshots for charts
            display_currency_code,
            portfolio_manager,
            start_date,
            end_date,
            benchmark_symbol=benchmark,
            benchmark_prices_aligned=bench_prices_aligned,
            portfolio_returns_twr=portfolio_returns_twr,  # Pass TWR returns for plotting
        )



        # Methodology explanation
        with st.expander("💡 Understanding Your Metrics"):
            st.markdown("""
            **Performance Metrics Explained:**

            📈 **Returns**: All returns use Time-Weighted Return (TWR) methodology, which eliminates the impact of cash flows to measure pure investment performance.

            ⚖️ **Risk-Adjusted Ratios**:
            - **Sharpe Ratio**: Measures excess return per unit of total risk (higher is better)
            - **Sortino Ratio**: Like Sharpe but only considers downside risk (higher is better)
            - **Calmar Ratio**: Annual return divided by maximum drawdown (higher is better)

            📉 **Risk Metrics**:
            - **Volatility**: Standard deviation of returns (lower means more stable)
            - **Max Drawdown**: Largest peak-to-trough decline (lower is better)
            - **VaR (5%)**: Expected maximum loss in 95% of cases
            - **Conditional VaR**: Average loss when VaR threshold is exceeded

            📊 **Benchmark Metrics** (when available):
            - **Beta**: Portfolio's sensitivity to market movements (1.0 = same as market)
            - **Alpha**: Excess return vs benchmark after adjusting for risk
            - **Information Ratio**: Active return per unit of tracking error
            """)

        # Charts section header
        st.subheader("📈 Visual Analysis")

        # Daily time-weighted returns chart
        if portfolio_returns_twr:
            st.markdown("**Daily Returns Distribution**")

            # Convert returns to percentages for better readability
            daily_returns_pct = [r * 100 for r in portfolio_returns_twr]

            # Create dates list from snapshots (skip first date since returns start from second snapshot)
            dates = [s.date for s in snapshots]

            fig3 = go.Figure()
            fig3.add_trace(
                go.Scatter(
                    x=dates[
                        1:
                    ],  # Skip first date since returns start from second snapshot
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
                    x=dates[
                        1:
                    ],  # Skip first date since returns start from second snapshot
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
                        x=dates[
                            1:
                        ],  # Skip first date since returns start from second snapshot
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

    def render_scenarios(self, portfolio_manager, metrics_calculator):
        """Render portfolio scenarios and what-if analysis."""
        st.header("🔮 Portfolio Scenarios")

        if not st.session_state.portfolio_loaded:
            st.warning("Please load a portfolio to run scenario analysis.")
            return

        if not portfolio_manager.current_portfolio:
            st.warning("No portfolio currently loaded. Please load or create a portfolio first.")
            return

        # Import here to avoid circular imports
        from src.portfolio.scenarios import PortfolioScenarioEngine, ScenarioType

        st.markdown("""
        Explore how your portfolio might perform under different market conditions.
        Our Monte Carlo simulations project potential outcomes based on various economic scenarios.
        """)

        # Initialize scenario engine
        if "scenario_engine" not in st.session_state:
            st.session_state.scenario_engine = PortfolioScenarioEngine(random_seed=42)

        engine = st.session_state.scenario_engine

        # Get current portfolio snapshot
        try:
            current_portfolio = portfolio_manager.create_snapshot(save=False)
        except Exception as e:
            st.error(f"Could not create current portfolio snapshot: {e}")
            return

        # Simulation Settings (in main tab area)
        st.subheader("🔧 Simulation Settings")

        settings_col1, settings_col2, settings_col3 = st.columns(3)

        with settings_col1:
            projection_years = st.slider(
                "Projection Period (Years)",
                1.0, 10.0, 5.0, 0.5,
                help="How far into the future to project",
                key="scenario_projection_years"
            )
            monte_carlo_runs = st.selectbox(
                "Monte Carlo Runs",
                [500, 1000, 2500, 5000],
                index=1,
                help="More runs = more accurate but slower",
                key="scenario_monte_carlo_runs"
            )

        with settings_col2:
            confidence_levels = st.multiselect(
                "Confidence Intervals (%)",
                [5, 10, 25, 50, 75, 90, 95],
                default=[5, 25, 50, 75, 95],
                help="Percentiles to show in charts",
                key="scenario_confidence_levels"
            )

        with settings_col3:
            st.markdown("**💰 Cash Flow Assumptions**")
            recurring_deposits = st.number_input(
                "Monthly Deposits ($)",
                min_value=0.0,
                value=0.0,
                step=100.0,
                help="Regular monthly contributions",
                key="scenario_monthly_deposits"
            )
            recurring_withdrawals = st.number_input(
                "Monthly Withdrawals ($)",
                min_value=0.0,
                value=0.0,
                step=100.0,
                help="Regular monthly withdrawals",
                key="scenario_monthly_withdrawals"
            )

        # Convert confidence levels to decimals
        confidence_intervals = [c / 100.0 for c in confidence_levels]

        # Scenario selection
        st.subheader("📋 Select Scenarios to Compare")

        col1, col2 = st.columns(2)

        with col1:
            run_optimistic = st.checkbox("🟢 **Optimistic** - Bull Market Growth", value=True)
            run_likely = st.checkbox("🟡 **Likely** - Historical Average", value=True)

        with col2:
            run_pessimistic = st.checkbox("🟠 **Pessimistic** - Economic Downturn", value=True)
            run_stress = st.checkbox("🔴 **Stress** - Market Crash", value=False)

        # Run simulations button
        if st.button("🚀 Run Scenario Analysis", type="primary"):

            # Get predefined scenarios
            scenarios = engine.create_predefined_scenarios(current_portfolio.total_value)

            # Filter selected scenarios
            selected_scenarios = {}
            if run_optimistic:
                selected_scenarios["optimistic"] = scenarios["optimistic"]
            if run_likely:
                selected_scenarios["likely"] = scenarios["likely"]
            if run_pessimistic:
                selected_scenarios["pessimistic"] = scenarios["pessimistic"]
            if run_stress:
                selected_scenarios["stress"] = scenarios["stress"]

            if not selected_scenarios:
                st.error("Please select at least one scenario to run.")
                return

            # Update scenario configurations with user settings
            for scenario_config in selected_scenarios.values():
                scenario_config.projection_years = projection_years
                scenario_config.monte_carlo_runs = monte_carlo_runs
                scenario_config.confidence_intervals = confidence_intervals
                scenario_config.recurring_deposits = recurring_deposits
                scenario_config.recurring_withdrawals = recurring_withdrawals

            # Run simulations with progress bar
            simulation_results = {}
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, (name, config) in enumerate(selected_scenarios.items()):
                status_text.text(f"Running {config.name} scenario...")
                progress_bar.progress((i + 1) / len(selected_scenarios))

                try:
                    result = engine.run_scenario_simulation(current_portfolio, config)
                    simulation_results[name] = result
                except Exception as e:
                    st.error(f"Error running {name} scenario: {e}")
                    continue

            progress_bar.empty()
            status_text.empty()

            if not simulation_results:
                st.error("No scenarios completed successfully.")
                return

            # Store results in session state
            st.session_state.simulation_results = simulation_results
            st.success(f"✅ Completed {len(simulation_results)} scenario(s)")

        # Display results if available
        if "simulation_results" in st.session_state and st.session_state.simulation_results:
            self._render_scenario_results(st.session_state.simulation_results, current_portfolio)

        # Add advanced what-if capabilities section
        st.markdown("---")
        self._render_advanced_what_if_section(portfolio_manager, current_portfolio)

    def _render_scenario_results(self, simulation_results, current_portfolio):
        """Render the results of scenario simulations."""
        st.header("📊 Scenario Analysis Results")

        # Summary comparison table
        st.subheader("📋 Scenario Comparison")

        from src.portfolio.scenarios import PortfolioScenarioEngine
        engine = PortfolioScenarioEngine()
        comparison = engine.compare_scenarios(simulation_results)

        # Create comparison DataFrame
        comparison_data = []
        for scenario_name, stats in comparison.items():
            scenario_result = simulation_results[scenario_name]
            comparison_data.append({
                "Scenario": scenario_result.scenario_config.name,
                "Type": scenario_result.scenario_config.scenario_type.value.title(),
                "Final Value (Mean)": f"${stats['mean_final_value']:,.0f}",
                "Final Value (Median)": f"${stats['median_final_value']:,.0f}",
                "Annualized Return": f"{stats['mean_annualized_return']*100:.1f}%",
                "Probability of Loss": f"{stats['probability_of_loss']*100:.1f}%",
                "Max Drawdown": f"{stats['mean_max_drawdown']*100:.1f}%",
                "Sharpe Ratio": f"{stats['mean_sharpe_ratio']:.2f}"
            })

        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

        # Portfolio value projections chart
        st.subheader("📈 Portfolio Value Projections")

        # Create chart data
        fig = go.Figure()

        # Color mapping for scenarios
        scenario_colors = {
            "optimistic": "#00ff00",
            "likely": "#ffff00",
            "pessimistic": "#ff8000",
            "stress": "#ff0000"
        }

        for scenario_name, result in simulation_results.items():
            color = scenario_colors.get(scenario_name, "#666666")

            # Add mean trajectory
            fig.add_trace(go.Scatter(
                x=result.dates,
                y=result.mean_trajectory,
                mode='lines',
                name=f"{result.scenario_config.name} (Mean)",
                line=dict(color=color, width=3)
            ))

            # Add confidence bands
            if 0.25 in result.percentiles and 0.75 in result.percentiles:
                fig.add_trace(go.Scatter(
                    x=result.dates + result.dates[::-1],
                    y=result.percentiles[0.75] + result.percentiles[0.25][::-1],
                    fill='tonexty',
                    fillcolor=f'rgba{tuple(list(int(color[i:i+2], 16) for i in (1, 3, 5)) + [0.2])}',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f"{result.scenario_config.name} (25%-75%)",
                    showlegend=False
                ))

            # Add extreme confidence bands
            if 0.05 in result.percentiles and 0.95 in result.percentiles:
                fig.add_trace(go.Scatter(
                    x=result.dates + result.dates[::-1],
                    y=result.percentiles[0.95] + result.percentiles[0.05][::-1],
                    fill='tonexty',
                    fillcolor=f'rgba{tuple(list(int(color[i:i+2], 16) for i in (1, 3, 5)) + [0.1])}',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f"{result.scenario_config.name} (5%-95%)",
                    showlegend=False
                ))

        # Add starting value line
        fig.add_hline(
            y=float(current_portfolio.total_value),
            line_dash="dash",
            line_color="gray",
            annotation_text="Current Value"
        )

        fig.update_layout(
            title="Portfolio Value Projections",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Final value distributions
        st.subheader("📊 Final Value Distributions")

        fig_hist = go.Figure()

        for scenario_name, result in simulation_results.items():
            color = scenario_colors.get(scenario_name, "#666666")

            fig_hist.add_trace(go.Histogram(
                x=result.final_values,
                name=result.scenario_config.name,
                opacity=0.7,
                nbinsx=50,
                marker_color=color
            ))

        fig_hist.update_layout(
            title="Distribution of Final Portfolio Values",
            xaxis_title="Final Portfolio Value ($)",
            yaxis_title="Frequency",
            barmode='overlay',
            height=400
        )

        st.plotly_chart(fig_hist, use_container_width=True)

        # Risk metrics
        st.subheader("⚠️ Risk Analysis")

        # Create risk metrics for each scenario
        for scenario_name, result in simulation_results.items():
            with st.expander(f"📋 {result.scenario_config.name} - Detailed Metrics"):

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Mean Final Value", f"${result.get_summary_stats()['mean_final_value']:,.0f}")
                    st.metric("Probability of Loss", f"{result.probability_of_loss*100:.1f}%")
                    st.metric("Probability of Doubling", f"{result.probability_of_doubling*100:.1f}%")

                with col2:
                    st.metric("Mean Annual Return", f"{result.get_summary_stats()['mean_annualized_return']*100:.1f}%")
                    st.metric("Mean Sharpe Ratio", f"{result.get_summary_stats()['mean_sharpe_ratio']:.2f}")
                    st.metric("Mean Max Drawdown", f"{result.get_summary_stats()['mean_max_drawdown']*100:.1f}%")

                with col3:
                    st.metric("Best Case (95%)", f"${result.percentiles[0.95][-1]:,.0f}" if 0.95 in result.percentiles else "N/A")
                    st.metric("Worst Case (5%)", f"${result.percentiles[0.05][-1]:,.0f}" if 0.05 in result.percentiles else "N/A")
                    st.metric("Standard Deviation", f"${result.get_summary_stats()['std_final_value']:,.0f}")

                # Show scenario assumptions
                st.markdown("**Market Assumptions:**")
                assumptions = result.scenario_config.market_assumptions
                st.write(f"• Expected Return: {assumptions.expected_return*100:.1f}%")
                st.write(f"• Volatility: {assumptions.volatility*100:.1f}%")
                st.write(f"• Inflation Rate: {assumptions.inflation_rate*100:.1f}%")

        # Key insights
        st.subheader("💡 Key Insights")

        # Find best and worst performing scenarios
        mean_final_values = {name: result.get_summary_stats()['mean_final_value'] for name, result in simulation_results.items()}
        best_scenario = max(mean_final_values.keys(), key=lambda k: mean_final_values[k])
        worst_scenario = min(mean_final_values.keys(), key=lambda k: mean_final_values[k])

        st.markdown(f"""
        **Scenario Comparison:**
        - 🏆 **Best Performing**: {simulation_results[best_scenario].scenario_config.name} with mean final value of ${mean_final_values[best_scenario]:,.0f}
        - 📉 **Worst Performing**: {simulation_results[worst_scenario].scenario_config.name} with mean final value of ${mean_final_values[worst_scenario]:,.0f}
        - 📊 **Spread**: ${mean_final_values[best_scenario] - mean_final_values[worst_scenario]:,.0f} difference between best and worst scenarios

        **Risk Considerations:**
        - Higher expected returns typically come with higher volatility
        - Diversification across asset classes can help reduce overall portfolio risk
        - Regular contributions can help smooth out market volatility through dollar-cost averaging
        """)

    def _render_advanced_what_if_section(self, portfolio_manager, current_portfolio):
        """Render the advanced what-if analysis section."""
        st.header("🔧 Advanced What-If Analysis")

        st.markdown("""
        Go beyond predefined scenarios with **precise portfolio modifications** and **custom market assumptions**.
        Perfect for testing specific investment ideas or allocation changes.
        """)

        # Import here to avoid circular imports
        from src.agents.tools import AdvancedWhatIfTool, HypotheticalPositionTool

        # Create tabs for different types of analysis
        tab1, tab2 = st.tabs(["🔧 Portfolio Modifications", "🧪 Hypothetical Positions"])

        with tab1:
            self._render_portfolio_modifications_tab(portfolio_manager, current_portfolio)

        with tab2:
            self._render_hypothetical_positions_tab(portfolio_manager, current_portfolio)

    def _render_portfolio_modifications_tab(self, portfolio_manager, current_portfolio):
        """Render the portfolio modifications tab."""
        st.subheader("🔧 Portfolio Modifications")
        st.markdown("Modify your existing positions and add new ones to see how your portfolio would perform.")

        # Portfolio modification inputs
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📊 Modify Existing Positions**")
            modify_help = """
            Format: SYMBOL:CHANGE

            Examples:
            • AAPL:+50% (increase by 50%)
            • MSFT:-25% (decrease by 25%)
            • GOOGL:=150 (set to exactly 150 shares)
            • TSLA:+100 (add 100 shares)
            • AMZN:-50 (remove 50 shares)

            Multiple: AAPL:+50%,MSFT:-25%,GOOGL:=150
            """
            modify_positions = st.text_area(
                "Position Modifications",
                placeholder="e.g., AAPL:+50%,MSFT:-25%",
                help=modify_help
            )

        with col2:
            st.markdown("**➕ Add New Positions**")
            add_help = """
            Format: SYMBOL:QUANTITY@PRICE

            Examples:
            • NVDA:100@$800 (100 shares at $800)
            • TSLA:50@$250 (50 shares at $250)

            Multiple: NVDA:100@$800,TSLA:50@$250
            """
            add_positions = st.text_area(
                "New Positions",
                placeholder="e.g., NVDA:100@$800,TSLA:50@$250",
                help=add_help
            )

        # Market assumptions
        st.markdown("**📈 Market Assumptions**")
        col1, col2, col3 = st.columns(3)

        with col1:
            market_return = st.slider(
                "Expected Annual Return (%)",
                min_value=-20.0, max_value=30.0, value=8.0, step=0.5,
                help="Expected market return per year",
                key="whatif_market_return"
            ) / 100.0

            projection_years = st.slider(
                "Projection Period (Years)",
                min_value=0.5, max_value=10.0, value=2.0, step=0.5,
                key="whatif_projection_years"
            )

        with col2:
            market_volatility = st.slider(
                "Market Volatility (%)",
                min_value=5.0, max_value=80.0, value=20.0, step=1.0,
                help="Annual volatility/risk level",
                key="whatif_market_volatility"
            ) / 100.0

            monte_carlo_runs = st.selectbox(
                "Simulation Runs",
                [500, 1000, 2500, 5000],
                index=1,
                help="More runs = more accurate but slower",
                key="whatif_monte_carlo_runs"
            )

        with col3:
            recurring_deposits = st.number_input(
                "Monthly Deposits ($)",
                min_value=0.0, value=0.0, step=100.0,
                help="Regular monthly contributions",
                key="whatif_monthly_deposits"
            )

            stress_test = st.checkbox(
                "Apply Stress Test Conditions",
                help="Force negative returns and high volatility",
                key="whatif_stress_test"
            )

        # Run analysis button
        if st.button("🚀 Run Advanced What-If Analysis", type="primary", key="advanced_whatif"):
            if not modify_positions and not add_positions:
                st.error("Please specify position modifications or new positions to add.")
                return

            try:
                # Import and run the advanced tool
                from src.agents.tools import AdvancedWhatIfTool

                tool = AdvancedWhatIfTool(portfolio_manager)

                with st.spinner("Running advanced Monte Carlo simulation..."):
                    result = tool._run(
                        scenario_type="custom",
                        projection_years=projection_years,
                        monte_carlo_runs=monte_carlo_runs,
                        modify_positions=modify_positions,
                        add_positions=add_positions,
                        market_return=market_return,
                        market_volatility=market_volatility,
                        recurring_deposits=recurring_deposits,
                        stress_test=stress_test
                    )

                # Display results
                if "Error" in result:
                    st.error(result)
                else:
                    st.success("✅ Analysis complete!")

                    # Format and display the result
                    st.markdown("### 📊 Analysis Results")

                    # Split result into sections for better formatting
                    sections = result.split("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                    if len(sections) > 1:
                        # Display the formatted result
                        for section in sections[1:]:  # Skip the title
                            lines = section.strip().split('\n')
                            for line in lines:
                                if line.startswith('📊') or line.startswith('🎯') or line.startswith('📈') or line.startswith('⚡') or line.startswith('⚠️') or line.startswith('💡'):
                                    st.markdown(f"**{line}**")
                                elif line.startswith('   •'):
                                    st.markdown(line)
                                elif line.strip():
                                    st.markdown(line)
                    else:
                        st.code(result)

            except Exception as e:
                st.error(f"Error running analysis: {e}")

    def _render_hypothetical_positions_tab(self, portfolio_manager, current_portfolio):
        """Render the hypothetical positions tab."""
        st.subheader("🧪 Hypothetical Position Testing")
        st.markdown("Test adding a single new investment to see how it would impact your portfolio.")

        # Input controls
        col1, col2 = st.columns(2)

        with col1:
            symbol = st.text_input(
                "Stock Symbol",
                placeholder="e.g., NVDA, TSLA, AMZN",
                help="Enter the stock symbol you want to test",
                key="hypo_symbol"
            ).upper()

            investment_type = st.radio(
                "Investment Method",
                ["Dollar Amount", "Share Quantity"],
                help="Choose how to specify the investment size",
                key="hypo_investment_type"
            )

            if investment_type == "Dollar Amount":
                investment_amount = st.number_input(
                    "Investment Amount ($)",
                    min_value=100.0, value=5000.0, step=100.0,
                    key="hypo_investment_amount"
                )
                purchase_price = st.number_input(
                    "Expected Purchase Price ($)",
                    min_value=0.01, value=100.0, step=0.01,
                    help="Price per share you expect to pay",
                    key="hypo_purchase_price_dollar"
                )
                quantity = investment_amount / purchase_price if purchase_price > 0 else 0
                st.info(f"This equals approximately {quantity:.1f} shares")
            else:
                quantity = st.number_input(
                    "Number of Shares",
                    min_value=1.0, value=100.0, step=1.0,
                    key="hypo_num_shares"
                )
                purchase_price = st.number_input(
                    "Purchase Price per Share ($)",
                    min_value=0.01, value=100.0, step=0.01,
                    key="hypo_purchase_price_shares"
                )
                investment_amount = quantity * purchase_price
                st.info(f"Total investment: ${investment_amount:,.2f}")

        with col2:
            scenario = st.selectbox(
                "Market Scenario",
                ["optimistic", "likely", "pessimistic", "stress"],
                index=1,
                help="Choose the market conditions to test under",
                key="hypo_scenario"
            )

            time_horizon = st.slider(
                "Time Horizon (Years)",
                min_value=0.5, max_value=5.0, value=1.0, step=0.5,
                help="How long to project the investment",
                key="hypo_time_horizon"
            )

            # Show scenario details
            scenario_details = {
                "optimistic": "🟢 Bull market: 12% return, 16% volatility",
                "likely": "🟡 Historical average: 8% return, 20% volatility",
                "pessimistic": "🟠 Economic downturn: 3% return, 28% volatility",
                "stress": "🔴 Market crash: -5% return, 40% volatility"
            }
            st.info(scenario_details[scenario])

            # Portfolio impact preview
            current_value = float(current_portfolio.total_value)
            allocation_pct = (investment_amount / current_value) * 100
            st.metric(
                "Portfolio Allocation",
                f"{allocation_pct:.1f}%",
                help=f"This investment would be {allocation_pct:.1f}% of your current portfolio"
            )

        # Run analysis button
        if st.button("🧪 Test Hypothetical Position", type="primary", key="hypothetical_test"):
            if not symbol:
                st.error("Please enter a stock symbol.")
                return

            if quantity <= 0 or purchase_price <= 0:
                st.error("Please enter valid quantity and price values.")
                return

            try:
                # Import and run the hypothetical tool
                from src.agents.tools import HypotheticalPositionTool

                tool = HypotheticalPositionTool(portfolio_manager)

                with st.spinner(f"Testing {symbol} investment under {scenario} scenario..."):
                    result = tool._run(
                        symbol=symbol,
                        quantity=quantity,
                        purchase_price=purchase_price,
                        scenario=scenario,
                        time_horizon=time_horizon
                    )

                # Display results
                if "Error" in result:
                    st.error(result)
                else:
                    st.success("✅ Hypothetical analysis complete!")

                    # Format and display the result
                    st.markdown("### 🧪 Hypothetical Position Results")

                    # Split result into sections for better formatting
                    sections = result.split("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                    if len(sections) > 1:
                        # Display the formatted result
                        for section in sections[1:]:  # Skip the title
                            lines = section.strip().split('\n')
                            for line in lines:
                                if line.startswith('💼') or line.startswith('📊') or line.startswith('🎯') or line.startswith('📈') or line.startswith('⚡') or line.startswith('⚠️') or line.startswith('💡'):
                                    st.markdown(f"**{line}**")
                                elif line.startswith('   •'):
                                    st.markdown(line)
                                elif line.strip():
                                    st.markdown(line)
                    else:
                        st.code(result)

                    # Add comparison with portfolio
                    original_value = float(current_portfolio.total_value)
                    st.markdown("### 📊 Portfolio Impact Summary")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Portfolio", f"${original_value:,.0f}")
                    with col2:
                        st.metric("Investment Amount", f"${investment_amount:,.0f}")
                    with col3:
                        st.metric("New Allocation", f"{allocation_pct:.1f}%")

            except Exception as e:
                st.error(f"Error running hypothetical analysis: {e}")

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

        st.subheader("📊 Data Management")
        st.info(
            "Market data updates are now handled in the main portfolio view. Use the '📈 Update Market Data' button there to refresh snapshots with current prices."
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

            # Show latest snapshot info
            latest_snapshot = portfolio_manager.storage.get_latest_snapshot(portfolio_id)
            if latest_snapshot:
                st.write(f"**Latest Snapshot:** {latest_snapshot.date.isoformat()}")
                st.write(f"**Snapshot Value:** ${latest_snapshot.total_value:,.2f}")
            else:
                st.warning("No snapshots found for this portfolio")

            # Keep a simple button for emergency data refresh but discourage its use
            if st.button("🔄 Emergency Data Refresh", help="Only use if the main update button isn't working"):
                with st.spinner("Updating snapshots..."):
                    try:
                        end_date = date.today()
                        start_date = end_date - timedelta(days=30)
                        refreshed = portfolio_manager.create_snapshots_for_range(start_date, end_date, save=True)
                        st.success(f"✅ Updated {len(refreshed)} snapshots with fresh market data (saved to storage)")

                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        logging.error(f"Emergency refresh failed: {error_details}")
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

    def _render_category_table(self, category_items: List[Dict], base_currency_code: str, portfolio, portfolio_manager) -> None:
        """Render a compact table for a specific category of positions."""

        if not category_items:
            return

        # Create a pandas DataFrame for easy table display
        table_data = []
        for pos in category_items:
            # Format values for display
            def fmt_money(val):
                if val is None:
                    return "-"
                return f"{float(val):,.2f}"

            def fmt_signed(val):
                if val is None:
                    return "-"
                return f"{float(val):+.2f}"

            def fmt_percent(val):
                if val is None:
                    return "-"
                return f"{float(val):+.2f}%"

            def fmt_signed_colored(val):
                """Format signed value (color applied via pandas styling)."""
                if val is None:
                    return "-"
                num_val = float(val)
                if num_val > 0:
                    return f"+{num_val:,.2f}"
                elif num_val < 0:
                    return f"{num_val:,.2f}"
                else:
                    return f"{num_val:,.2f}"

            def fmt_percent_colored(val):
                """Format percentage (color applied via pandas styling)."""
                if val is None:
                    return "-"
                num_val = float(val)
                if num_val > 0:
                    return f"+{num_val:.2f}%"
                elif num_val < 0:
                    return f"{num_val:.2f}%"
                else:
                    return f"{num_val:.2f}%"

            def fmt_pnl_colored(pnl_val, pct_val, currency_code):
                """Format P&L with both absolute and percentage values (color applied via pandas styling)."""
                if pnl_val is None or pnl_val == 0:
                    return "-"
                pnl_num = float(pnl_val)
                pct_num = float(pct_val) if pct_val is not None else 0

                if pnl_num > 0:
                    return f"+{pnl_num:,.2f} {currency_code} (+{pct_num:.2f}%)"
                elif pnl_num < 0:
                    return f"{pnl_num:,.2f} {currency_code} ({pct_num:.2f}%)"
                else:
                    return f"{pnl_num:,.2f} {currency_code} ({pct_num:.2f}%)"

            # Determine quantity display
            qty = pos.get("quantity")
            price = pos.get("current_price")
            if qty is not None and price is not None:
                qty_display = f"{fmt_money(qty)} @ {fmt_money(price)}"
            elif pos.get("instrument_type") == "cash":
                qty_display = "Cash"
            else:
                qty_display = fmt_money(qty) if qty is not None else "-"

            # Market value display
            mv_base = pos.get("market_value_base")
            mv_native = pos.get("market_value")
            if pos.get("instrument_type") == "cash":
                mv_display = f"{fmt_money(mv_native)} {pos.get('currency')}"
            else:
                mv_display = f"{fmt_money(mv_base)} {base_currency_code}"

            # P&L display with color coding
            pnl_base = pos.get("unrealized_pnl_base")
            pnl_pct = pos.get("unrealized_pnl_percent")
            if pnl_base is not None and pnl_base != 0:
                pnl_display = fmt_pnl_colored(pnl_base, pnl_pct, base_currency_code)
            else:
                pnl_display = "-"

            # YTD performance with color coding
            ytd_val = pos.get("ytd_market_pnl")
            ytd_pct = pos.get("ytd_market_pnl_percent")
            if ytd_val is not None and ytd_val != 0:
                # Format YTD with color coding
                if float(ytd_val) > 0:
                    ytd_display = f"{float(ytd_val):,.2f} (+{float(ytd_pct):.2f}%)" if ytd_pct else f"{float(ytd_val):,.2f}"
                elif float(ytd_val) < 0:
                    ytd_display = f"{float(ytd_val):,.2f} ({float(ytd_pct):.2f}%)" if ytd_pct else f"{float(ytd_val):,.2f}"
                else:
                    ytd_display = f"{float(ytd_val):,.2f} ({float(ytd_pct):.2f}%)" if ytd_pct else f"{float(ytd_val):,.2f}"
            else:
                ytd_display = "-"

            table_data.append({
                "Symbol": pos.get("symbol", "-"),
                "Name": pos.get("name", "-"),
                "ISIN": pos.get("isin", "-"),
                "Qty/Price": qty_display,
                "Market Value": mv_display,
                "Unrealized P&L": pnl_display,
                "YTD": ytd_display,
                "Currency": pos.get("currency", "-"),
            })

        # Create DataFrame and display
        if table_data:
            df = pd.DataFrame(table_data)

            # Add summary row for this category
            total_mv = sum(
                float(pos.get("market_value_base", 0) or 0)
                for pos in category_items
            )
            total_pnl = sum(
                float(pos.get("unrealized_pnl_base", 0) or 0)
                for pos in category_items
            )

            # Format summary row with color coding for P&L
            if total_pnl > 0:
                pnl_summary = f"{total_pnl:,.2f} {base_currency_code}"
            elif total_pnl < 0:
                pnl_summary = f"{total_pnl:,.2f} {base_currency_code}"
            else:
                pnl_summary = f"{total_pnl:,.2f} {base_currency_code}"

            summary_row = {
                "Symbol": "TOTAL",
                "Name": "",
                "ISIN": "",
                "Qty/Price": "",
                "Market Value": f"{total_mv:,.2f} {base_currency_code}",
                "Unrealized P&L": pnl_summary,
                "YTD": "",
                "Currency": "",
            }

            # Add summary row to DataFrame
            df_summary = pd.DataFrame([summary_row])
            df_final = pd.concat([df, df_summary], ignore_index=True)

            # Create styled dataframe using pandas styling for colors
            def apply_pnl_color(val):
                """Apply color styling to P&L values."""
                if pd.isna(val) or val == "-":
                    return ""

                # Extract numeric value from formatted string
                val_str = str(val)
                if val_str.startswith("+") or ("+" in val_str and not val_str.startswith("-")):
                    # Contains positive P&L
                    return "color: green; font-weight: bold"
                elif val_str.startswith("-"):
                    # Contains negative P&L
                    return "color: red; font-weight: bold"
                return ""

            def apply_ytd_color(val):
                """Apply color styling to YTD values."""
                if pd.isna(val) or val == "-":
                    return ""

                val_str = str(val)
                if "+" in val_str:
                    return "color: green; font-weight: bold"
                elif val_str.startswith("-"):
                    return "color: red; font-weight: bold"
                return ""

            # Apply styling to the dataframe
            styled_df = df_final.style.map(
                apply_pnl_color, subset=["Unrealized P&L"]
            ).map(
                apply_ytd_color, subset=["YTD"]
            ).set_table_styles([
                {'selector': 'th', 'props': [('font-weight', 'bold'), ('text-align', 'center')]},
                {'selector': 'td', 'props': [('text-align', 'left'), ('padding', '8px')]},
                {'selector': 'tr:last-child', 'props': [('font-weight', 'bold'), ('border-top', '2px solid #ccc')]}  # Style summary row
            ])

            # Display the styled table
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", width="medium"),
                    "Name": st.column_config.TextColumn("Name", width="large"),
                    "ISIN": st.column_config.TextColumn("ISIN", width="medium"),
                    "Qty/Price": st.column_config.TextColumn("Qty/Price", width="medium"),
                    "Market Value": st.column_config.TextColumn("Market Value", width="medium"),
                    "Unrealized P&L": st.column_config.TextColumn("Unrealized P&L", width="medium"),
                    "YTD": st.column_config.TextColumn("YTD", width="small"),
                    "Currency": st.column_config.TextColumn("Currency", width="small"),
                }
            )

            # Add category summary metrics below the table
            col1, col2, col3 = st.columns(3)

            with col1:
                positions_count = len([p for p in category_items if p.get("instrument_type") != "cash"])
                st.metric("Positions", positions_count)

            with col2:
                cash_count = len([p for p in category_items if p.get("instrument_type") == "cash"])
                st.metric("Cash Items", cash_count)

            with col3:
                if total_mv > 0:
                    pnl_percent = (total_pnl / total_mv) * 100
                    # Add color coding to the P&L percentage metric
                    if pnl_percent > 0:
                        st.metric("P&L %", f"+{pnl_percent:.2f}%", delta=f"+{pnl_percent:.2f}%")
                    elif pnl_percent < 0:
                        st.metric("P&L %", f"{pnl_percent:.2f}%", delta=f"{pnl_percent:.2f}%")
                    else:
                        st.metric("P&L %", f"{pnl_percent:.2f}%")
                else:
                    st.metric("P&L %", "N/A")
        else:
            st.info("No items in this category.")




def main():
    """Main function to run the Streamlit app."""
    app = PortfolioTrackerUI()
    app.run()


if __name__ == "__main__":
    main()
