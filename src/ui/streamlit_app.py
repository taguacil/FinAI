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
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(_PROJECT_ROOT)

# Use absolute path for data directory to match MCP server
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

from src.agents.llm_config import MODEL_REGISTRY, get_available_models
from src.agents.portfolio_agent import PortfolioAgent
from src.data_providers.manager import DataProviderManager
from src.portfolio.manager import PortfolioManager
from src.portfolio.models import Currency, TransactionType
from src.portfolio.optimizer import OptimizationMethod, OptimizationObjective, PortfolioOptimizer
from src.portfolio.storage import FileBasedStorage
from src.services.market_data_service import MarketDataService
from src.utils.logging_config import setup_logging
from src.utils.metrics import FinancialMetricsCalculator
from src.ui.market_data_page import render_market_data_page


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

            # Use absolute data directory to ensure consistency with MCP server
            storage = FileBasedStorage(_DATA_DIR)
            data_provider = DataProviderManager()

            # Create MarketDataService wrapping the DataProviderManager
            market_data_service = MarketDataService(data_provider)

            # Pass MarketDataService to PortfolioManager with same data_dir
            portfolio_manager = PortfolioManager(storage, market_data_service, data_dir=_DATA_DIR)

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
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            ["💬 AI Chat", "📊 Portfolio", "📈 Analytics", "⚖️ Optimize", "🔮 Scenarios", "🗄️ Market Data", "⚙️ Settings"]
        )

        with tab1:
            self.render_chat_interface(agent)

        with tab2:
            self.render_portfolio_overview(portfolio_manager, market_data_service)

        with tab3:
            self.render_analytics(portfolio_manager, metrics_calculator)

        with tab4:
            self.render_optimization(portfolio_manager, market_data_service)

        with tab5:
            self.render_scenarios(portfolio_manager, metrics_calculator)

        with tab6:
            render_market_data_page(_DATA_DIR)

        with tab7:
            self.render_settings()

    def init_session_state(self):
        """Initialize session state variables."""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "portfolio_loaded" not in st.session_state:
            st.session_state.portfolio_loaded = False
        if "selected_portfolio" not in st.session_state:
            st.session_state.selected_portfolio = None
        # Price fallback prompt state
        if "pending_price_fallback" not in st.session_state:
            st.session_state.pending_price_fallback = None

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

            # Delete portfolio button with confirmation
            with st.sidebar.expander("⚠️ Delete Portfolio", expanded=False):
                st.warning(
                    f"This will permanently delete **{portfolio.name}** and all associated data "
                    "(transactions, market data, backups)."
                )
                market_data_symbols = len(portfolio_manager.market_data_store.get_symbols())
                st.caption(
                    f"• {len(portfolio.transactions)} transactions\n"
                    f"• {len(portfolio.positions)} positions\n"
                    f"• {market_data_symbols} symbols with market data"
                )
                confirm_name = st.text_input(
                    "Type portfolio name to confirm:",
                    key="delete_confirm_input",
                    placeholder=portfolio.name,
                )
                if st.button("Delete Portfolio", type="primary", use_container_width=True):
                    if confirm_name == portfolio.name:
                        try:
                            portfolio_manager.delete_portfolio(portfolio.id, delete_all_data=True)
                            st.session_state.portfolio_loaded = False
                            st.session_state.selected_portfolio = None
                            st.success(f"Deleted portfolio: {portfolio.name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error deleting portfolio: {e}")
                    else:
                        st.error("Portfolio name doesn't match. Deletion cancelled.")

            # Data freshness indicator (compact)
            if market_data_service is not None:
                freshness = market_data_service.freshness
                freshness_text = freshness.freshness_display

                if freshness.is_stale:
                    st.sidebar.caption(f"📡 Data: ⚠️ {freshness_text}")
                else:
                    st.sidebar.caption(f"📡 Data: ✅ {freshness_text}")

            # FX rate status indicator
            if "fx_fallback_warnings" in st.session_state and st.session_state.fx_fallback_warnings:
                num_warnings = len(st.session_state.fx_fallback_warnings)
                st.sidebar.caption(f"💱 FX: ⚠️ {num_warnings} rate(s) unavailable")
            else:
                st.sidebar.caption("💱 FX: ✅ All rates available")

            # Cache clear button - forces reload of all data from database
            st.sidebar.divider()
            if st.sidebar.button("🔄 Reload Data", help="Clear cache and reload all data from database"):
                # Clear the cached components to force fresh load
                st.cache_resource.clear()
                st.session_state.portfolio_loaded = False
                st.session_state.selected_portfolio = portfolio.id  # Remember selection
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
                # Get models from central registry
                available_models = get_available_models()
                model_choice = st.selectbox(
                    "Select Model",
                    available_models,
                    format_func=lambda x: x[1],  # Display name
                    key="model_selector"
                )
                if st.button("Apply Model", use_container_width=True):
                    try:
                        model_key, display_name, config = model_choice
                        # Use the new simplified interface
                        agent.set_llm_config(model_key=model_key)
                        st.success(f"Model set to {display_name}")
                    except Exception as e:
                        st.error(f"Failed to set model: {e}")

    def _render_fx_warnings(self):
        """Render FX rate fallback warnings if any exist."""
        if "fx_fallback_warnings" in st.session_state and st.session_state.fx_fallback_warnings:
            warnings = st.session_state.fx_fallback_warnings
            with st.expander(f"⚠️ FX Rate Issues ({len(warnings)} currency pair(s))", expanded=False):
                st.warning(
                    "Some FX rates were unavailable and defaulted to 1.0 (no conversion). "
                    "This may cause inaccurate values for multi-currency portfolios."
                )
                for warning in warnings[:10]:  # Show first 10
                    st.caption(f"• {warning}")
                if len(warnings) > 10:
                    st.caption(f"... and {len(warnings) - 10} more")
                if st.button("Clear FX Warnings", key="clear_fx_warnings"):
                    st.session_state.fx_fallback_warnings = []
                    st.rerun()

    def _render_price_fallback_prompt(self, portfolio_manager):
        """Render price fallback prompt when an instrument has no market price."""
        if not st.session_state.get("pending_price_fallback"):
            return

        data = st.session_state.pending_price_fallback

        with st.container():
            st.warning(
                f"No market price found for **{data['symbol']}** ({data.get('instrument_name', 'Unknown')}).\n\n"
                "This instrument may not have automatic price feeds (e.g., bonds, private securities)."
            )

            choice = st.radio(
                "What would you like to do?",
                [
                    f"Use purchase price (${data['purchase_price']:.2f}) as market price",
                    "Enter a custom market price",
                    "Skip (leave market price as unavailable)"
                ],
                key="price_fallback_choice"
            )

            custom_price = None
            if choice.startswith("Enter"):
                custom_price = st.number_input(
                    "Custom market price",
                    min_value=0.01,
                    value=float(data['purchase_price']),
                    step=0.01,
                    key="custom_market_price"
                )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Confirm", key="confirm_price_fallback", type="primary"):
                    try:
                        if choice.startswith("Use purchase"):
                            # Use purchase price as market price
                            success = portfolio_manager.set_position_price(
                                symbol=data["symbol"],
                                price=Decimal(str(data["purchase_price"])),
                                target_date=date.fromisoformat(data["date"]),
                            )
                            if success:
                                st.success(f"Set market price for {data['symbol']} to ${data['purchase_price']:.2f}")
                        elif choice.startswith("Enter") and custom_price:
                            # Use custom price
                            success = portfolio_manager.set_position_price(
                                symbol=data["symbol"],
                                price=Decimal(str(custom_price)),
                                target_date=date.fromisoformat(data["date"]),
                            )
                            if success:
                                st.success(f"Set market price for {data['symbol']} to ${custom_price:.2f}")
                        # Clear the pending fallback
                        st.session_state.pending_price_fallback = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error setting price: {e}")

            with col2:
                if st.button("Cancel", key="cancel_price_fallback"):
                    st.session_state.pending_price_fallback = None
                    st.rerun()

    def _render_positions_without_prices(self, portfolio_manager, positions):
        """Render UI for bulk setting prices on positions without prices."""
        positions_without_prices = [
            pos for pos in positions
            if not pos.get("has_current_price") and pos.get("symbol") != "CASH"
        ]

        if not positions_without_prices:
            return

        with st.expander(
            f"💰 Set prices for {len(positions_without_prices)} instruments without prices",
            expanded=False
        ):
            st.info(
                "These instruments don't have automatic price feeds. "
                "You can set their prices manually using purchase price or a custom value."
            )

            # Create a form for bulk price setting
            for pos in positions_without_prices:
                symbol = pos["symbol"]
                avg_cost = pos.get("average_cost", Decimal("0"))

                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    st.text(f"{symbol}")
                    st.caption(f"{pos.get('name', 'Unknown')}")

                with col2:
                    use_purchase = st.checkbox(
                        f"Use ${float(avg_cost):.2f} (purchase price)",
                        key=f"use_purchase_{symbol}",
                        value=True
                    )
                    if not use_purchase:
                        st.number_input(
                            "Custom price",
                            key=f"custom_price_{symbol}",
                            min_value=0.01,
                            value=float(avg_cost),
                            step=0.01
                        )

                with col3:
                    if st.button("Set", key=f"set_price_{symbol}"):
                        try:
                            if st.session_state.get(f"use_purchase_{symbol}", True):
                                price_to_set = avg_cost
                            else:
                                price_to_set = Decimal(str(st.session_state.get(f"custom_price_{symbol}", avg_cost)))

                            success = portfolio_manager.set_position_price(
                                symbol=symbol,
                                price=price_to_set,
                                target_date=date.today(),
                            )
                            if success:
                                st.success(f"Set price for {symbol}")
                                st.rerun()
                            else:
                                st.error(f"Failed to set price for {symbol}")
                        except Exception as e:
                            st.error(f"Error: {e}")

                st.divider()

    def render_portfolio_overview(self, portfolio_manager, market_data_service):
        """Render portfolio overview with transactions."""
        st.header("📊 Portfolio Overview")

        # Clear market data cache to pick up any external updates (e.g., from MCP server)
        portfolio_manager.market_data_store.clear_cache()
        # Also invalidate portfolio history cache to recalculate values with fresh prices
        portfolio_manager._invalidate_portfolio_history()

        # Show FX rate warnings if any
        self._render_fx_warnings()

        # Show price fallback prompt if there's a pending one
        self._render_price_fallback_prompt(portfolio_manager)

        if not st.session_state.portfolio_loaded:
            st.warning("Please load or create a portfolio to view details.")
            return

        # Portfolio summary
        portfolio = portfolio_manager.current_portfolio

        # Use PortfolioHistory for on-demand value calculation (single source of truth)
        total_value = portfolio_manager.get_portfolio_value()
        positions = portfolio_manager.get_positions_with_prices()

        # Initialize variables (will be populated if positions exist)
        enriched = []
        base_currency = portfolio.base_currency
        start_of_year = date(date.today().year, 1, 1)

        # Prepare YTD reference prices from MarketDataStore (no network)
        ytd_start = date(date.today().year, 1, 1)
        today = date.today()
        try:
            # Get prices from centralized MarketDataStore
            market_data_store = portfolio_manager.market_data_store
            symbols = [pos.get("symbol") for pos in positions if pos.get("symbol")]

            ref_prices_by_symbol: Dict[str, Decimal] = {}
            curr_prices_by_symbol: Dict[str, Decimal] = {}

            for sym in symbols:
                if not sym or sym == "CASH":
                    continue

                # Get YTD start price (reference)
                ref_price = market_data_store.get_price_with_fallback(sym, ytd_start, fallback_days=30)
                if ref_price:
                    ref_prices_by_symbol[sym] = ref_price

                # Get current price
                curr_price = market_data_store.get_price_with_fallback(sym, today, fallback_days=7)
                if curr_price:
                    curr_prices_by_symbol[sym] = curr_price

        except Exception:
            ref_prices_by_symbol = {}
            curr_prices_by_symbol = {}

        # Disable automatic FX fetching - use cached rates only
        # FX rates are fetched when user clicks "Update Market Data"
        fetch_live = False

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
            # Show UI for setting prices on positions without prices
            self._render_positions_without_prices(portfolio_manager, positions)

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
            # Show market data store statistics
            market_data_store = portfolio_manager.market_data_store
            price_count = market_data_store.get_price_count()
            if price_count > 0:
                st.caption(f"🗓️ Market data: {price_count} price records")
            else:
                st.caption("🗓️ Market data: No data yet")

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
            if st.button("📈 Update Market Data", help="Fetch historical prices and store in MarketDataStore", type="primary"):
                with st.spinner(f"Updating market data for last {days_to_update} days..."):
                    try:
                        end_date = date.today()
                        start_date = end_date - timedelta(days=days_to_update)
                        logging.info(f"Updating market data from {start_date} to {end_date}")
                        results = portfolio_manager.update_market_data(start_date, end_date)
                        # Update freshness tracking
                        if market_data_service and portfolio_manager.current_portfolio:
                            market_data_service.refresh_all(portfolio_manager.current_portfolio)
                        successful = sum(1 for v in results.values() if v)
                        st.success(f"✅ Updated market data for {successful}/{len(results)} symbols")
                        st.rerun()
                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        logging.error(f"Failed to update market data: {error_details}")
                        st.error(f"❌ Failed to update: {str(e)}")

        with col3:
            if st.button("💰 Quick Refresh", help="Update current prices for all positions"):
                with st.spinner("Refreshing prices..."):
                    # Update current prices and store in MarketDataStore
                    results = portfolio_manager.update_current_prices()
                    if results:
                        successful = sum(1 for v in results.values() if v)
                        st.success(f"✅ Updated prices for {successful}/{len(results)} positions")
                    else:
                        st.warning("⚠️ No portfolio loaded or no positions")
                    st.rerun()

        with col4:
            st.caption("💡 **Market Data**: Fetches historical prices for analytics. **Quick Refresh**: Updates today's prices only.")

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
                        allow_fetch=fetch_live,
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
            today = date.today()
            # Use PortfolioHistory for YTD calculation
            ytd_history_df = portfolio_manager.get_portfolio_history(start_of_year, today)

            if not ytd_history_df.empty and len(ytd_history_df) >= 2:
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

                # Compute TWR using the calculator's DataFrame-based return function
                calc = FinancialMetricsCalculator(portfolio_manager.data_manager)
                daily_returns = calc.calculate_returns_from_df(ytd_history_df, "total_value", flows_float)

                if daily_returns:
                    # Geometric aggregation for period return
                    twr = 1.0
                    for r in daily_returns:
                        twr *= 1.0 + r
                    ytd_perf_pct = (twr - 1.0) * 100.0
                    ytd_col1.metric("YTD Performance (TWR)", f"{ytd_perf_pct:.2f}%", delta=f"{ytd_perf_pct:.2f}%")
                else:
                    ytd_col1.metric("YTD Performance (TWR)", "N/A")
            else:
                ytd_col1.metric("YTD Performance (TWR)", "N/A")
        except Exception:
            ytd_col1.metric("YTD Performance (TWR)", "N/A")

        # Unrealized P&L percentage from current positions
        try:
            # Calculate cost basis from quantity * average_cost since cost_basis isn't in the summary
            total_cost_basis = sum(
                float(pos.get("quantity", 0) or 0) * float(pos.get("average_cost", 0) or 0)
                for pos in positions
            )
            if total_cost_basis > 0:
                unrealized_pct = total_pnl / total_cost_basis * 100.0
                ytd_col2.metric("Unrealized P&L (%)", f"{unrealized_pct:.2f}%", delta=f"{unrealized_pct:.2f}%")
            else:
                ytd_col2.metric("Unrealized P&L (%)", "N/A")
        except Exception:
            ytd_col2.metric("Unrealized P&L (%)", "N/A")

        # Link to Analytics tab for detailed metrics
        st.caption("💡 For detailed performance metrics (Sharpe ratio, volatility, benchmark comparison), see the **Analytics** tab.")

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
                    allow_fetch=fetch_live,
                )

                # Convert unrealized PnL to base currency
                unreal_val_native = pos.get("unrealized_pnl") or Decimal("0")
                unreal_val_base = self._convert_to_base(
                    portfolio_manager,
                    unreal_val_native,
                    currency_code,
                    base_currency.value,
                    allow_fetch=fetch_live,
                )

                # Category classification
                category = self._classify_position(pos, instrument)

                # YTD market-only: compare current price to first YTD price from market data
                ytd_market_pnl_native = None
                ytd_market_pct = None
                try:
                    if symbol in ref_prices_by_symbol:
                        ref_price = Decimal(str(ref_prices_by_symbol[symbol]))
                        qty_dec = Decimal(str(pos.get("quantity") or 0))
                        # Prefer latest market data price if available; fallback to UI position price
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
                # Calculate from quantity * average_cost since cost_basis isn't in the summary
                qty_val = pos.get("quantity")
                avg_cost_val = pos.get("average_cost")
                total_buy_native = (
                    Decimal(str(qty_val)) * Decimal(str(avg_cost_val))
                    if qty_val is not None and avg_cost_val is not None
                    else None
                )
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
                        fx_summary = self._get_cash_fx_summary(portfolio_manager, allow_fetch=fetch_live).get(
                            curr
                        )
                        amt_base = self._convert_to_base(
                            portfolio_manager,
                            Decimal(str(amt)),
                            curr_code,
                            base_currency.value,
                            allow_fetch=fetch_live,
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

        from src.portfolio.models import Currency

        try:
            from_currency = Currency(from_currency_code)
            to_currency = Currency(base_currency_code)

            if not allow_fetch:
                # Only use cached FX rates - no network calls
                rate = portfolio_manager.data_manager.fx_cache.get_current_rate(from_currency, to_currency)
                if not rate:
                    rate = portfolio_manager.data_manager.fx_cache.get_rate(
                        from_currency, to_currency, date.today()
                    )
                if rate:
                    return Decimal(str(amount)) * rate
                # No cached rate available - return unconverted
                return Decimal(str(amount))

            # allow_fetch=True: Try real-time FX first
            rate = portfolio_manager.data_manager.get_exchange_rate(from_currency, to_currency)
            if rate:
                return Decimal(str(amount)) * rate

            # Fallback to historical FX for today (handles cases where live quote is missing)
            hist_rate = portfolio_manager.data_manager.get_historical_fx_rate_on(
                date.today(), from_currency, to_currency
            )
            if hist_rate:
                return Decimal(str(amount)) * hist_rate
        except Exception:
            pass
        return Decimal(str(amount))

    def _get_cash_fx_summary(self, portfolio_manager, allow_fetch: bool = False) -> Dict:
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
                fx = None
                if allow_fetch:
                    fx = (
                        portfolio_manager.data_manager.get_historical_fx_rate_on(
                            txn.timestamp.date(), cur, base
                        )
                        or portfolio_manager.data_manager.get_exchange_rate(cur, base)
                    )
                else:
                    # Only use cached FX rates
                    fx = portfolio_manager.data_manager.fx_cache.get_rate(
                        cur, base, txn.timestamp.date()
                    )
                if not fx:
                    fx = Decimal("1")
                    logging.warning(
                        f"FX rate unavailable for {cur}->{base} on {txn.timestamp.date()}, "
                        f"using 1.0 fallback - FX summary may be inaccurate"
                    )
                    if "fx_fallback_warnings" not in st.session_state:
                        st.session_state.fx_fallback_warnings = []
                    warning_msg = f"{cur.value}/{base.value} on {txn.timestamp.date()}"
                    if warning_msg not in st.session_state.fx_fallback_warnings:
                        st.session_state.fx_fallback_warnings.append(warning_msg)
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
            if cur == base:
                rate = Decimal("1")
            else:
                rate = None
                if allow_fetch:
                    rate = (
                        portfolio_manager.data_manager.get_exchange_rate(cur, base)
                        or portfolio_manager.data_manager.get_historical_fx_rate_on(
                            date.today(), cur, base
                        )
                    )
                else:
                    # Only use cached FX rates - try current rate from cache
                    rate = portfolio_manager.data_manager.fx_cache.get_current_rate(cur, base)
                    if not rate:
                        # Fall back to most recent cached rate
                        rate = portfolio_manager.data_manager.fx_cache.get_rate(
                            cur, base, date.today()
                        )
                if not rate:
                    rate = Decimal("1")
                    logging.warning(
                        f"Current FX rate unavailable for {cur}->{base}, "
                        f"using 1.0 fallback - FX summary may be inaccurate"
                    )
                    if "fx_fallback_warnings" not in st.session_state:
                        st.session_state.fx_fallback_warnings = []
                    warning_msg = f"{cur.value}/{base.value} (current)"
                    if warning_msg not in st.session_state.fx_fallback_warnings:
                        st.session_state.fx_fallback_warnings.append(warning_msg)
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
                            # Check if the position has a current price
                            position = portfolio_manager.current_portfolio.positions.get(label)
                            if position and position.current_price is None:
                                # Set up the price fallback prompt
                                st.session_state.pending_price_fallback = {
                                    "symbol": label,
                                    "purchase_price": float(price),
                                    "date": trade_date.isoformat(),
                                    "instrument_name": position.instrument.name,
                                }
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
        st.caption("Historical performance analysis and risk metrics. See **Portfolio** tab for current positions and values.")

        # Clear market data cache to pick up any external updates (e.g., from MCP server)
        # This ensures fresh data is loaded from SQLite on each page view
        portfolio_manager.market_data_store.clear_cache()
        # Also invalidate portfolio history cache to recalculate values with fresh prices
        portfolio_manager._invalidate_portfolio_history()

        # Show FX rate warnings if any
        self._render_fx_warnings()

        if not st.session_state.portfolio_loaded:
            st.warning("Please load a portfolio to view analytics.")
            return

        # Analysis Configuration Panel
        with st.container():
            st.markdown("#### ⚙️ Analysis Configuration")
            config_col1, config_col2, config_col3, config_col4, config_col5 = st.columns([2, 1, 1, 1, 1])

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

            with config_col5:
                view_mode_options = {
                    "All Assets": "all",
                    "Equities Only": "equities_only",
                    "Fixed Income Only": "fixed_income_only",
                }
                selected_view_label = st.selectbox(
                    "📁 View Mode",
                    list(view_mode_options.keys()),
                    index=0,
                    help="Filter analytics by asset category. Equities/Fixed Income views include realized gains from sold positions."
                )
                selected_view_mode = view_mode_options[selected_view_label]

        # Normalize date range
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = default_range

        if start_date > end_date:
            st.error("Start date must be on or before end date.")
            return

        # Get data from PortfolioHistory (supports filtered views)
        history_df = portfolio_manager.get_portfolio_history_filtered(
            start_date, end_date, view_mode=selected_view_mode
        )

        # Show indicator if filtered view is active
        if selected_view_mode != "all":
            st.info(f"📊 Viewing **{selected_view_label}** - includes realized gains from sold positions in this category")

        if history_df.empty or len(history_df) < 2:
            st.warning(
                "Insufficient data for selected period. Click 'Update Market Data' in the Portfolio tab to fetch historical prices."
            )
            return

        # Show data freshness
        market_data_store = portfolio_manager.market_data_store
        price_count = market_data_store.get_price_count()
        if price_count > 0:
            st.success(f"✅ Market data: {price_count} price records available")
        else:
            st.warning("⚠️ No market data available. Use 'Update Market Data' to fetch prices.")

        # Calculate metrics using DataFrame-based methods from metrics calculator
        with st.spinner("Calculating time-weighted metrics..."):
            # Get external cash flows for the period
            external_cash_flows = portfolio_manager.get_external_cash_flows_by_day(
                start_date, end_date
            )

            # Convert cash flows to float for metrics calculator
            cash_flows_float = {d: float(v) for d, v in external_cash_flows.items()}

            # Calculate returns using DataFrame
            portfolio_returns_twr = metrics_calculator.calculate_returns_from_df(
                history_df, "total_value", cash_flows_float
            )

            if not portfolio_returns_twr:
                # Check if portfolio has value
                has_positive_values = (history_df["total_value"] > 0).sum() >= 2
                if not has_positive_values:
                    st.warning("⚠️ Cannot calculate returns: Portfolio needs at least 2 days with positive values.")

                    # Provide context based on view mode
                    if selected_view_mode != "all":
                        st.info("💡 **Filtered View Note:** In category-specific views, 'total_value' represents pure trading performance:")
                        st.info("   • Total Value = Position Value + Attributed Cash")
                        st.info("   • Attributed Cash = Sale proceeds − Purchase costs (starts at 0)")
                        st.info("   • If positions haven't gained/lost value, total may be near zero")

                        # Show breakdown for filtered views
                        if "positions_value" in history_df.columns and "attributed_cash" in history_df.columns:
                            st.write("**Value breakdown (first 5 days):**")
                            for idx in history_df.index[:5]:
                                pos_val = history_df.loc[idx, "positions_value"]
                                attr_cash = history_df.loc[idx, "attributed_cash"]
                                total = history_df.loc[idx, "total_value"]
                                date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
                                st.write(f"  {date_str}: Positions=${float(pos_val):,.2f}, Attributed Cash=${float(attr_cash):,.2f}, Total=${float(total):,.2f}")
                            if len(history_df) > 5:
                                st.write(f"  ... and {len(history_df) - 5} more days")

                            # Check if this is a "no positions" vs "no gains" situation
                            avg_positions = history_df["positions_value"].mean()
                            if avg_positions < 1:
                                st.warning(f"⚠️ No {selected_view_label.lower()} positions found in this period.")
                            else:
                                st.info(f"📊 Average position value: ${avg_positions:,.2f}. The near-zero total indicates positions haven't gained/lost significantly.")
                    else:
                        st.info("💡 This typically happens when:")
                        st.info("   • Portfolio is just starting")
                        st.info("   • Not enough market data has been fetched")
                        st.info("   • Portfolio values are zero or negative")

                        # Show values for context
                        st.write("**Portfolio values:**")
                        for idx in history_df.index[:5]:
                            val = history_df.loc[idx, "total_value"]
                            date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
                            st.write(f"  {date_str}: ${float(val):,.2f}")
                        if len(history_df) > 5:
                            st.write(f"  ... and {len(history_df) - 5} more days")
                else:
                    st.error("Could not calculate time-weighted returns.")

                return

            # Calculate comprehensive metrics using DataFrame-based method
            comprehensive_metrics = metrics_calculator.calculate_metrics_from_df(
                history_df, "total_value", benchmark, cash_flows_float
            )

            if "error" in comprehensive_metrics:
                st.error(comprehensive_metrics["error"])
                return

            # Extract key metrics
            metrics = {
                "total_return_twr": comprehensive_metrics.get("total_return", 0.0),
                "annualized_return_twr": comprehensive_metrics.get("annualized_return", 0.0),
                "volatility": comprehensive_metrics.get("volatility", 0.0),
                "sharpe_ratio": comprehensive_metrics.get("sharpe_ratio", 0.0),
                "sortino_ratio": comprehensive_metrics.get("sortino_ratio", 0.0),
                "max_drawdown": comprehensive_metrics.get("max_drawdown", 0.0),
                "max_drawdown_duration": comprehensive_metrics.get("max_drawdown_duration", 0),
                "var_5pct": comprehensive_metrics.get("var_5pct", 0.0),
                "cvar_5pct": comprehensive_metrics.get("cvar_5pct", 0.0),
                "calmar_ratio": comprehensive_metrics.get("calmar_ratio", 0.0),
                "beta": comprehensive_metrics.get("beta", 0.0),
                "alpha": comprehensive_metrics.get("alpha", 0.0),
                "information_ratio": comprehensive_metrics.get("information_ratio", 0.0),
                "benchmark_return": comprehensive_metrics.get("benchmark_return", 0.0),
                "benchmark_volatility": comprehensive_metrics.get("benchmark_volatility", 0.0),
                "benchmark_available": comprehensive_metrics.get("benchmark_available", False),
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

                # Get YTD data from PortfolioHistory
                ytd_df = portfolio_manager.get_portfolio_history(ytd_start, end_date)

                if not ytd_df.empty and len(ytd_df) >= 2:
                    ytd_flows = portfolio_manager.get_external_cash_flows_by_day(
                        ytd_start, end_date
                    )
                    ytd_flows_f = {d: float(v) for d, v in ytd_flows.items()}

                    # Calculate YTD returns using DataFrame
                    daily_returns = metrics_calculator.calculate_returns_from_df(
                        ytd_df, "total_value", ytd_flows_f
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
                # Calculate simple return (MWR approximation)
                first_value = float(history_df["total_value"].iloc[0])
                last_value = float(history_df["total_value"].iloc[-1])
                total_cash_flows = sum(cash_flows_float.values()) if cash_flows_float else 0.0

                # Simple MWR: (End - Start - Cash Flows) / (Start + weighted cash flows)
                if first_value > 0:
                    mwr_period_pct = ((last_value - first_value - total_cash_flows) / first_value) * 100.0
                else:
                    mwr_period_pct = 0.0

                # Use the same TWR calculation (total return for period)
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

        # Align benchmark prices to history dates with forward-fill
        bench_prices_aligned: List[Optional[float]] = []
        last_px: Optional[float] = None
        for idx in history_df.index:
            day = idx.date() if hasattr(idx, 'date') else idx
            px = bench_map.get(day, last_px)
            bench_prices_aligned.append(px)
            if px is not None:
                last_px = px

        # Portfolio value chart using PortfolioHistory DataFrame
        self.plot_portfolio_value_chart(
            history_df,
            display_currency_code,
            portfolio_manager,
            start_date,
            end_date,
            benchmark_symbol=benchmark,
            benchmark_prices_aligned=bench_prices_aligned,
            portfolio_returns_twr=portfolio_returns_twr,
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

            # Create dates list from history_df (skip first date since returns start from second day)
            dates = list(history_df.index)

            fig3 = go.Figure()
            fig3.add_trace(
                go.Scatter(
                    x=dates[
                        1:
                    ],  # Skip first date since returns start from second day
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

    def plot_portfolio_value_chart(
        self,
        history_df: pd.DataFrame,
        display_currency_code: str,
        portfolio_manager,
        start_date: date,
        end_date: date,
        benchmark_symbol: Optional[str] = None,
        benchmark_prices_aligned: Optional[List[Optional[float]]] = None,
        portfolio_returns_twr: Optional[List[float]] = None,
    ):
        """Plot portfolio value chart using PortfolioHistory DataFrame."""
        import plotly.graph_objects as go

        if history_df.empty or len(history_df) < 2:
            st.warning("Insufficient data for chart")
            return

        # Extract dates and values
        dates = [idx.date() if hasattr(idx, 'date') else idx for idx in history_df.index]
        portfolio_values = [float(v) for v in history_df["total_value"].values]

        # Create main portfolio line chart
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=portfolio_values,
                mode="lines",
                name="Portfolio",
                line=dict(color="blue", width=2),
            )
        )

        # Add benchmark if available
        if benchmark_symbol and benchmark_prices_aligned:
            first_bench = next((p for p in benchmark_prices_aligned if p is not None), None)
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
            title=f"Portfolio Value Over Time ({display_currency_code})",
            xaxis_title="Date",
            yaxis_title=f"Value ({display_currency_code})",
            height=420,
            showlegend=True,
        )
        fig.update_xaxes(range=[start_date, end_date])
        st.plotly_chart(fig, use_container_width=True)

        # Cumulative returns chart using TWR
        if portfolio_returns_twr:
            cumulative = []
            running_product = 1.0
            for daily_return in portfolio_returns_twr:
                running_product *= 1 + daily_return
                cumulative.append((running_product - 1.0) * 100.0)

            # Use dates from day 2 onwards (first return is between day 1 and day 2)
            return_dates = dates[1:len(cumulative)+1]

            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=return_dates,
                    y=cumulative,
                    mode="lines",
                    name="Portfolio Cumulative Return",
                    line=dict(color="green", width=2),
                )
            )

            fig2.update_layout(
                title="Cumulative Return (%)",
                xaxis_title="Date",
                yaxis_title="Cumulative Return (%)",
                height=350,
                showlegend=True,
            )
            st.plotly_chart(fig2, use_container_width=True)

    def render_optimization(self, portfolio_manager, market_data_service):
        """Render portfolio optimization with weight suggestions and visualizations."""
        st.header("⚖️ Portfolio Optimization")

        if not st.session_state.portfolio_loaded:
            st.warning("Please load a portfolio to run optimization.")
            return

        portfolio = portfolio_manager.current_portfolio
        if not portfolio:
            st.warning("No portfolio currently loaded.")
            return

        positions = portfolio.positions
        if len(positions) < 2:
            st.info("Need at least 2 positions to run portfolio optimization.")
            return

        # Configuration section
        st.subheader("Configuration")
        col1, col2 = st.columns(2)

        with col1:
            lookback_days = st.slider(
                "Lookback Period (days)",
                min_value=60,
                max_value=504,
                value=252,
                step=21,
                help="Historical data used for optimization (252 = 1 year)"
            )

            risk_free_rate = st.slider(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=4.0,
                step=0.25,
                help="Annual risk-free rate for Sharpe ratio calculation"
            ) / 100

        with col2:
            position_symbols = list(positions.keys())
            locked_symbols = st.multiselect(
                "Lock Positions",
                options=position_symbols,
                default=[],
                help="These positions will keep their current weights"
            )

            include_cash = st.checkbox(
                "Include cash in rebalancing",
                value=True,
                help="Consider cash as part of the portfolio. Enables cash allocation to achieve target volatility and includes existing cash in trade calculations."
            )

        # Risk constraint section - applies to both HRP and Markowitz
        st.subheader("📊 Risk Constraint & Currency")
        st.caption("Both HRP and Markowitz can target specific volatility by blending with cash.")

        risk_col1, risk_col2, risk_col3 = st.columns(3)

        with risk_col1:
            objective_options = {
                "Maximize Sharpe Ratio": OptimizationObjective.MAX_SHARPE,
                "Minimize Volatility": OptimizationObjective.MIN_VOLATILITY,
                "Target Specific Volatility": OptimizationObjective.EFFICIENT_RISK,
            }
            selected_objective = st.selectbox(
                "Optimization Objective",
                options=list(objective_options.keys()),
                index=0,
                help="Choose optimization objective (applies to both methods)"
            )
            objective = objective_options[selected_objective]

        with risk_col2:
            target_volatility = None
            if objective == OptimizationObjective.EFFICIENT_RISK:
                target_volatility = st.slider(
                    "Target Annual Volatility (%)",
                    min_value=1.0,
                    max_value=40.0,
                    value=15.0,
                    step=1.0,
                    help="Target volatility - if below minimum achievable, will blend with cash"
                ) / 100
                if include_cash:
                    st.caption(f"💡 Will recommend cash allocation if needed")
                else:
                    st.warning("⚠️ Cash disabled - target volatility may not be achievable")
            else:
                st.info(f"📈 Using: {selected_objective}")

        with risk_col3:
            # Optimization currency selector
            currency_options = [c.value for c in Currency]
            default_currency = portfolio.base_currency.value
            default_idx = currency_options.index(default_currency) if default_currency in currency_options else 0

            selected_opt_currency = st.selectbox(
                "Optimization Currency",
                options=currency_options,
                index=default_idx,
                help="Currency perspective for optimization. Different currencies show different volatility due to FX effects."
            )
            optimization_currency = Currency(selected_opt_currency)

            if optimization_currency != portfolio.base_currency:
                st.caption(f"⚠️ Optimizing from {optimization_currency.value} perspective (portfolio base: {portfolio.base_currency.value})")

        # Run optimization button
        if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
            with st.spinner("Fetching price data and running optimization..."):
                try:
                    # Update prices first
                    portfolio_manager.update_current_prices()

                    # Calculate total portfolio value
                    total_value = sum(
                        pos.market_value or (pos.quantity * pos.average_cost)
                        for pos in positions.values()
                        if pos.quantity > 0
                    )

                    # Get cash balances if including cash
                    cash_balances = portfolio.cash_balances if include_cash else None

                    # Create optimizer with storage for historical data
                    optimizer = PortfolioOptimizer(
                        portfolio_manager.data_manager,
                        base_currency=portfolio.base_currency,
                        storage=portfolio_manager.storage,
                        portfolio_id=portfolio.id,
                    )
                    results = optimizer.compare_methods(
                        positions=positions,
                        locked_symbols=locked_symbols if locked_symbols else None,
                        lookback_days=lookback_days,
                        risk_free_rate=risk_free_rate,
                        total_portfolio_value=total_value,
                        cash_balances=cash_balances,
                        objective=objective,
                        target_volatility=target_volatility,
                        include_cash=include_cash,
                        optimization_currency=optimization_currency,
                    )

                    # Store results in session state
                    st.session_state.optimization_results = results
                    st.session_state.optimization_risk_free_rate = risk_free_rate
                    st.session_state.optimization_locked = locked_symbols
                    st.session_state.optimization_base_currency = portfolio.base_currency.value
                    st.session_state.optimization_currency = optimization_currency.value
                    st.session_state.optimization_include_cash = include_cash
                    st.success("Optimization complete!")

                except Exception as e:
                    st.error(f"Optimization failed: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
                    return

        # Display results if available
        if "optimization_results" not in st.session_state:
            st.info("Click 'Run Optimization' to generate weight suggestions.")
            return

        results = st.session_state.optimization_results
        locked = st.session_state.get("optimization_locked", [])
        base_currency = st.session_state.get("optimization_base_currency", "USD")
        opt_currency = st.session_state.get("optimization_currency", base_currency)

        # Show optimization currency if different from base
        if opt_currency != base_currency:
            st.info(f"📌 Results shown from **{opt_currency}** investor perspective (portfolio base: {base_currency})")

        # Results tabs
        result_tab1, result_tab2, result_tab3 = st.tabs([
            "📊 Weight Comparison", "📈 Metrics", "📋 Rebalancing Trades"
        ])

        with result_tab1:
            self._render_weight_comparison(results, locked)

        with result_tab2:
            self._render_optimization_metrics(results, base_currency, opt_currency)

        with result_tab3:
            self._render_rebalancing_trades(results, base_currency)

    def _render_weight_comparison(self, results, locked_symbols):
        """Render weight comparison charts."""
        st.subheader("Weight Allocation Comparison")

        hrp_result = results.get(OptimizationMethod.HRP)
        mk_result = results.get(OptimizationMethod.MARKOWITZ)

        if not hrp_result or not hrp_result.weights:
            st.warning("No HRP results available.")
            return

        # Prepare data for comparison
        symbols = sorted(hrp_result.weights.keys())
        current_weights = [hrp_result.current_weights.get(s, 0) * 100 for s in symbols]
        hrp_weights = [hrp_result.weights.get(s, 0) * 100 for s in symbols]
        mk_weights = [mk_result.weights.get(s, 0) * 100 if mk_result and mk_result.weights else 0 for s in symbols]

        # Add cash if there's any cash involvement (current or target)
        hrp_cash = hrp_result.cash_weight * 100 if hrp_result.cash_weight else 0
        mk_cash = (mk_result.cash_weight * 100 if mk_result and mk_result.cash_weight else 0)

        # Calculate current cash weight from the difference between total weights
        # If current weights don't sum to 100%, the remainder is cash
        current_risky_total = sum(hrp_result.current_weights.values()) if hrp_result.current_weights else 1.0
        current_cash_weight = max(0, (1.0 - current_risky_total)) * 100

        # Show cash row if there's current cash OR target cash allocation
        if current_cash_weight > 0.1 or hrp_cash > 0.1 or mk_cash > 0.1:
            symbols = symbols + ["CASH"]
            current_weights.append(current_cash_weight)
            hrp_weights.append(hrp_cash)
            mk_weights.append(mk_cash)

        # Create comparison bar chart
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Current',
            x=symbols,
            y=current_weights,
            marker_color='lightgray',
            text=[f"{w:.1f}%" for w in current_weights],
            textposition='outside'
        ))

        fig.add_trace(go.Bar(
            name='HRP (Recommended)',
            x=symbols,
            y=hrp_weights,
            marker_color='#2ca02c',
            text=[f"{w:.1f}%" for w in hrp_weights],
            textposition='outside'
        ))

        fig.add_trace(go.Bar(
            name='Markowitz',
            x=symbols,
            y=mk_weights,
            marker_color='#1f77b4',
            text=[f"{w:.1f}%" for w in mk_weights],
            textposition='outside'
        ))

        fig.update_layout(
            title="Portfolio Weight Comparison",
            xaxis_title="Asset",
            yaxis_title="Weight (%)",
            barmode='group',
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Pie charts side by side
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**HRP Target Allocation**")
            fig_hrp = px.pie(
                values=hrp_weights,
                names=symbols,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_hrp.update_layout(height=350, showlegend=True)
            st.plotly_chart(fig_hrp, use_container_width=True)

        with col2:
            st.markdown("**Markowitz Target Allocation**")
            fig_mk = px.pie(
                values=mk_weights,
                names=symbols,
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_mk.update_layout(height=350, showlegend=True)
            st.plotly_chart(fig_mk, use_container_width=True)

        # Locked symbols indicator
        if locked_symbols:
            st.info(f"🔒 Locked positions (unchanged): {', '.join(locked_symbols)}")

    def _render_optimization_metrics(self, results, base_currency: str = "USD", opt_currency: str = None):
        """Render optimization metrics comparison."""
        opt_currency = opt_currency or base_currency
        st.subheader(f"Risk-Return Metrics ({opt_currency})")

        hrp_result = results.get(OptimizationMethod.HRP)
        mk_result = results.get(OptimizationMethod.MARKOWITZ)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### HRP (Recommended)")
            if hrp_result and hrp_result.weights:
                m1, m2, m3 = st.columns(3)
                with m1:
                    exp_ret = hrp_result.expected_annual_return
                    st.metric("Expected Return", f"{exp_ret:.1%}" if exp_ret else "N/A")
                with m2:
                    st.metric("Volatility", f"{hrp_result.annual_volatility:.1%}")
                with m3:
                    sharpe = hrp_result.sharpe_ratio
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe else "N/A")

                if hrp_result.cash_weight > 0:
                    st.info(f"💵 Cash weight: {hrp_result.cash_weight:.1%}")

                if hrp_result.warnings:
                    for warn in hrp_result.warnings:
                        st.warning(warn)
            else:
                st.error("HRP optimization failed")

        with col2:
            st.markdown("### Markowitz")
            if mk_result and mk_result.weights:
                m1, m2, m3 = st.columns(3)
                with m1:
                    exp_ret = mk_result.expected_annual_return
                    st.metric("Expected Return", f"{exp_ret:.1%}" if exp_ret else "N/A")
                with m2:
                    st.metric("Volatility", f"{mk_result.annual_volatility:.1%}")
                with m3:
                    sharpe = mk_result.sharpe_ratio
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}" if sharpe else "N/A")

                # Show if risk-constrained
                if mk_result.target_volatility:
                    st.info(f"🎯 Target volatility: {mk_result.target_volatility:.1%}")

                if mk_result.cash_weight > 0:
                    st.info(f"💵 Cash weight: {mk_result.cash_weight:.1%}")

                if mk_result.warnings:
                    for warn in mk_result.warnings:
                        st.warning(warn)
            else:
                st.error("Markowitz optimization failed")

        # Risk-Return scatter plot with all assets and Capital Allocation Lines
        st.markdown("### Risk-Return Comparison")

        # Get settings from session state
        risk_free_rate = st.session_state.get("optimization_risk_free_rate", 0.04)
        risk_free_pct = risk_free_rate * 100
        include_cash_in_plot = st.session_state.get("optimization_include_cash", True)

        # Build plot data with individual assets + portfolio options
        plot_data = []

        # Add individual assets from asset_metrics
        if hrp_result and hrp_result.asset_metrics:
            for am in hrp_result.asset_metrics:
                plot_data.append({
                    "Label": am.symbol,
                    "Type": "Individual Asset",
                    "Volatility": am.volatility * 100,
                    "Return": am.expected_return * 100,
                    "Weight": am.current_weight * 100,
                })

        # Add portfolio options
        if hrp_result and hrp_result.weights and hrp_result.expected_annual_return:
            plot_data.append({
                "Label": "HRP Portfolio",
                "Type": "Optimized Portfolio",
                "Volatility": hrp_result.annual_volatility * 100,
                "Return": hrp_result.expected_annual_return * 100,
                "Weight": 100.0,  # Full portfolio
            })
        if mk_result and mk_result.weights and mk_result.expected_annual_return:
            plot_data.append({
                "Label": "Markowitz Portfolio",
                "Type": "Optimized Portfolio",
                "Volatility": mk_result.annual_volatility * 100,
                "Return": mk_result.expected_annual_return * 100,
                "Weight": 100.0,
            })

        if plot_data:
            df = pd.DataFrame(plot_data)

            # Create scatter plot
            fig = go.Figure()

            # Add Cash point and CAL lines only when cash is included
            portfolio_colors = {"HRP Portfolio": "#2ca02c", "Markowitz Portfolio": "#1f77b4"}
            portfolios_df = df[df["Type"] == "Optimized Portfolio"]

            if include_cash_in_plot:
                # Add Cash point at (0%, risk_free_rate%)
                fig.add_trace(go.Scatter(
                    x=[0],
                    y=[risk_free_pct],
                    mode="markers+text",
                    name="Cash (Risk-Free)",
                    text=["Cash"],
                    textposition="top right",
                    marker=dict(
                        size=20,
                        color="#FFD700",  # Gold color for cash
                        symbol="diamond",
                        line=dict(width=2, color="black")
                    ),
                    hovertemplate="<b>Cash</b><br>Volatility: 0%<br>Return: %{y:.1f}% (risk-free)<extra></extra>",
                ))

                # Add Capital Allocation Lines from cash to each optimized portfolio
                for _, row in portfolios_df.iterrows():
                    portfolio_vol = row["Volatility"]
                    portfolio_ret = row["Return"]
                    portfolio_label = row["Label"]
                    color = portfolio_colors.get(portfolio_label, "#ff7f0e")

                    # Draw CAL line from cash to portfolio
                    fig.add_trace(go.Scatter(
                        x=[0, portfolio_vol],
                        y=[risk_free_pct, portfolio_ret],
                        mode="lines",
                        name=f"CAL ({portfolio_label.split()[0]})",
                        line=dict(color=color, width=2, dash="dash"),
                        hoverinfo="skip",
                        showlegend=True,
                    ))

                    # Add intermediate points along the CAL
                    num_points = 50
                    for i in range(1, num_points + 1):
                        fraction = i / (num_points + 1)  # Points between 0 and 1
                        point_vol = fraction * portfolio_vol
                        point_ret = risk_free_pct + fraction * (portfolio_ret - risk_free_pct)
                        cash_pct = (1 - fraction) * 100

                        fig.add_trace(go.Scatter(
                            x=[point_vol],
                            y=[point_ret],
                            mode="markers",
                            name=f"{portfolio_label.split()[0]} @ {point_vol:.0f}% vol",
                            marker=dict(
                                size=10,
                                color=color,
                                opacity=0.5,
                                symbol="circle",
                            ),
                            hovertemplate=f"<b>{portfolio_label.split()[0]} blend</b><br>Volatility: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<br>Cash: {cash_pct:.0f}%<extra></extra>",
                            showlegend=False,
                        ))

            # Add individual assets as smaller points
            assets_df = df[df["Type"] == "Individual Asset"]
            if not assets_df.empty:
                fig.add_trace(go.Scatter(
                    x=assets_df["Volatility"],
                    y=assets_df["Return"],
                    mode="markers+text",
                    name="Individual Assets",
                    text=assets_df["Label"],
                    textposition="top center",
                    marker=dict(
                        size=assets_df["Weight"] * 2 + 8,  # Size based on weight
                        color="rgba(128, 128, 128, 0.6)",
                        line=dict(width=1, color="gray")
                    ),
                    hovertemplate="<b>%{text}</b><br>Volatility: %{x:.1f}%<br>Return: %{y:.1f}%<br>Weight: %{customdata:.1f}%<extra></extra>",
                    customdata=assets_df["Weight"],
                ))

            # Add optimized portfolios as larger, colored points (stars)
            for _, row in portfolios_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row["Volatility"]],
                    y=[row["Return"]],
                    mode="markers+text",
                    name=row["Label"],
                    text=[row["Label"]],
                    textposition="top center",
                    marker=dict(
                        size=25,
                        color=portfolio_colors.get(row["Label"], "#ff7f0e"),
                        symbol="star",
                        line=dict(width=2, color="white")
                    ),
                    hovertemplate="<b>%{text}</b><br>Volatility: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
                ))

            plot_title = "Risk-Return Trade-off"
            if include_cash_in_plot:
                plot_title += " with Capital Allocation Lines"

            fig.update_layout(
                title=plot_title,
                xaxis_title="Annual Volatility (%)",
                yaxis_title="Expected Annual Return (%)",
                height=550,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

            if include_cash_in_plot:
                st.caption(
                    "💡 **Stars**: Optimized portfolios (100% risky). "
                    "**Diamond**: Cash (risk-free). "
                    "**Dashed lines**: Capital Allocation Lines showing risk-return at different cash blends. "
                    "**Small circles**: Intermediate blend points (hover for cash %)."
                )
            else:
                st.caption(
                    "💡 **Stars**: Optimized portfolios. "
                    "**Gray circles**: Individual assets (size = current weight)."
                )

    def _render_rebalancing_trades(self, results, base_currency: str = "USD"):
        """Render suggested rebalancing trades."""
        st.subheader("Suggested Rebalancing Trades")

        hrp_result = results.get(OptimizationMethod.HRP)

        if not hrp_result or not hrp_result.rebalancing_trades:
            st.info("No rebalancing trades needed or optimization hasn't been run yet.")
            return

        st.markdown("**Based on HRP (Recommended) allocation:**")

        # Show cash weight info if applicable
        if hrp_result.cash_weight > 0:
            st.info(f"💵 Cash is {hrp_result.cash_weight:.1%} of portfolio (kept separate from optimization)")

        # Create trades table with currency info
        trades_data = []
        for trade in hrp_result.rebalancing_trades:
            # Handle both old and new RebalancingTrade format
            currency = getattr(trade, "currency", base_currency)
            est_value_native = getattr(trade, "estimated_value_native", trade.estimated_value)

            trades_data.append({
                "Symbol": trade.symbol,
                "Action": trade.action,
                "Shares": f"{trade.shares:.0f}",
                "Value (Base)": f"{trade.estimated_value:,.0f} {base_currency}",
                "Value (Native)": f"{est_value_native:,.0f} {currency}" if currency != base_currency else "-",
                "Currency": currency,
                "Current %": f"{trade.current_weight:.1%}",
                "Target %": f"{trade.target_weight:.1%}",
                "Change": f"{(trade.target_weight - trade.current_weight):+.1%}"
            })

        df = pd.DataFrame(trades_data)

        # Style the dataframe
        def highlight_action(val):
            if val == "BUY":
                return "background-color: #d4edda; color: #155724"  # Green
            elif val == "SELL":
                return "background-color: #f8d7da; color: #721c24"  # Red
            elif val == "HOLD":
                return "background-color: #cce5ff; color: #004085"  # Blue for cash hold
            elif val == "DEPLOY":
                return "background-color: #d4edda; color: #155724"  # Green - deploy cash into assets
            elif val == "REDUCE":
                return "background-color: #fff3cd; color: #856404"  # Yellow for reduce
            return ""

        styled_df = df.style.map(highlight_action, subset=["Action"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Summary
        total_buy = sum(t.estimated_value for t in hrp_result.rebalancing_trades if t.action == "BUY")
        total_sell = sum(t.estimated_value for t in hrp_result.rebalancing_trades if t.action == "SELL")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total to Buy", f"{total_buy:,.0f} {base_currency}")
        with col2:
            st.metric("Total to Sell", f"{total_sell:,.0f} {base_currency}")
        with col3:
            net = total_buy - total_sell
            st.metric("Net Cash Flow", f"{net:,.0f} {base_currency}", delta=f"{'Inflow' if net < 0 else 'Outflow'}")

        # Check for multi-currency trades
        currencies = set(t.currency for t in hrp_result.rebalancing_trades if hasattr(t, "currency"))
        if len(currencies) > 1:
            st.info(f"💱 Multi-currency rebalancing: {', '.join(sorted(currencies))}. Values shown in both base ({base_currency}) and native currency.")

        st.caption("⚠️ These are suggestions only. Consider transaction costs, taxes, FX rates, and market conditions before trading.")

    def render_scenarios(self, portfolio_manager, metrics_calculator):
        """Render portfolio scenarios and what-if analysis."""
        st.header("🔮 Portfolio Scenarios")

        if not st.session_state.portfolio_loaded:
            st.warning("Please load a portfolio to run scenario analysis.")
            return

        if not portfolio_manager.current_portfolio:
            st.warning("No portfolio currently loaded. Please load or create a portfolio first.")
            return

        current_portfolio = portfolio_manager.current_portfolio

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

        # Get current portfolio value
        try:
            current_portfolio_value = portfolio_manager.get_portfolio_value()
            if current_portfolio_value is None or current_portfolio_value <= 0:
                st.error("Could not calculate current portfolio value. Update market data first.")
                return
        except Exception as e:
            st.error(f"Could not get current portfolio value: {e}")
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

            # Create a snapshot for the simulation (run_scenario_simulation expects PortfolioSnapshot)
            try:
                portfolio_snapshot = portfolio_manager.create_current_snapshot()
            except Exception as e:
                st.error(f"Error creating portfolio snapshot: {e}")
                return

            # Get predefined scenarios
            scenarios = engine.create_predefined_scenarios(float(current_portfolio_value))

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
                    result = engine.run_scenario_simulation(portfolio_snapshot, config)
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

        # Add starting value line (get from first simulation result)
        first_result = next(iter(simulation_results.values()))
        start_value = float(first_result.start_value)
        fig.add_hline(
            y=start_value,
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
        """Render a simplified what-if analysis section."""
        st.subheader("🧪 Quick What-If Analysis")
        st.markdown("Test how adding a new investment would impact your portfolio.")

        # Simple single-flow interface
        col1, col2 = st.columns(2)

        with col1:
            # Stock selection
            symbol = st.text_input(
                "Stock Symbol",
                placeholder="e.g., NVDA, AAPL, MSFT",
                key="whatif_symbol"
            ).upper().strip()

            # Investment amount (simple dollar input)
            investment_amount = st.number_input(
                "Investment Amount ($)",
                min_value=100.0,
                value=5000.0,
                step=500.0,
                key="whatif_amount",
                help="How much you want to invest"
            )

            # Show current portfolio context
            current_value = float(portfolio_manager.get_portfolio_value() or 0)
            if current_value > 0:
                allocation_pct = (investment_amount / current_value) * 100
                st.caption(f"This would be **{allocation_pct:.1f}%** of your portfolio")

        with col2:
            # Market scenario (simplified)
            scenario = st.selectbox(
                "Market Outlook",
                [
                    ("🟢 Optimistic", "optimistic"),
                    ("🟡 Normal", "likely"),
                    ("🟠 Cautious", "pessimistic"),
                ],
                format_func=lambda x: x[0],
                index=1,
                key="whatif_scenario"
            )[1]

            # Time horizon
            time_horizon = st.selectbox(
                "Time Horizon",
                [
                    ("6 months", 0.5),
                    ("1 year", 1.0),
                    ("2 years", 2.0),
                    ("5 years", 5.0),
                ],
                format_func=lambda x: x[0],
                index=1,
                key="whatif_horizon"
            )[1]

            # Scenario explanation
            scenario_info = {
                "optimistic": "Assumes 12% annual return, lower volatility",
                "likely": "Assumes 8% annual return, typical volatility",
                "pessimistic": "Assumes 3% annual return, higher volatility",
            }
            st.caption(scenario_info.get(scenario, ""))

        # Run button
        if st.button("🚀 Analyze Investment", type="primary", key="whatif_run", use_container_width=True):
            if not symbol:
                st.warning("Please enter a stock symbol.")
                return

            try:
                from src.agents.tools import HypotheticalPositionTool

                tool = HypotheticalPositionTool(portfolio_manager)

                # Fetch current market price for the symbol
                with st.spinner(f"Fetching current price for {symbol}..."):
                    current_price = portfolio_manager.data_manager.get_current_price(symbol)

                if current_price is None:
                    st.error(f"Could not fetch current price for {symbol}. Please check the symbol is valid.")
                    return

                current_price_float = float(current_price)
                quantity = investment_amount / current_price_float

                st.info(f"Current price for {symbol}: ${current_price_float:,.2f} → {quantity:,.2f} shares")

                with st.spinner(f"Analyzing {symbol}..."):
                    result = tool._run(
                        symbol=symbol,
                        quantity=quantity,
                        purchase_price=current_price_float,
                        scenario=scenario,
                        time_horizon=time_horizon
                    )

                if "Error" in result:
                    st.error(result)
                else:
                    st.success("Analysis complete!")

                    # Show key results in a clean format
                    st.markdown("#### Results")
                    st.markdown(result)

            except Exception as e:
                st.error(f"Analysis failed: {e}")

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
            "Market data updates are now handled in the main portfolio view. Use the '📈 Update Market Data' button there to refresh prices."
        )

        # Get current portfolio
        portfolio_manager = self.initialize_components()[0]
        if (
            portfolio_manager
            and hasattr(portfolio_manager, "current_portfolio")
            and portfolio_manager.current_portfolio
        ):
            portfolio_name = portfolio_manager.current_portfolio.name

            st.write(f"**Current Portfolio:** {portfolio_name}")

            # Show market data info
            market_data = portfolio_manager.market_data_store
            symbols = market_data.get_symbols()
            if symbols:
                st.write(f"**Market Data:** {len(symbols)} symbols tracked")
                # Show current portfolio value
                current_value = portfolio_manager.get_portfolio_value()
                if current_value:
                    st.write(f"**Current Value:** ${float(current_value):,.2f}")
            else:
                st.warning("No market data found. Use 'Update Market Data' to fetch prices.")

            # Keep a simple button for emergency data refresh but discourage its use
            if st.button("🔄 Emergency Data Refresh", help="Only use if the main update button isn't working"):
                with st.spinner("Updating market data..."):
                    try:
                        end_date = date.today()
                        start_date = end_date - timedelta(days=30)
                        portfolio_manager.update_market_data(start_date, end_date)
                        st.success("✅ Updated market data with fresh prices")

                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        logging.error(f"Emergency refresh failed: {error_details}")
                        st.error(f"❌ Error updating market data: {e}")
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
