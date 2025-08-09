"""
Streamlit UI for the Portfolio Tracker with AI Agent.
"""

import os
import sys
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, Optional, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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

        # Portfolio selection
        portfolios = portfolio_manager.list_portfolios()

        if portfolios:
            selected = st.sidebar.selectbox(
                "Select Portfolio:",
                ["None"] + portfolios,
                index=(
                    0
                    if not st.session_state.selected_portfolio
                    else portfolios.index(st.session_state.selected_portfolio) + 1
                ),
            )

            if selected != "None" and selected != st.session_state.selected_portfolio:
                portfolio = portfolio_manager.load_portfolio(selected)
                if portfolio:
                    st.session_state.portfolio_loaded = True
                    st.session_state.selected_portfolio = selected
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
            if st.sidebar.button("🔄 Update Portfolio (Prices + Snapshots)", type="primary"):
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

        # Model selection + Quick action buttons
        st.subheader("Quick Actions")
        col0, col1, col2, col3 = st.columns([2, 1, 1, 1])

        with col0:
            st.markdown("**AI Model**")
            provider = st.selectbox("Provider", ["Azure OpenAI", "Anthropic", "Google Vertex AI"], index=0)
            if provider == "Azure OpenAI":
                model_choice = st.selectbox(
                    "Model",
                    [
                        ("Azure GPT-4.1", ("https://kallamai.openai.azure.com/", "gpt-4.1")),
                        ("Azure GPT-4.1 Mini", ("https://kallamai.openai.azure.com/", "gpt-4.1-mini")),
                        ("Azure o4-mini", ("https://kallamai.openai.azure.com/", "o4-mini")),
                        ("Azure GPT-5 Mini", ("https://kallamai.openai.azure.com/", "gpt-5-mini")),
                    ],
                    format_func=lambda x: x[0],
                )
                if st.button("Apply Model"):
                    try:
                        endpoint, model = model_choice[1]
                        azure_key = os.getenv("AZURE_OPENAI_API_KEY", "")
                        agent.set_llm_config(
                            provider="azure-openai",
                            azure_endpoint=endpoint,
                            azure_api_key=azure_key,
                            azure_model=model,
                        )
                        st.success(f"Azure model set to {model}")
                    except Exception as e:
                        st.error(f"Failed to set Azure model: {e}")
            elif provider == "Anthropic":
                model_choice = st.selectbox(
                    "Model",
                    [
                        ("Claude Sonnet 4 (thinking)", "claude-sonnet-4-20250514"),
                    ],
                    format_func=lambda x: x[0],
                )
                if st.button("Apply Model"):
                    try:
                        model = model_choice[1]
                        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
                        agent.set_llm_config(
                            provider="anthropic",
                            anthropic_api_key=anthropic_key,
                            anthropic_model=model,
                        )
                        st.success(f"Anthropic model set to {model}")
                    except Exception as e:
                        st.error(f"Failed to set Anthropic model: {e}")
            else:
                # Google Vertex AI
                model_choice = st.selectbox(
                    "Model",
                    [
                        ("Gemini 2.0 Flash Lite", "gemini-2.0-flash-lite-001"),
                        ("Gemini 2.5 Pro (thinking)", "gemini-2.5-pro"),
                    ],
                    format_func=lambda x: x[0],
                )
                if st.button("Apply Model"):
                    try:
                        model = model_choice[1]
                        project = os.getenv("GOOGLE_VERTEX_PROJECT", "mystic-fountain-415918")
                        location = os.getenv("GOOGLE_VERTEX_LOCATION", "us-central1")
                        # Credentials supplied via GOOGLE_APPLICATION_CREDENTIALS
                        agent.set_llm_config(
                            provider="vertex-ai",
                            vertex_project=project,
                            vertex_location=location,
                            vertex_model=model,
                        )
                        st.success(f"Vertex AI model set to {model}")
                    except Exception as e:
                        st.error(f"Failed to set Vertex AI model: {e}")

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

        # Overview uses only locally stored data (no network fetches)
        fetch_live = False

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
            cash_total = sum(portfolio.cash_balances.values())
            st.metric("Cash", f"${cash_total:,.2f}")

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
                        twr *= (1.0 + r)
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
                instrument = portfolio.positions.get(symbol).instrument if symbol in portfolio.positions else None
                isin = instrument.isin if instrument else None
                currency_code = pos.get("currency")

                # Market value in base currency
                mv = pos.get("market_value") or Decimal("0")
                mv_base = self._convert_to_base(
                    portfolio_manager, mv, currency_code, base_currency.value, allow_fetch=False
                )

                # Convert unrealized PnL to base currency
                unreal_val_native = pos.get("unrealized_pnl") or Decimal("0")
                unreal_val_base = self._convert_to_base(
                    portfolio_manager, unreal_val_native, currency_code, base_currency.value
                )

                # Category classification
                category = self._classify_position(pos, instrument)

                # YTD metrics are not computed in overview to avoid network calls

                # Total buy price (cost basis) in native and base
                total_buy_native = pos.get("cost_basis")
                total_buy_base = self._convert_to_base(
                    portfolio_manager,
                    total_buy_native if total_buy_native is not None else Decimal("0"),
                    currency_code,
                    base_currency.value,
                    allow_fetch=False,
                ) if total_buy_native is not None else None

                # Latest purchase date for this position
                latest_buy_date = None
                try:
                    buys = [
                        t.timestamp.date()
                        for t in portfolio.transactions
                        if t.instrument.symbol == symbol and t.transaction_type == TransactionType.BUY
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
                        "total_buy_price": total_buy_native,
                        "total_buy_price_base": total_buy_base,
                        "latest_buy_date": latest_buy_date,
                    }
                )

            # No YTD aggregate metrics in overview to avoid external data fetches

            # Render by category using cards
            categories = ["Short Term", "Bonds", "Equities", "Alternatives", "Miscellaneous"]
            for cat in categories:
                group = [e for e in enriched if e.get("category") == cat]
                # Include cash under Short Term as its own items
                cash_items = []
                if cat == "Short Term" and portfolio.cash_balances:
                    for curr, amt in portfolio.cash_balances.items():
                        curr_code = getattr(curr, "value", str(curr))
                        amt_base = self._convert_to_base(
                            portfolio_manager, Decimal(str(amt)), curr_code, base_currency.value
                        )
                        cash_items.append({
                            "name": f"Cash ({curr_code})",
                            "isin": "-",
                            "currency": curr_code,
                            "quantity": None,
                            "current_price": None,
                            "market_value_base": amt_base,
                            "unrealized_pnl_base": None,
                            "unrealized_pnl_percent": None,
                            "ytd_unrealized_pnl": None,
                            "ytd_unrealized_pnl_percent": None,
                        })

                if not group and not cash_items:
                    continue

                st.markdown(f"### {cat}")
                items = group + cash_items
                # Display in rows of 3 cards
                for i in range(0, len(items), 3):
                    cols = st.columns(3)
                    for col, item in zip(cols, items[i:i+3]):
                        with col:
                            self._render_position_card(item, base_currency.value)

            # Allocation charts
            st.subheader("📊 Allocation")
            col_a, col_b = st.columns(2)
            with col_a:
                self.plot_allocation_by_category(enriched, base_currency.value, portfolio_manager, allow_fetch=fetch_live)
            with col_b:
                self.plot_allocation_by_currency(enriched, base_currency.value, portfolio_manager, allow_fetch=fetch_live)

        # Add transaction form
        st.subheader("➕ Add Transaction")
        self.render_transaction_form(portfolio_manager)

        # Recent transactions
        st.subheader("📝 Recent Transactions")
        transactions = portfolio_manager.get_transaction_history(30)

        if transactions:
            df_transactions = pd.DataFrame(transactions)
            df_transactions["timestamp"] = pd.to_datetime(
                df_transactions["timestamp"]
            ).dt.strftime("%Y-%m-%d %H:%M")

            # Format monetary columns
            for col in ["quantity", "price", "fees", "total_value"]:
                if col in df_transactions.columns:
                    df_transactions[col] = df_transactions[col].apply(
                        lambda x: f"{float(x):,.2f}"
                    )

            st.dataframe(df_transactions, use_container_width=True)
        else:
            st.info("No recent transactions found.")

    def plot_allocation_by_category(self, positions, base_currency_code: str, portfolio_manager, allow_fetch: bool = False):
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
                    portfolio_manager, Decimal(str(amt)), curr_code, base_currency_code, allow_fetch=allow_fetch
                )
        if cash_total_base > 0:
            totals["Short Term"] = totals.get("Short Term", 0.0) + float(cash_total_base)

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

    def plot_allocation_by_currency(self, positions, base_currency_code: str, portfolio_manager, allow_fetch: bool = False):
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
                        portfolio_manager, Decimal(str(amt)), curr_code, base_currency_code, allow_fetch=allow_fetch
                    )
                    totals[curr_code] = totals.get(curr_code, 0.0) + float(amt_base)
                else:
                    if curr_code == base_currency_code:
                        totals[curr_code] = totals.get(curr_code, 0.0) + float(Decimal(str(amt)))

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
        pnl_base = item.get("unrealized_pnl_base")
        pnl_pct = item.get("unrealized_pnl_percent")
        ytd_base = item.get("ytd_unrealized_pnl")
        ytd_pct = item.get("ytd_unrealized_pnl_percent")
        ytd_mkt_base = item.get("ytd_market_pnl")
        ytd_mkt_pct = item.get("ytd_market_pnl_percent")
        ytd_fx_base = item.get("ytd_fx_pnl")
        ytd_fx_pct = item.get("ytd_fx_pnl_percent")
        total_buy = item.get("total_buy_price")
        total_buy_base = item.get("total_buy_price_base")

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
            f"<div><span style='color:#666;'>Market Value:</span> <strong>{fmt_money(mv_base)} {base_currency_code}</strong></div>",
            f"<div><span style='color:#666;'>Unrealized PnL:</span> {colored(pnl_base)} {base_currency_code} ({fmt_signed(pnl_pct)}%)</div>",
            f"<div><span style='color:#666;'>YTD PnL:</span> {colored(ytd_base)} {base_currency_code} ({colored(ytd_pct)}%)</div>",
            f"<div style='font-size:13px; margin-top:4px;'><em>YTD Market:</em> {colored(ytd_mkt_base)} {base_currency_code} ({colored(ytd_mkt_pct)}%)</div>",
            f"<div style='font-size:13px;'><em>YTD FX:</em> {colored(ytd_fx_base)} {base_currency_code} ({colored(ytd_fx_pct)}%)</div>",
        ]
        if total_buy is not None:
            lines.append(
                f"<div style='margin-top:4px;'><span style='color:#666;'>Total Buy:</span> {fmt_money(total_buy_base)} {base_currency_code}"
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

    def _convert_to_base(self, portfolio_manager, amount: Decimal, from_currency_code: str, base_currency_code: str, allow_fetch: bool = False) -> Decimal:
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
            rate = portfolio_manager.data_manager.get_exchange_rate(
                Currency(from_currency_code), Currency(base_currency_code)
            )
            if rate:
                return Decimal(str(amount)) * rate
        except Exception:
            pass
        return Decimal(str(amount))

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
        if any(hint in name for hint in ["gold", "bullion"]) or symbol in {"GLD", "IAU", "PHYS"}:
            return "Alternatives"

        # Equities default for stocks and ETFs
        if itype in {"stock", "etf"}:
            return "Equities"

        return "Miscellaneous"

    def _get_reference_price_for_date(self, portfolio_manager, symbol: str, ref_date: date) -> Optional[Decimal]:
        """Get historical close price for the given date, with short lookahead if missing."""
        try:
            # Exact date
            prices = portfolio_manager.data_manager.get_historical_prices(symbol, ref_date, ref_date)
            if prices:
                pd0 = prices[0]
                return (
                    pd0.close_price or pd0.open_price or pd0.high_price or pd0.low_price
                )
            # Look ahead a few business days (first available of the year)
            end = ref_date + timedelta(days=5)
            prices = portfolio_manager.data_manager.get_historical_prices(symbol, ref_date, end)
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
                isin = st.text_input("ISIN (optional)", placeholder="e.g., US0378331005")
                transaction_type = st.selectbox(
                    "Type", ["buy", "sell", "dividend", "deposit", "withdrawal"]
                )

            with col2:
                quantity = st.number_input("Quantity/Amount", min_value=0.0, step=0.01)
                price = st.number_input("Price", min_value=0.0, step=0.01)

            with col3:
                fees = st.number_input("Fees", min_value=0.0, step=0.01, value=0.0)
                trade_date = st.date_input("Date", value=date.today())

            notes = st.text_area("Notes (optional)")

            if st.form_submit_button("Add Transaction"):
                if (symbol or isin) and quantity > 0 and price > 0:
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
                            fees=Decimal(str(fees)),
                            notes=notes if notes else None,
                            isin=(isin.upper() if isin else None),
                        )

                        if success:
                            label = (symbol.upper() if symbol else isin.upper())
                            st.success(
                                f"Added {transaction_type} transaction: {quantity} {label} @ ${price}"
                            )
                            st.rerun()
                        else:
                            st.error("Failed to add transaction")

                    except Exception as e:
                        st.error(f"Error adding transaction: {e}")
                else:
                    st.error("Please provide a Symbol or ISIN, and ensure quantity/price are greater than 0")

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
            st.warning("Insufficient data for selected period. Please add more historical data or create snapshots.")
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

        # Calculate metrics (time-weighted returns using external flows)
        with st.spinner("Calculating metrics..."):
            flows = portfolio_manager.get_external_cash_flows_by_day(start_date, end_date)
            flows_float = {d: float(v) for d, v in flows.items()}

            # Returns with flows
            portfolio_returns = metrics_calculator.calculate_returns(snapshots, flows_float)
            if len(portfolio_returns) == 0:
                st.error("Could not calculate returns")
                return

            # Base metrics dict
            metrics = {}

            # TWR total and annualized
            import numpy as np
            twr_product = float(np.prod([1.0 + r for r in portfolio_returns]))
            total_return_twr = twr_product - 1.0
            n = len(portfolio_returns)
            annualized_return_twr = (twr_product ** (252.0 / n)) - 1.0 if n > 0 else 0.0

            metrics["total_return"] = total_return_twr
            metrics["annualized_return"] = annualized_return_twr
            metrics["volatility"] = metrics_calculator.calculate_volatility(portfolio_returns)
            metrics["sharpe_ratio"] = metrics_calculator.calculate_sharpe_ratio(portfolio_returns)
            metrics["sortino_ratio"] = metrics_calculator.calculate_sortino_ratio(portfolio_returns)
            # Risk metrics
            md, md_dur = metrics_calculator.calculate_max_drawdown(snapshots)
            metrics["max_drawdown"] = md
            metrics["max_drawdown_duration"] = md_dur
            metrics["var_5pct"] = metrics_calculator.calculate_value_at_risk(portfolio_returns, 0.05)
            metrics["cvar_5pct"] = metrics_calculator.calculate_conditional_var(portfolio_returns, 0.05)
            metrics["calmar_ratio"] = metrics_calculator.calculate_calmar_ratio(portfolio_returns, snapshots)

            # Benchmark-relative metrics computed against the same date range
            bench_returns = metrics_calculator.get_benchmark_returns(benchmark, start_date, end_date)
            min_len = min(len(portfolio_returns), len(bench_returns))
            if min_len > 1:
                pr = portfolio_returns[-min_len:]
                br = bench_returns[-min_len:]
                metrics["beta"] = metrics_calculator.calculate_beta(pr, br)
                metrics["alpha"] = metrics_calculator.calculate_alpha(pr, br)
                metrics["information_ratio"] = metrics_calculator.calculate_information_ratio(pr, br)
                metrics["benchmark_return"] = float(np.mean(br) * 252)
                metrics["benchmark_volatility"] = metrics_calculator.calculate_volatility(br)
                metrics["benchmark_available"] = True
            else:
                metrics["benchmark_available"] = False

        if "error" in metrics:
            st.error(metrics["error"])
            return

        # Display key metrics
        st.subheader("📊 Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Return", f"{metrics.get('total_return', 0)*100:.2f}%")
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")

        with col2:
            st.metric(
                "Annualized Return", f"{metrics.get('annualized_return', 0)*100:.2f}%"
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

        # YTD TWR and Unrealized PnL (Value) in selected currency
        try:
            ytd_start = date(date.today().year, 1, 1)
            if end_date < ytd_start:
                ytd_start = end_date  # guard
            ytd_snaps = portfolio_manager.storage.load_snapshots(
                portfolio_manager.current_portfolio.id, ytd_start, end_date
            )

            col_ytd1, col_ytd2 = st.columns(2)

            # YTD Performance (TWR)
            if len(ytd_snaps) >= 2:
                ytd_flows = portfolio_manager.get_external_cash_flows_by_day(ytd_start, end_date)
                ytd_flows_f = {d: float(v) for d, v in ytd_flows.items()}
                ytd_returns = metrics_calculator.calculate_returns(ytd_snaps, ytd_flows_f)
                if ytd_returns:
                    import numpy as np
                    prod = float(np.prod([1.0 + r for r in ytd_returns]))
                    ytd_twr_pct = (prod - 1.0) * 100.0
                    col_ytd1.metric("YTD Performance (TWR)", f"{ytd_twr_pct:.2f}%")
                else:
                    col_ytd1.metric("YTD Performance (TWR)", "N/A")
            else:
                col_ytd1.metric("YTD Performance (TWR)", "N/A")

            # Unrealized PnL (Value) as of end_date in selected currency
            unreal_sum = 0.0
            if ytd_snaps:
                last_snap = ytd_snaps[-1]
                disp_code = display_currency_code
                from src.portfolio.models import Currency as Cur
                for pos in last_snap.positions.values():
                    if pos.current_price is None or pos.average_cost is None or pos.quantity is None:
                        continue
                    try:
                        pnl_native = (pos.current_price - pos.average_cost) * pos.quantity
                        from_code = pos.instrument.currency.value if hasattr(pos.instrument.currency, "value") else str(pos.instrument.currency)
                        if from_code == disp_code:
                            unreal_sum += float(pnl_native)
                        else:
                            rate = portfolio_manager.data_manager.get_historical_fx_rate_on(last_snap.date, Cur(from_code), Cur(disp_code))
                            if rate:
                                unreal_sum += float(pnl_native * rate)
                            else:
                                unreal_sum += float(pnl_native)
                    except Exception:
                        continue
            col_ytd2.metric(f"Unrealized PnL ({display_currency_code})", f"{unreal_sum:,.2f} {display_currency_code}")
        except Exception:
            pass

        # Portfolio value chart with category overlays in selected currency
        self.plot_portfolio_and_categories(snapshots, display_currency_code, portfolio_manager, start_date, end_date)

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

    def plot_portfolio_and_categories(self, snapshots, display_currency_code: str, portfolio_manager, start_date: date, end_date: date):
        """Plot portfolio and category series in selected currency, plus cumulative returns."""
        if len(snapshots) < 2:
            return

        # FX rate cache
        fx_cache: Dict[tuple, Optional[Decimal]] = {}

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
            return rate

        # Category classifier (reuse UI logic)
        def classify(instrument) -> str:
            itype = instrument.instrument_type.value if hasattr(instrument.instrument_type, "value") else str(instrument.instrument_type)
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
            if any(h in name for h in ["gold", "bullion"]) or symbol in {"GLD", "IAU", "PHYS"}:
                return "Alternatives"
            if itype in {"stock", "etf"}:
                return "Equities"
            return "Miscellaneous"

        dates: List[date] = [s.date for s in snapshots]
        # Portfolio line in display currency
        portfolio_values: List[float] = []
        base_code = snapshots[0].base_currency.value if hasattr(snapshots[0].base_currency, "value") else str(snapshots[0].base_currency)
        for s in snapshots:
            val = Decimal(str(s.total_value))
            rate = get_rate(s.date, base_code, display_currency_code)
            if rate is not None:
                val = val * rate
            portfolio_values.append(float(val))

        # Category lines in display currency
        categories = ["Short Term", "Bonds", "Equities", "Alternatives", "Miscellaneous"]
        cat_series: Dict[str, List[float]] = {c: [] for c in categories}

        for s in snapshots:
            # init per-snapshot sums
            sums = {c: Decimal("0") for c in categories}
            for pos in s.positions.values():
                if pos.market_value is None:
                    continue
                instr = pos.instrument
                cat = classify(instr)
                from_code = instr.currency.value if hasattr(instr.currency, "value") else str(instr.currency)
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
            go.Scatter(x=dates, y=portfolio_values, mode="lines", name="Portfolio", line=dict(width=2))
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
                go.Scatter(x=dates, y=cat_series[c], mode="lines", name=c, line=dict(color=color_map.get(c)))
            )

        fig.update_layout(
            title=f"Performance Over Time (in {display_currency_code})",
            xaxis_title="Date",
            yaxis_title=f"Value ({display_currency_code})",
            height=420,
            showlegend=True,
        )
        fig.update_xaxes(range=[start_date, end_date])
        st.plotly_chart(fig, use_container_width=True)

        # Cumulative returns for portfolio and categories
        def to_cum_returns(values: List[float]) -> List[float]:
            if not values or values[0] == 0:
                return [0.0 for _ in values]
            base = values[0]
            return [((v / base) - 1) * 100.0 for v in values]

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(x=dates, y=to_cum_returns(portfolio_values), mode="lines", name="Portfolio", line=dict(width=2, color="#000"))
        )
        for c in categories:
            fig2.add_trace(
                go.Scatter(x=dates, y=to_cum_returns(cat_series[c]), mode="lines", name=c, line=dict(color=color_map.get(c)))
            )
        fig2.update_layout(
            title=f"Cumulative Returns (in {display_currency_code})",
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

        st.subheader("🔑 API Keys")
        st.info(
            "Configure your API keys for data providers. Leave as placeholder for demo mode."
        )

        openai_key = st.text_input(
            "OpenAI API Key", type="password", value="OPENAI_API_KEY_PLACEHOLDER"
        )
        alpha_vantage_key = st.text_input(
            "Alpha Vantage API Key",
            type="password",
            value="ALPHA_VANTAGE_API_KEY_PLACEHOLDER",
        )

        if st.button("Update API Keys"):
            st.success("API keys updated (demo mode)")

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
