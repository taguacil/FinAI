"""
Streamlit UI for the Portfolio Tracker with AI Agent.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, date, timedelta
from decimal import Decimal
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.portfolio.manager import PortfolioManager
from src.portfolio.storage import FileBasedStorage
from src.data_providers.manager import DataProviderManager
from src.utils.metrics import FinancialMetricsCalculator
from src.agents.portfolio_agent import PortfolioAgent
from src.portfolio.models import Currency


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
            initial_sidebar_state="expanded"
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
                metrics_calculator=metrics_calculator
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
        st.markdown("*Your intelligent financial companion for portfolio management and investment advice*")
        
        # Sidebar
        self.render_sidebar(portfolio_manager)
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["💬 AI Chat", "📊 Portfolio", "📈 Analytics", "⚙️ Settings"])
        
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
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'portfolio_loaded' not in st.session_state:
            st.session_state.portfolio_loaded = False
        if 'selected_portfolio' not in st.session_state:
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
                index=0 if not st.session_state.selected_portfolio else portfolios.index(st.session_state.selected_portfolio) + 1
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
                        portfolio = portfolio_manager.create_portfolio(new_name, Currency(base_currency))
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
            st.sidebar.info(f"""
            **Name:** {portfolio.name}
            **Base Currency:** {portfolio.base_currency.value}
            **Created:** {portfolio.created_at.strftime('%Y-%m-%d')}
            **Positions:** {len(portfolio.positions)}
            """)
            
            # Quick actions
            if st.sidebar.button("🔄 Update Prices"):
                with st.spinner("Updating prices..."):
                    results = portfolio_manager.update_current_prices()
                    success_count = sum(results.values())
                    st.sidebar.success(f"Updated {success_count}/{len(results)} prices")
    
    def render_chat_interface(self, agent):
        """Render the AI chat interface."""
        st.header("💬 Chat with AI Financial Advisor")
        
        # Chat history
        chat_container = st.container()
        
        with chat_container:
            if not st.session_state.chat_history:
                # Initialize conversation
                initial_message = agent.initialize_conversation()
                st.session_state.chat_history.append({"role": "assistant", "content": initial_message})
            
            # Display chat history
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])
        
        # Chat input
        user_input = st.chat_input("Ask me about your portfolio, investments, or market conditions...")
        
        if user_input:
            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Get AI response
            with st.spinner("Thinking..."):
                try:
                    response = agent.chat(user_input)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            
            st.rerun()
        
        # Quick action buttons
        st.subheader("Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Portfolio Summary"):
                if st.session_state.portfolio_loaded:
                    response = agent.chat("Please show me my current portfolio summary")
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()
                else:
                    st.warning("Please load a portfolio first")
        
        with col2:
            if st.button("📈 Performance Analysis"):
                if st.session_state.portfolio_loaded:
                    response = agent.analyze_portfolio_performance()
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
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
        
        # Update prices
        with st.spinner("Updating current prices..."):
            portfolio_manager.update_current_prices()
        
        total_value = portfolio_manager.get_portfolio_value()
        positions = portfolio_manager.get_position_summary()
        
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
            total_pnl = sum(float(pos.get('unrealized_pnl', 0) or 0) for pos in positions)
            st.metric("Unrealized P&L", f"${total_pnl:,.2f}", delta=f"{total_pnl:,.2f}")
        
        # Positions table
        if positions:
            st.subheader("📈 Current Positions")
            
            # Convert to DataFrame
            df_positions = pd.DataFrame(positions)
            
            # Format columns
            df_display = df_positions.copy()
            for col in ['quantity', 'average_cost', 'current_price', 'market_value', 'cost_basis']:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(lambda x: f"{float(x):,.2f}" if x is not None else "N/A")
            
            for col in ['unrealized_pnl', 'unrealized_pnl_percent']:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(
                        lambda x: f"{float(x):+.2f}" if x is not None else "N/A"
                    )
            
            st.dataframe(df_display, use_container_width=True)
            
            # Portfolio allocation chart
            self.plot_portfolio_allocation(positions)
        
        # Add transaction form
        st.subheader("➕ Add Transaction")
        self.render_transaction_form(portfolio_manager)
        
        # Recent transactions
        st.subheader("📝 Recent Transactions")
        transactions = portfolio_manager.get_transaction_history(30)
        
        if transactions:
            df_transactions = pd.DataFrame(transactions)
            df_transactions['timestamp'] = pd.to_datetime(df_transactions['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Format monetary columns
            for col in ['quantity', 'price', 'fees', 'total_value']:
                if col in df_transactions.columns:
                    df_transactions[col] = df_transactions[col].apply(lambda x: f"{float(x):,.2f}")
            
            st.dataframe(df_transactions, use_container_width=True)
        else:
            st.info("No recent transactions found.")
    
    def plot_portfolio_allocation(self, positions):
        """Plot portfolio allocation pie chart."""
        if not positions:
            return
        
        # Prepare data
        symbols = []
        values = []
        
        for pos in positions:
            if pos.get('market_value') and float(pos['market_value']) > 0:
                symbols.append(pos['symbol'])
                values.append(float(pos['market_value']))
        
        if not symbols:
            return
        
        # Create pie chart
        fig = px.pie(
            values=values,
            names=symbols,
            title="Portfolio Allocation by Position"
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_transaction_form(self, portfolio_manager):
        """Render transaction input form."""
        with st.form("add_transaction"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                symbol = st.text_input("Symbol", placeholder="e.g., AAPL")
                transaction_type = st.selectbox("Type", ["buy", "sell", "dividend", "deposit", "withdrawal"])
            
            with col2:
                quantity = st.number_input("Quantity/Amount", min_value=0.0, step=0.01)
                price = st.number_input("Price", min_value=0.0, step=0.01)
            
            with col3:
                fees = st.number_input("Fees", min_value=0.0, step=0.01, value=0.0)
                days_ago = st.number_input("Days Ago", min_value=0, max_value=365, value=0)
            
            notes = st.text_area("Notes (optional)")
            
            if st.form_submit_button("Add Transaction"):
                if symbol and quantity > 0 and price > 0:
                    try:
                        timestamp = datetime.now() - timedelta(days=days_ago)
                        
                        # Map transaction types
                        from src.portfolio.models import TransactionType
                        txn_type_map = {
                            'buy': TransactionType.BUY,
                            'sell': TransactionType.SELL,
                            'dividend': TransactionType.DIVIDEND,
                            'deposit': TransactionType.DEPOSIT,
                            'withdrawal': TransactionType.WITHDRAWAL
                        }
                        
                        success = portfolio_manager.add_transaction(
                            symbol=symbol.upper(),
                            transaction_type=txn_type_map[transaction_type],
                            quantity=Decimal(str(quantity)),
                            price=Decimal(str(price)),
                            timestamp=timestamp,
                            fees=Decimal(str(fees)),
                            notes=notes if notes else None
                        )
                        
                        if success:
                            st.success(f"Added {transaction_type} transaction: {quantity} {symbol} @ ${price}")
                            st.rerun()
                        else:
                            st.error("Failed to add transaction")
                    
                    except Exception as e:
                        st.error(f"Error adding transaction: {e}")
                else:
                    st.error("Please fill in all required fields")
    
    def render_analytics(self, portfolio_manager, metrics_calculator):
        """Render portfolio analytics and charts."""
        st.header("📈 Portfolio Analytics")
        
        if not st.session_state.portfolio_loaded:
            st.warning("Please load a portfolio to view analytics.")
            return
        
        # Time period selection
        col1, col2 = st.columns(2)
        with col1:
            days = st.selectbox("Analysis Period", [30, 90, 180, 365], index=3)
        with col2:
            benchmark = st.text_input("Benchmark Symbol", value="SPY")
        
        # Get portfolio snapshots
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        snapshots = portfolio_manager.storage.load_snapshots(
            portfolio_manager.current_portfolio.id, start_date, end_date
        )
        
        if len(snapshots) < 2:
            st.warning(f"Insufficient data for {days}-day analysis. Please add more historical data.")
            return
        
        # Calculate metrics
        with st.spinner("Calculating metrics..."):
            metrics = metrics_calculator.calculate_portfolio_metrics(snapshots, benchmark)
        
        if 'error' in metrics:
            st.error(metrics['error'])
            return
        
        # Display key metrics
        st.subheader("📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{metrics.get('total_return', 0)*100:.2f}%")
            st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.3f}")
        
        with col2:
            st.metric("Annualized Return", f"{metrics.get('annualized_return', 0)*100:.2f}%")
            st.metric("Volatility", f"{metrics.get('volatility', 0)*100:.2f}%")
        
        with col3:
            st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.2f}%")
            st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.3f}")
        
        with col4:
            if metrics.get('benchmark_available'):
                st.metric("Beta", f"{metrics.get('beta', 0):.3f}")
                st.metric("Alpha", f"{metrics.get('alpha', 0)*100:.2f}%")
            else:
                st.info("Benchmark data not available")
        
        # Portfolio value chart
        self.plot_portfolio_performance(snapshots)
        
        # Risk metrics
        st.subheader("⚠️ Risk Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Value at Risk (5%)", f"{metrics.get('var_5pct', 0)*100:.2f}%")
            st.metric("Conditional VaR (5%)", f"{metrics.get('cvar_5pct', 0)*100:.2f}%")
        
        with col2:
            st.metric("Calmar Ratio", f"{metrics.get('calmar_ratio', 0):.3f}")
            if metrics.get('benchmark_available'):
                st.metric("Information Ratio", f"{metrics.get('information_ratio', 0):.3f}")
    
    def plot_portfolio_performance(self, snapshots):
        """Plot portfolio performance over time."""
        if len(snapshots) < 2:
            return
        
        # Prepare data
        dates = [snapshot.date for snapshot in snapshots]
        values = [float(snapshot.total_value) for snapshot in snapshots]
        
        # Calculate cumulative returns
        initial_value = values[0]
        cumulative_returns = [(value / initial_value - 1) * 100 for value in values]
        
        # Create plot
        fig = go.Figure()
        
        # Portfolio value
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Performance Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Cumulative returns chart
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=dates,
            y=cumulative_returns,
            mode='lines',
            name='Cumulative Return (%)',
            line=dict(color='green', width=2)
        ))
        
        fig2.update_layout(
            title="Cumulative Returns",
            xaxis_title="Date",
            yaxis_title="Return (%)",
            height=300
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    def render_settings(self):
        """Render settings and configuration."""
        st.header("⚙️ Settings")
        
        st.subheader("🔑 API Keys")
        st.info("Configure your API keys for data providers. Leave as placeholder for demo mode.")
        
        openai_key = st.text_input("OpenAI API Key", type="password", value="OPENAI_API_KEY_PLACEHOLDER")
        alpha_vantage_key = st.text_input("Alpha Vantage API Key", type="password", value="ALPHA_VANTAGE_API_KEY_PLACEHOLDER")
        
        if st.button("Update API Keys"):
            st.success("API keys updated (demo mode)")
        
        st.subheader("📊 Data Providers")
        st.write("Current data providers:")
        st.write("- Yahoo Finance (Free)")
        st.write("- Alpha Vantage (API Key required)")
        
        st.subheader("💾 Data Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Cache"):
                st.success("Cache cleared")
        
        with col2:
            if st.button("Export Data"):
                st.info("Export functionality would be implemented here")
        
        st.subheader("ℹ️ About")
        st.markdown("""
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
        """)


def main():
    """Main function to run the Streamlit app."""
    app = PortfolioTrackerUI()
    app.run()


if __name__ == "__main__":
    main()