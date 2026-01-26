"""
Market Data Store Viewer - Debug and verification page for market prices.
"""

import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st


def render_market_data_page(data_dir: str):
    """Render the market data store viewer page."""
    st.header("🗄️ Market Data Store")
    st.caption("View and manage centralized market price data used across all portfolios.")

    db_path = Path(data_dir) / "market_data.db"

    if not db_path.exists():
        st.warning("Market data database not found.")
        return

    # Load data
    conn = sqlite3.connect(str(db_path))

    # Get summary stats
    stats_df = pd.read_sql_query("""
        SELECT
            COUNT(*) as total_prices,
            COUNT(DISTINCT symbol) as unique_symbols,
            COUNT(DISTINCT date) as unique_dates,
            MIN(date) as earliest_date,
            MAX(date) as latest_date
        FROM market_prices
    """, conn)

    # Display summary
    st.subheader("📊 Summary")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Prices", f"{stats_df['total_prices'].iloc[0]:,}")
    with col2:
        st.metric("Unique Symbols", stats_df['unique_symbols'].iloc[0])
    with col3:
        st.metric("Unique Dates", stats_df['unique_dates'].iloc[0])
    with col4:
        st.metric("Earliest Date", stats_df['earliest_date'].iloc[0] or "N/A")
    with col5:
        st.metric("Latest Date", stats_df['latest_date'].iloc[0] or "N/A")

    st.divider()

    # Filters
    st.subheader("🔍 Filter & View")

    col1, col2, col3 = st.columns([2, 2, 1])

    # Get available symbols for dropdown
    symbols_df = pd.read_sql_query(
        "SELECT DISTINCT symbol FROM market_prices ORDER BY symbol", conn
    )
    available_symbols = ["All"] + symbols_df['symbol'].tolist()

    # Get available dates for dropdown
    dates_df = pd.read_sql_query(
        "SELECT DISTINCT date FROM market_prices ORDER BY date DESC", conn
    )
    available_dates = ["All"] + dates_df['date'].tolist()

    with col1:
        selected_symbol = st.selectbox("Symbol", available_symbols, key="mds_symbol")

    with col2:
        selected_date = st.selectbox("Date", available_dates, key="mds_date")

    with col3:
        limit = st.number_input("Limit", min_value=10, max_value=1000, value=100, step=50)

    # Build query
    query = "SELECT symbol, date, price, currency, source, created_at FROM market_prices WHERE 1=1"
    params = []

    if selected_symbol != "All":
        query += " AND symbol = ?"
        params.append(selected_symbol)

    if selected_date != "All":
        query += " AND date = ?"
        params.append(selected_date)

    query += " ORDER BY date DESC, symbol ASC LIMIT ?"
    params.append(limit)

    # Execute query
    prices_df = pd.read_sql_query(query, conn, params=params)

    if not prices_df.empty:
        # Format for display
        prices_df['price'] = prices_df['price'].astype(float)

        st.dataframe(
            prices_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "symbol": st.column_config.TextColumn("Symbol", width="medium"),
                "date": st.column_config.TextColumn("Date", width="small"),
                "price": st.column_config.NumberColumn("Price", format="%.4f"),
                "currency": st.column_config.TextColumn("Currency", width="small"),
                "source": st.column_config.TextColumn("Source", width="medium"),
                "created_at": st.column_config.TextColumn("Created", width="medium"),
            }
        )
        st.caption(f"Showing {len(prices_df)} of {stats_df['total_prices'].iloc[0]} total prices")
    else:
        st.info("No prices match the selected filters.")

    st.divider()

    # Prices by date chart
    st.subheader("📈 Price Coverage by Date")

    coverage_df = pd.read_sql_query("""
        SELECT date, COUNT(*) as count
        FROM market_prices
        GROUP BY date
        ORDER BY date
    """, conn)

    if not coverage_df.empty:
        coverage_df['date'] = pd.to_datetime(coverage_df['date'])
        fig = px.bar(
            coverage_df,
            x='date',
            y='count',
            labels={'date': 'Date', 'count': 'Number of Prices'},
            title='Prices Available per Date'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Symbol coverage
    st.subheader("📋 Symbol Coverage")

    symbol_coverage_df = pd.read_sql_query("""
        SELECT
            symbol,
            COUNT(*) as price_count,
            MIN(date) as first_date,
            MAX(date) as last_date,
            currency
        FROM market_prices
        GROUP BY symbol
        ORDER BY symbol
    """, conn)

    if not symbol_coverage_df.empty:
        st.dataframe(
            symbol_coverage_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "symbol": st.column_config.TextColumn("Symbol", width="medium"),
                "price_count": st.column_config.NumberColumn("# Prices", width="small"),
                "first_date": st.column_config.TextColumn("First Date", width="small"),
                "last_date": st.column_config.TextColumn("Last Date", width="small"),
                "currency": st.column_config.TextColumn("Currency", width="small"),
            }
        )

    st.divider()

    # Price history for a symbol
    st.subheader("📉 Price History")

    if selected_symbol != "All":
        history_symbol = selected_symbol
    else:
        history_symbol = st.selectbox(
            "Select symbol for price history",
            symbols_df['symbol'].tolist() if not symbols_df.empty else [],
            key="mds_history_symbol"
        )

    if history_symbol:
        history_df = pd.read_sql_query(
            "SELECT date, price FROM market_prices WHERE symbol = ? ORDER BY date",
            conn,
            params=[history_symbol]
        )

        if not history_df.empty and len(history_df) > 1:
            history_df['date'] = pd.to_datetime(history_df['date'])
            history_df['price'] = history_df['price'].astype(float)

            fig = px.line(
                history_df,
                x='date',
                y='price',
                title=f'{history_symbol} Price History',
                labels={'date': 'Date', 'price': 'Price'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        elif len(history_df) == 1:
            st.info(f"Only 1 price point for {history_symbol}: {history_df['price'].iloc[0]} on {history_df['date'].iloc[0]}")
        else:
            st.info(f"No price history for {history_symbol}")

    st.divider()

    # Danger zone - data management
    with st.expander("⚠️ Data Management", expanded=False):
        st.warning("These actions modify the market data store and affect all portfolios.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Delete prices for a specific date**")
            delete_date = st.date_input("Date to delete", key="mds_delete_date")
            if st.button("Delete Date", type="secondary", key="mds_delete_date_btn"):
                cursor = conn.cursor()
                cursor.execute("DELETE FROM market_prices WHERE date = ?", (delete_date.isoformat(),))
                deleted = cursor.rowcount
                conn.commit()
                st.success(f"Deleted {deleted} prices for {delete_date}")
                st.rerun()

        with col2:
            st.markdown("**Delete prices for a specific symbol**")
            delete_symbol = st.text_input("Symbol to delete", key="mds_delete_symbol")
            if st.button("Delete Symbol", type="secondary", key="mds_delete_symbol_btn"):
                if delete_symbol:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM market_prices WHERE symbol = ?", (delete_symbol.upper(),))
                    deleted = cursor.rowcount
                    conn.commit()
                    st.success(f"Deleted {deleted} prices for {delete_symbol.upper()}")
                    st.rerun()

        st.markdown("---")
        st.markdown("**Clear all market data**")
        confirm_clear = st.checkbox("I understand this will delete ALL market prices", key="mds_confirm_clear")
        if st.button("Clear All Data", type="primary", disabled=not confirm_clear, key="mds_clear_all"):
            cursor = conn.cursor()
            cursor.execute("DELETE FROM market_prices")
            deleted = cursor.rowcount
            conn.commit()
            st.success(f"Deleted all {deleted} prices")
            st.rerun()

    conn.close()
