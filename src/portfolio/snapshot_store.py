"""
SQLite-based snapshot storage with efficient date-range queries.

This module provides a SQLite backend for portfolio snapshots, replacing
the JSON-based storage for improved performance on range queries and
atomic operations.
"""

import json
import logging
import sqlite3
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .models import Currency, FinancialInstrument, Portfolio, PortfolioSnapshot, Position


class SnapshotStore:
    """SQLite-based snapshot storage with efficient queries.

    Features:
    - Efficient date-range queries using indexes
    - Atomic operations with built-in locking
    - Decimal precision preserved via TEXT storage
    - DataFrame export for analytics
    - Automatic migration from JSON files
    """

    def __init__(self, data_dir: str = "data"):
        """Initialize the snapshot store.

        Args:
            data_dir: Base data directory
        """
        self.data_dir = Path(data_dir)
        self.snapshots_dir = self.data_dir / "snapshots"
        self.db_path = self.data_dir / "snapshots.db"

        # Ensure directories exist
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")  # Better concurrent access
        return conn

    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Create snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    total_value TEXT NOT NULL,
                    cash_balance TEXT NOT NULL,
                    positions_value TEXT NOT NULL,
                    base_currency TEXT NOT NULL,
                    total_cost_basis TEXT NOT NULL,
                    total_unrealized_pnl TEXT NOT NULL,
                    total_unrealized_pnl_percent TEXT NOT NULL,
                    cash_balances_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(portfolio_id, date)
                )
            """)

            # Create positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS snapshot_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    instrument_json TEXT NOT NULL,
                    quantity TEXT NOT NULL,
                    average_cost TEXT NOT NULL,
                    current_price TEXT,
                    last_updated TEXT,
                    FOREIGN KEY (snapshot_id) REFERENCES snapshots(id) ON DELETE CASCADE
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshots_portfolio_date
                ON snapshots(portfolio_id, date)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_positions_snapshot
                ON snapshot_positions(snapshot_id)
            """)

            conn.commit()
            logging.info("Snapshot database initialized")

        except Exception as e:
            logging.error(f"Error initializing snapshot database: {e}")
            raise
        finally:
            conn.close()

    def _decimal_to_str(self, value: Decimal) -> str:
        """Convert Decimal to string for storage."""
        return str(value)

    def _str_to_decimal(self, value: str) -> Decimal:
        """Convert stored string back to Decimal."""
        return Decimal(value)

    def _instrument_to_json(self, instrument: FinancialInstrument) -> str:
        """Serialize instrument to JSON string."""
        return json.dumps({
            "symbol": instrument.symbol,
            "isin": instrument.isin,
            "name": instrument.name,
            "instrument_type": instrument.instrument_type.value,
            "currency": instrument.currency.value,
            "exchange": instrument.exchange,
        })

    def _json_to_instrument(self, json_str: str) -> FinancialInstrument:
        """Deserialize instrument from JSON string."""
        from .models import InstrumentType

        data = json.loads(json_str)
        return FinancialInstrument(
            symbol=data["symbol"],
            isin=data.get("isin"),
            name=data["name"],
            instrument_type=InstrumentType(data["instrument_type"]),
            currency=Currency(data["currency"]),
            exchange=data.get("exchange"),
        )

    def _cash_balances_to_json(self, balances: Dict[Currency, Decimal]) -> str:
        """Serialize cash balances to JSON string."""
        return json.dumps({
            cur.value if hasattr(cur, "value") else cur: str(val)
            for cur, val in balances.items()
        })

    def _json_to_cash_balances(self, json_str: str) -> Dict[Currency, Decimal]:
        """Deserialize cash balances from JSON string."""
        data = json.loads(json_str)
        return {Currency(k): Decimal(v) for k, v in data.items()}

    def save_snapshot(self, portfolio_id: str, snapshot: PortfolioSnapshot) -> None:
        """Save a single snapshot.

        If a snapshot for this portfolio and date exists, it is replaced.

        Args:
            portfolio_id: The portfolio ID
            snapshot: The snapshot to save
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Delete existing snapshot for this date if exists
            cursor.execute("""
                DELETE FROM snapshots
                WHERE portfolio_id = ? AND date = ?
            """, (portfolio_id, snapshot.date.isoformat()))

            # Insert new snapshot
            cursor.execute("""
                INSERT INTO snapshots (
                    portfolio_id, date, total_value, cash_balance, positions_value,
                    base_currency, total_cost_basis, total_unrealized_pnl,
                    total_unrealized_pnl_percent, cash_balances_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                portfolio_id,
                snapshot.date.isoformat(),
                self._decimal_to_str(snapshot.total_value),
                self._decimal_to_str(snapshot.cash_balance),
                self._decimal_to_str(snapshot.positions_value),
                snapshot.base_currency.value,
                self._decimal_to_str(snapshot.total_cost_basis),
                self._decimal_to_str(snapshot.total_unrealized_pnl),
                self._decimal_to_str(snapshot.total_unrealized_pnl_percent),
                self._cash_balances_to_json(snapshot.cash_balances),
                datetime.now().isoformat(),
            ))

            snapshot_id = cursor.lastrowid

            # Insert positions
            for symbol, position in snapshot.positions.items():
                cursor.execute("""
                    INSERT INTO snapshot_positions (
                        snapshot_id, symbol, instrument_json, quantity,
                        average_cost, current_price, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    snapshot_id,
                    symbol,
                    self._instrument_to_json(position.instrument),
                    self._decimal_to_str(position.quantity),
                    self._decimal_to_str(position.average_cost),
                    self._decimal_to_str(position.current_price) if position.current_price else None,
                    position.last_updated.isoformat() if position.last_updated else None,
                ))

            conn.commit()
            logging.debug(f"Saved snapshot for {portfolio_id} on {snapshot.date}")

        except Exception as e:
            conn.rollback()
            logging.error(f"Error saving snapshot: {e}")
            raise
        finally:
            conn.close()

    def save_snapshots_batch(
        self, portfolio_id: str, snapshots: List[PortfolioSnapshot]
    ) -> None:
        """Save multiple snapshots in a single transaction.

        Much more efficient than calling save_snapshot multiple times.

        Args:
            portfolio_id: The portfolio ID
            snapshots: List of snapshots to save
        """
        if not snapshots:
            return

        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Get dates to update
            dates = [s.date.isoformat() for s in snapshots]
            placeholders = ",".join("?" * len(dates))

            # Delete existing snapshots for these dates
            cursor.execute(f"""
                DELETE FROM snapshots
                WHERE portfolio_id = ? AND date IN ({placeholders})
            """, [portfolio_id] + dates)

            # Insert all snapshots
            for snapshot in snapshots:
                cursor.execute("""
                    INSERT INTO snapshots (
                        portfolio_id, date, total_value, cash_balance, positions_value,
                        base_currency, total_cost_basis, total_unrealized_pnl,
                        total_unrealized_pnl_percent, cash_balances_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    portfolio_id,
                    snapshot.date.isoformat(),
                    self._decimal_to_str(snapshot.total_value),
                    self._decimal_to_str(snapshot.cash_balance),
                    self._decimal_to_str(snapshot.positions_value),
                    snapshot.base_currency.value,
                    self._decimal_to_str(snapshot.total_cost_basis),
                    self._decimal_to_str(snapshot.total_unrealized_pnl),
                    self._decimal_to_str(snapshot.total_unrealized_pnl_percent),
                    self._cash_balances_to_json(snapshot.cash_balances),
                    datetime.now().isoformat(),
                ))

                snapshot_id = cursor.lastrowid

                # Insert positions
                for symbol, position in snapshot.positions.items():
                    cursor.execute("""
                        INSERT INTO snapshot_positions (
                            snapshot_id, symbol, instrument_json, quantity,
                            average_cost, current_price, last_updated
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        snapshot_id,
                        symbol,
                        self._instrument_to_json(position.instrument),
                        self._decimal_to_str(position.quantity),
                        self._decimal_to_str(position.average_cost),
                        self._decimal_to_str(position.current_price) if position.current_price else None,
                        position.last_updated.isoformat() if position.last_updated else None,
                    ))

            conn.commit()
            logging.info(f"Batch saved {len(snapshots)} snapshots for portfolio {portfolio_id}")

        except Exception as e:
            conn.rollback()
            logging.error(f"Error batch saving snapshots: {e}")
            raise
        finally:
            conn.close()

    def load_snapshots(
        self,
        portfolio_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[PortfolioSnapshot]:
        """Load snapshots within a date range.

        Args:
            portfolio_id: The portfolio ID
            start_date: Optional start date (inclusive)
            end_date: Optional end date (inclusive)

        Returns:
            List of PortfolioSnapshot objects, sorted by date
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Build query with optional date filters
            query = "SELECT * FROM snapshots WHERE portfolio_id = ?"
            params: List = [portfolio_id]

            if start_date:
                query += " AND date >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND date <= ?"
                params.append(end_date.isoformat())

            query += " ORDER BY date"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            snapshots = []
            for row in rows:
                snapshot_id = row["id"]

                # Load positions for this snapshot
                cursor.execute("""
                    SELECT * FROM snapshot_positions WHERE snapshot_id = ?
                """, (snapshot_id,))
                position_rows = cursor.fetchall()

                positions: Dict[str, Position] = {}
                for pos_row in position_rows:
                    instrument = self._json_to_instrument(pos_row["instrument_json"])
                    positions[pos_row["symbol"]] = Position(
                        instrument=instrument,
                        quantity=self._str_to_decimal(pos_row["quantity"]),
                        average_cost=self._str_to_decimal(pos_row["average_cost"]),
                        current_price=self._str_to_decimal(pos_row["current_price"]) if pos_row["current_price"] else None,
                        last_updated=datetime.fromisoformat(pos_row["last_updated"]) if pos_row["last_updated"] else None,
                    )

                snapshot = PortfolioSnapshot(
                    date=date.fromisoformat(row["date"]),
                    total_value=self._str_to_decimal(row["total_value"]),
                    cash_balance=self._str_to_decimal(row["cash_balance"]),
                    positions_value=self._str_to_decimal(row["positions_value"]),
                    base_currency=Currency(row["base_currency"]),
                    positions=positions,
                    cash_balances=self._json_to_cash_balances(row["cash_balances_json"]),
                    total_cost_basis=self._str_to_decimal(row["total_cost_basis"]),
                    total_unrealized_pnl=self._str_to_decimal(row["total_unrealized_pnl"]),
                    total_unrealized_pnl_percent=self._str_to_decimal(row["total_unrealized_pnl_percent"]),
                )
                snapshots.append(snapshot)

            return snapshots

        except Exception as e:
            logging.error(f"Error loading snapshots: {e}")
            return []
        finally:
            conn.close()

    def load_snapshots_df(
        self,
        portfolio_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Load snapshots as a DataFrame for analytics.

        Returns a DataFrame with columns:
        - date (index)
        - total_value
        - cash_balance
        - positions_value
        - total_cost_basis
        - total_unrealized_pnl
        - total_unrealized_pnl_percent

        Args:
            portfolio_id: The portfolio ID
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with snapshot data
        """
        conn = self._get_connection()
        try:
            query = """
                SELECT date, total_value, cash_balance, positions_value,
                       total_cost_basis, total_unrealized_pnl, total_unrealized_pnl_percent
                FROM snapshots
                WHERE portfolio_id = ?
            """
            params: List = [portfolio_id]

            if start_date:
                query += " AND date >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND date <= ?"
                params.append(end_date.isoformat())

            query += " ORDER BY date"

            df = pd.read_sql_query(query, conn, params=params)

            if df.empty:
                return pd.DataFrame(columns=[
                    "date", "total_value", "cash_balance", "positions_value",
                    "total_cost_basis", "total_unrealized_pnl", "total_unrealized_pnl_percent"
                ]).set_index("date")

            # Convert string columns to numeric
            numeric_cols = [
                "total_value", "cash_balance", "positions_value",
                "total_cost_basis", "total_unrealized_pnl", "total_unrealized_pnl_percent"
            ]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])

            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

            return df

        except Exception as e:
            logging.error(f"Error loading snapshots DataFrame: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def get_latest_snapshot(self, portfolio_id: str) -> Optional[PortfolioSnapshot]:
        """Get the most recent snapshot for a portfolio.

        Args:
            portfolio_id: The portfolio ID

        Returns:
            The most recent PortfolioSnapshot or None
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT date FROM snapshots
                WHERE portfolio_id = ?
                ORDER BY date DESC
                LIMIT 1
            """, (portfolio_id,))

            row = cursor.fetchone()
            if not row:
                return None

            latest_date = date.fromisoformat(row["date"])
            snapshots = self.load_snapshots(portfolio_id, latest_date, latest_date)
            return snapshots[0] if snapshots else None

        except Exception as e:
            logging.error(f"Error getting latest snapshot: {e}")
            return None
        finally:
            conn.close()

    def get_date_range(self, portfolio_id: str) -> Optional[Tuple[date, date]]:
        """Get the date range of available snapshots.

        Args:
            portfolio_id: The portfolio ID

        Returns:
            Tuple of (earliest_date, latest_date) or None if no snapshots
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT MIN(date) as min_date, MAX(date) as max_date
                FROM snapshots
                WHERE portfolio_id = ?
            """, (portfolio_id,))

            row = cursor.fetchone()
            if not row or not row["min_date"]:
                return None

            return (
                date.fromisoformat(row["min_date"]),
                date.fromisoformat(row["max_date"]),
            )

        except Exception as e:
            logging.error(f"Error getting date range: {e}")
            return None
        finally:
            conn.close()

    def delete_snapshots(
        self,
        portfolio_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> int:
        """Delete snapshots within a date range.

        Args:
            portfolio_id: The portfolio ID
            start_date: Optional start date (inclusive)
            end_date: Optional end date (inclusive)

        Returns:
            Number of snapshots deleted
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            query = "DELETE FROM snapshots WHERE portfolio_id = ?"
            params: List = [portfolio_id]

            if start_date:
                query += " AND date >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND date <= ?"
                params.append(end_date.isoformat())

            cursor.execute(query, params)
            deleted = cursor.rowcount
            conn.commit()

            logging.info(f"Deleted {deleted} snapshots for {portfolio_id}")
            return deleted

        except Exception as e:
            conn.rollback()
            logging.error(f"Error deleting snapshots: {e}")
            return 0
        finally:
            conn.close()

    def get_snapshot_count(self, portfolio_id: str) -> int:
        """Get the number of snapshots for a portfolio.

        Args:
            portfolio_id: The portfolio ID

        Returns:
            Number of snapshots
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as count FROM snapshots WHERE portfolio_id = ?
            """, (portfolio_id,))
            row = cursor.fetchone()
            return row["count"] if row else 0
        except Exception as e:
            logging.error(f"Error getting snapshot count: {e}")
            return 0
        finally:
            conn.close()

    def migrate_from_json(self, portfolio_id: str, json_path: Path) -> int:
        """Migrate snapshots from a JSON file to SQLite.

        Args:
            portfolio_id: The portfolio ID
            json_path: Path to the JSON file

        Returns:
            Number of snapshots migrated
        """
        if not json_path.exists():
            return 0

        try:
            with open(json_path, "r") as f:
                snapshots_data = json.load(f)

            if isinstance(snapshots_data, dict):
                # Single snapshot format
                snapshots_data = [snapshots_data]

            snapshots = []
            for item in snapshots_data:
                try:
                    # Parse positions
                    positions: Dict[str, Position] = {}
                    for symbol, pos_data in item.get("positions", {}).items():
                        inst_data = pos_data.get("instrument", {})
                        from .models import InstrumentType

                        instrument = FinancialInstrument(
                            symbol=inst_data.get("symbol", ""),
                            isin=inst_data.get("isin"),
                            name=inst_data.get("name", "Unknown"),
                            instrument_type=InstrumentType(inst_data.get("instrument_type", "stock")),
                            currency=Currency(inst_data.get("currency", "USD")),
                            exchange=inst_data.get("exchange"),
                        )

                        current_price = pos_data.get("current_price")
                        if current_price is not None:
                            current_price = Decimal(str(current_price))

                        last_updated = pos_data.get("last_updated")
                        if last_updated:
                            last_updated = datetime.fromisoformat(last_updated)

                        positions[symbol] = Position(
                            instrument=instrument,
                            quantity=Decimal(str(pos_data.get("quantity", 0))),
                            average_cost=Decimal(str(pos_data.get("average_cost", 0))),
                            current_price=current_price,
                            last_updated=last_updated,
                        )

                    # Parse cash balances
                    cash_balances: Dict[Currency, Decimal] = {}
                    for cur, amount in item.get("cash_balances", {}).items():
                        cash_balances[Currency(cur)] = Decimal(str(amount))

                    snapshot = PortfolioSnapshot(
                        date=date.fromisoformat(item["date"]),
                        total_value=Decimal(str(item.get("total_value", 0))),
                        cash_balance=Decimal(str(item.get("cash_balance", 0))),
                        positions_value=Decimal(str(item.get("positions_value", 0))),
                        base_currency=Currency(item.get("base_currency", "USD")),
                        positions=positions,
                        cash_balances=cash_balances,
                        total_cost_basis=Decimal(str(item.get("total_cost_basis", 0))),
                        total_unrealized_pnl=Decimal(str(item.get("total_unrealized_pnl", 0))),
                        total_unrealized_pnl_percent=Decimal(str(item.get("total_unrealized_pnl_percent", 0))),
                    )
                    snapshots.append(snapshot)

                except Exception as e:
                    logging.warning(f"Error parsing snapshot item during migration: {e}")
                    continue

            if snapshots:
                self.save_snapshots_batch(portfolio_id, snapshots)
                logging.info(f"Migrated {len(snapshots)} snapshots from {json_path}")

            return len(snapshots)

        except Exception as e:
            logging.error(f"Error migrating from JSON: {e}")
            return 0

    def has_snapshots(self, portfolio_id: str) -> bool:
        """Check if any snapshots exist for a portfolio.

        Args:
            portfolio_id: The portfolio ID

        Returns:
            True if snapshots exist
        """
        return self.get_snapshot_count(portfolio_id) > 0
