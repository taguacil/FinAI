"""
Portfolio storage system for persisting data to filesystem.

This module provides file-based storage for portfolios and delegates
snapshot storage to the SQLite-based SnapshotStore for improved performance.
"""

import json
import logging
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from .models import Portfolio, PortfolioSnapshot
from .snapshot_store import SnapshotStore


class PortfolioEncoder(json.JSONEncoder):
    """Custom JSON encoder for portfolio objects."""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif hasattr(obj, "dict"):  # Pydantic models
            return obj.dict()
        return super().default(obj)


class PortfolioDecoder:
    """Helper class to decode JSON back to portfolio objects."""

    @staticmethod
    def decimal_hook(dct):
        """Convert float values back to Decimal for precision."""
        for key, value in dct.items():
            if isinstance(value, float) and key in [
                "quantity",
                "price",
                "fees",
                "average_cost",
                "current_price",
                "total_value",
                "cash_balance",
                "positions_value",
            ]:
                dct[key] = Decimal(str(value))
        return dct


class FileBasedStorage:
    """File-based storage system for portfolio data.

    Portfolio data is stored as JSON files for simplicity.
    Snapshot data is delegated to SQLite-based SnapshotStore for performance.
    """

    def __init__(self, data_dir: str = "data"):
        """Initialize storage with data directory."""
        self.data_dir = Path(data_dir)
        self.portfolios_dir = self.data_dir / "portfolios"
        self.snapshots_dir = self.data_dir / "snapshots"

        # Create directories if they don't exist
        self.portfolios_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SQLite-based snapshot store
        self._snapshot_store = SnapshotStore(data_dir)

        # Track which portfolios have been migrated
        self._migrated_portfolios: set = set()

    def save_portfolio(self, portfolio: Portfolio) -> None:
        """Save portfolio to file."""
        try:
            filepath = self.portfolios_dir / f"{portfolio.id}.json"

            # Validate portfolio ID to prevent path traversal
            if not portfolio.id.replace("-", "").replace("_", "").isalnum():
                raise ValueError(f"Invalid portfolio ID: {portfolio.id}")

            # Create backup if file exists
            if filepath.exists():
                backup_path = filepath.with_suffix(".json.backup")
                filepath.rename(backup_path)

            with open(filepath, "w") as f:
                json.dump(portfolio.dict(), f, cls=PortfolioEncoder, indent=2)

        except Exception as e:
            # Restore backup if save failed
            backup_path = filepath.with_suffix(".json.backup")
            if backup_path.exists():
                backup_path.rename(filepath)
            raise RuntimeError(f"Failed to save portfolio {portfolio.id}: {e}") from e

    def load_portfolio(self, portfolio_id: str) -> Optional[Portfolio]:
        """Load portfolio from file."""
        filepath = self.portfolios_dir / f"{portfolio_id}.json"

        if not filepath.exists():
            return None

        try:
            with open(filepath, "r") as f:
                data = json.load(f, object_hook=PortfolioDecoder.decimal_hook)

            return Portfolio(**data)
        except Exception as e:
            print(f"Error loading portfolio {portfolio_id}: {e}")
            return None

    def list_portfolios(self) -> List[str]:
        """List all available portfolio IDs."""
        return [f.stem for f in self.portfolios_dir.glob("*.json")]

    def delete_portfolio(self, portfolio_id: str, delete_all_data: bool = True) -> dict:
        """Delete a portfolio and optionally all associated data.

        Args:
            portfolio_id: The portfolio ID to delete
            delete_all_data: If True, also deletes snapshots, backups, and exports

        Returns:
            Dictionary with deletion results:
            - portfolio_deleted: bool
            - snapshots_deleted: int (number of snapshots deleted)
            - backup_deleted: bool
            - legacy_data_deleted: bool
        """
        result = {
            "portfolio_deleted": False,
            "snapshots_deleted": 0,
            "backup_deleted": False,
            "legacy_data_deleted": False,
        }

        # Delete portfolio JSON file
        filepath = self.portfolios_dir / f"{portfolio_id}.json"
        if filepath.exists():
            filepath.unlink()
            result["portfolio_deleted"] = True

        # Delete backup file if exists
        backup_path = filepath.with_suffix(".json.backup")
        if backup_path.exists():
            backup_path.unlink()
            result["backup_deleted"] = True

        if delete_all_data:
            # Delete all snapshots from SQLite (no date range = delete all)
            result["snapshots_deleted"] = self._snapshot_store.delete_snapshots(portfolio_id)

            # Delete legacy JSON snapshot file if exists
            legacy_json = self.snapshots_dir / f"{portfolio_id}.json"
            if legacy_json.exists():
                legacy_json.unlink()
                result["legacy_data_deleted"] = True

            # Delete legacy snapshot directory if exists
            legacy_dir = self.snapshots_dir / portfolio_id
            if legacy_dir.exists() and legacy_dir.is_dir():
                import shutil
                shutil.rmtree(legacy_dir)
                result["legacy_data_deleted"] = True

            # Delete backups directory for this portfolio
            backup_dir = self.data_dir / "backups" / portfolio_id
            if backup_dir.exists() and backup_dir.is_dir():
                import shutil
                shutil.rmtree(backup_dir)
                result["backup_deleted"] = True

            # Delete exports for this portfolio
            exports_dir = self.data_dir / "exports"
            if exports_dir.exists():
                for export_file in exports_dir.glob(f"{portfolio_id}_*"):
                    export_file.unlink()

            # Remove from migrated portfolios cache
            self._migrated_portfolios.discard(portfolio_id)

        return result

    def save_snapshot(self, portfolio_id: str, snapshot: PortfolioSnapshot) -> None:
        """Save portfolio snapshot to SQLite store.

        Args:
            portfolio_id: The portfolio ID
            snapshot: The snapshot to save
        """
        if not portfolio_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid portfolio ID: {portfolio_id}")

        # Ensure migration has happened for this portfolio
        self._ensure_migrated(portfolio_id)

        # Delegate to SQLite store
        self._snapshot_store.save_snapshot(portfolio_id, snapshot)

    def save_snapshots_batch(self, portfolio_id: str, snapshots: List[PortfolioSnapshot]) -> None:
        """Save multiple portfolio snapshots efficiently.

        Uses SQLite transactions for atomic batch inserts.

        Args:
            portfolio_id: The portfolio ID
            snapshots: List of snapshots to save
        """
        if not snapshots:
            return

        if not portfolio_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid portfolio ID: {portfolio_id}")

        # Ensure migration has happened for this portfolio
        self._ensure_migrated(portfolio_id)

        # Delegate to SQLite store
        self._snapshot_store.save_snapshots_batch(portfolio_id, snapshots)

    def load_snapshots(
        self,
        portfolio_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[PortfolioSnapshot]:
        """Load portfolio snapshots within date range.

        Uses SQLite for efficient indexed queries.

        Args:
            portfolio_id: The portfolio ID
            start_date: Optional start date (inclusive)
            end_date: Optional end date (inclusive)

        Returns:
            List of PortfolioSnapshot objects, sorted by date
        """
        # Ensure migration has happened for this portfolio
        self._ensure_migrated(portfolio_id)

        # Delegate to SQLite store
        return self._snapshot_store.load_snapshots(portfolio_id, start_date, end_date)

    def get_latest_snapshot(self, portfolio_id: str) -> Optional[PortfolioSnapshot]:
        """Get the most recent snapshot for a portfolio."""
        self._ensure_migrated(portfolio_id)
        return self._snapshot_store.get_latest_snapshot(portfolio_id)

    def get_snapshots_in_range(
        self, portfolio_id: str, start_date: date, end_date: date
    ) -> List[PortfolioSnapshot]:
        """Get snapshots within a specific date range."""
        return self.load_snapshots(portfolio_id, start_date, end_date)

    def load_snapshots_df(
        self,
        portfolio_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """Load snapshots as a DataFrame for analytics.

        Args:
            portfolio_id: The portfolio ID
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            DataFrame with snapshot data (date index, value columns)
        """
        self._ensure_migrated(portfolio_id)
        return self._snapshot_store.load_snapshots_df(portfolio_id, start_date, end_date)

    def get_snapshot_date_range(self, portfolio_id: str) -> Optional[Tuple[date, date]]:
        """Get the date range of available snapshots.

        Args:
            portfolio_id: The portfolio ID

        Returns:
            Tuple of (earliest_date, latest_date) or None if no snapshots
        """
        self._ensure_migrated(portfolio_id)
        return self._snapshot_store.get_date_range(portfolio_id)

    def get_snapshot_count(self, portfolio_id: str) -> int:
        """Get the number of snapshots for a portfolio.

        Args:
            portfolio_id: The portfolio ID

        Returns:
            Number of snapshots
        """
        self._ensure_migrated(portfolio_id)
        return self._snapshot_store.get_snapshot_count(portfolio_id)

    def delete_snapshots(
        self,
        portfolio_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> int:
        """Delete snapshots within a date range.

        Args:
            portfolio_id: The portfolio ID
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Number of snapshots deleted
        """
        self._ensure_migrated(portfolio_id)
        return self._snapshot_store.delete_snapshots(portfolio_id, start_date, end_date)

    def _ensure_migrated(self, portfolio_id: str) -> None:
        """Ensure JSON snapshots have been migrated to SQLite.

        Checks if there's a JSON file that hasn't been migrated yet,
        and if so, migrates it to the SQLite store.

        Args:
            portfolio_id: The portfolio ID
        """
        # Skip if already migrated in this session
        if portfolio_id in self._migrated_portfolios:
            return

        # Check if SQLite already has data for this portfolio
        if self._snapshot_store.has_snapshots(portfolio_id):
            self._migrated_portfolios.add(portfolio_id)
            return

        # Check for JSON file to migrate
        json_path = self.snapshots_dir / f"{portfolio_id}.json"
        if json_path.exists():
            logging.info(f"Migrating snapshots from JSON for portfolio {portfolio_id}")
            migrated_count = self._snapshot_store.migrate_from_json(portfolio_id, json_path)
            if migrated_count > 0:
                logging.info(f"Successfully migrated {migrated_count} snapshots to SQLite")
                # Keep JSON file as backup (don't delete)

        # Also check for legacy per-day directory structure
        legacy_dir = self.snapshots_dir / portfolio_id
        if legacy_dir.exists() and legacy_dir.is_dir():
            # First consolidate to JSON, then migrate
            try:
                legacy_snapshots: List[PortfolioSnapshot] = []
                for filepath in legacy_dir.glob("*.json"):
                    try:
                        with open(filepath, "r") as f:
                            data = json.load(f, object_hook=PortfolioDecoder.decimal_hook)
                        legacy_snapshots.append(PortfolioSnapshot(**data))
                    except Exception as e:
                        logging.warning(f"Error loading legacy snapshot {filepath}: {e}")
                        continue

                if legacy_snapshots:
                    self._snapshot_store.save_snapshots_batch(portfolio_id, legacy_snapshots)
                    logging.info(f"Migrated {len(legacy_snapshots)} legacy snapshots for {portfolio_id}")
            except Exception as e:
                logging.error(f"Error migrating legacy snapshots for {portfolio_id}: {e}")

        self._migrated_portfolios.add(portfolio_id)

    def backup_portfolio(self, portfolio_id: str) -> str:
        """Create a backup of portfolio data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.data_dir / "backups" / portfolio_id
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Copy portfolio file
        portfolio_file = self.portfolios_dir / f"{portfolio_id}.json"
        if portfolio_file.exists():
            backup_file = backup_dir / f"{portfolio_id}_{timestamp}.json"
            backup_file.write_text(portfolio_file.read_text())
            return str(backup_file)

        return ""

    def export_transactions(self, portfolio_id: str, format: str = "csv") -> str:
        """Export transactions to CSV or JSON format."""
        portfolio = self.load_portfolio(portfolio_id)
        if not portfolio:
            return ""

        export_dir = self.data_dir / "exports"
        export_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format.lower() == "csv":
            import csv

            filepath = export_dir / f"{portfolio_id}_transactions_{timestamp}.csv"

            with open(filepath, "w", newline="") as csvfile:
                fieldnames = [
                    "id",
                    "timestamp",
                    "symbol",
                    "instrument_name",
                    "transaction_type",
                    "quantity",
                    "price",
                    "currency",
                    "total_value",
                    "notes",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for txn in portfolio.transactions:
                    writer.writerow(
                        {
                            "id": txn.id,
                            "timestamp": txn.timestamp.isoformat(),
                            "symbol": txn.instrument.symbol,
                            "instrument_name": txn.instrument.name,
                            "transaction_type": txn.transaction_type,
                            "quantity": float(txn.quantity),
                            "price": float(txn.price),
                            "currency": txn.currency,
                            "total_value": float(txn.total_value),
                            "notes": txn.notes or "",
                        }
                    )

        else:  # JSON format
            filepath = export_dir / f"{portfolio_id}_transactions_{timestamp}.json"

            transactions_data = [txn.dict() for txn in portfolio.transactions]
            with open(filepath, "w") as f:
                json.dump(transactions_data, f, cls=PortfolioEncoder, indent=2)

        return str(filepath)
