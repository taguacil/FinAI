"""
Portfolio storage system for persisting data to filesystem.
"""

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import List, Optional

from .models import Portfolio, PortfolioSnapshot


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
    """File-based storage system for portfolio data."""

    def __init__(self, data_dir: str = "data"):
        """Initialize storage with data directory."""
        self.data_dir = Path(data_dir)
        self.portfolios_dir = self.data_dir / "portfolios"
        self.snapshots_dir = self.data_dir / "snapshots"

        # Create directories if they don't exist
        self.portfolios_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

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

    def delete_portfolio(self, portfolio_id: str) -> bool:
        """Delete a portfolio file."""
        filepath = self.portfolios_dir / f"{portfolio_id}.json"

        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def save_snapshot(self, portfolio_id: str, snapshot: PortfolioSnapshot) -> None:
        """Save portfolio snapshot into a consolidated file per portfolio.

        New structure: data/snapshots/{portfolio_id}.json containing an array of
        daily snapshots. If a snapshot for the date exists, it is replaced.
        """
        try:
            if not portfolio_id.replace("-", "").replace("_", "").isalnum():
                raise ValueError(f"Invalid portfolio ID: {portfolio_id}")

            consolidated_path = self.snapshots_dir / f"{portfolio_id}.json"

            snapshots_data = []
            if consolidated_path.exists():
                with open(consolidated_path, "r") as f:
                    snapshots_data = json.load(
                        f, object_hook=PortfolioDecoder.decimal_hook
                    )

                # Ensure snapshots_data is a list
                if isinstance(snapshots_data, dict):
                    # Migrate old single-snapshot structure if encountered
                    snapshots_data = [snapshots_data]

            # Remove any existing snapshot for this date
            snapshot_date_str = snapshot.date.isoformat()
            snapshots_data = [
                s for s in snapshots_data if s.get("date") != snapshot_date_str
            ]

            # Append new snapshot (ensure date is a string for consistent sorting)
            new_item = snapshot.dict()
            new_item["date"] = snapshot_date_str
            snapshots_data.append(new_item)

            # Sort by date
            snapshots_data.sort(key=lambda s: s.get("date"))

            with open(consolidated_path, "w") as f:
                json.dump(snapshots_data, f, cls=PortfolioEncoder, indent=2)

        except Exception as e:
            raise RuntimeError(
                f"Failed to save snapshot for portfolio {portfolio_id}: {e}"
            ) from e

    def load_snapshots(
        self,
        portfolio_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[PortfolioSnapshot]:
        """Load portfolio snapshots within date range."""
        consolidated_path = self.snapshots_dir / f"{portfolio_id}.json"

        # If consolidated file doesn't exist, attempt migration from legacy per-day files
        if not consolidated_path.exists():
            legacy_dir = self.snapshots_dir / portfolio_id
            if legacy_dir.exists():
                try:
                    legacy_snapshots: List[PortfolioSnapshot] = []
                    for filepath in legacy_dir.glob("*.json"):
                        try:
                            with open(filepath, "r") as f:
                                data = json.load(
                                    f, object_hook=PortfolioDecoder.decimal_hook
                                )
                            legacy_snapshots.append(PortfolioSnapshot(**data))
                        except Exception as e:
                            print(f"Error loading legacy snapshot {filepath}: {e}")
                            continue

                    # Write consolidated file
                    legacy_snapshots_sorted = sorted(
                        legacy_snapshots, key=lambda x: x.date
                    )
                    with open(consolidated_path, "w") as f:
                        json.dump(
                            [s.dict() for s in legacy_snapshots_sorted],
                            f,
                            cls=PortfolioEncoder,
                            indent=2,
                        )
                except Exception as e:
                    print(f"Error migrating legacy snapshots for {portfolio_id}: {e}")
                    # Fall through to return from legacy in-memory if migration fails
                    pass

        if not consolidated_path.exists():
            return []

        try:
            with open(consolidated_path, "r") as f:
                snapshots_data = json.load(f, object_hook=PortfolioDecoder.decimal_hook)

            snapshots: List[PortfolioSnapshot] = []
            for item in snapshots_data:
                try:
                    snap = PortfolioSnapshot(**item)
                    if start_date and snap.date < start_date:
                        continue
                    if end_date and snap.date > end_date:
                        continue
                    snapshots.append(snap)
                except Exception as e:
                    print(f"Error parsing snapshot item: {e}")
                    continue

            return sorted(snapshots, key=lambda x: x.date)
        except Exception as e:
            print(f"Error loading snapshots for {portfolio_id}: {e}")
            return []

    def get_latest_snapshot(self, portfolio_id: str) -> Optional[PortfolioSnapshot]:
        """Get the most recent snapshot for a portfolio."""
        snapshots = self.load_snapshots(portfolio_id)
        return snapshots[-1] if snapshots else None

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
