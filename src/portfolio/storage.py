"""
Portfolio storage system for persisting data to filesystem.
"""

import json
import os
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

from .models import Portfolio, Transaction, PortfolioSnapshot


class PortfolioEncoder(json.JSONEncoder):
    """Custom JSON encoder for portfolio objects."""
    
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif hasattr(obj, 'dict'):  # Pydantic models
            return obj.dict()
        return super().default(obj)


class PortfolioDecoder:
    """Helper class to decode JSON back to portfolio objects."""
    
    @staticmethod
    def decimal_hook(dct):
        """Convert float values back to Decimal for precision."""
        for key, value in dct.items():
            if isinstance(value, float) and key in [
                'quantity', 'price', 'fees', 'average_cost', 'current_price',
                'total_value', 'cash_balance', 'positions_value'
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
        filepath = self.portfolios_dir / f"{portfolio.id}.json"
        
        with open(filepath, 'w') as f:
            json.dump(portfolio.dict(), f, cls=PortfolioEncoder, indent=2)
    
    def load_portfolio(self, portfolio_id: str) -> Optional[Portfolio]:
        """Load portfolio from file."""
        filepath = self.portfolios_dir / f"{portfolio_id}.json"
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r') as f:
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
        """Save portfolio snapshot to file."""
        portfolio_snapshots_dir = self.snapshots_dir / portfolio_id
        portfolio_snapshots_dir.mkdir(exist_ok=True)
        
        filepath = portfolio_snapshots_dir / f"{snapshot.date.isoformat()}.json"
        
        with open(filepath, 'w') as f:
            json.dump(snapshot.dict(), f, cls=PortfolioEncoder, indent=2)
    
    def load_snapshots(self, portfolio_id: str, 
                      start_date: Optional[date] = None,
                      end_date: Optional[date] = None) -> List[PortfolioSnapshot]:
        """Load portfolio snapshots within date range."""
        portfolio_snapshots_dir = self.snapshots_dir / portfolio_id
        
        if not portfolio_snapshots_dir.exists():
            return []
        
        snapshots = []
        for filepath in portfolio_snapshots_dir.glob("*.json"):
            try:
                snapshot_date = date.fromisoformat(filepath.stem)
                
                # Filter by date range if specified
                if start_date and snapshot_date < start_date:
                    continue
                if end_date and snapshot_date > end_date:
                    continue
                
                with open(filepath, 'r') as f:
                    data = json.load(f, object_hook=PortfolioDecoder.decimal_hook)
                
                snapshots.append(PortfolioSnapshot(**data))
            except Exception as e:
                print(f"Error loading snapshot {filepath}: {e}")
                continue
        
        return sorted(snapshots, key=lambda x: x.date)
    
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
            
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = [
                    'id', 'timestamp', 'symbol', 'instrument_name', 'transaction_type',
                    'quantity', 'price', 'fees', 'currency', 'total_value', 'notes'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for txn in portfolio.transactions:
                    writer.writerow({
                        'id': txn.id,
                        'timestamp': txn.timestamp.isoformat(),
                        'symbol': txn.instrument.symbol,
                        'instrument_name': txn.instrument.name,
                        'transaction_type': txn.transaction_type,
                        'quantity': float(txn.quantity),
                        'price': float(txn.price),
                        'fees': float(txn.fees),
                        'currency': txn.currency,
                        'total_value': float(txn.total_value),
                        'notes': txn.notes or ''
                    })
        
        else:  # JSON format
            filepath = export_dir / f"{portfolio_id}_transactions_{timestamp}.json"
            
            transactions_data = [txn.dict() for txn in portfolio.transactions]
            with open(filepath, 'w') as f:
                json.dump(transactions_data, f, cls=PortfolioEncoder, indent=2)
        
        return str(filepath)