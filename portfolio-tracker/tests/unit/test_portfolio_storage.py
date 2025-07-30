"""
Unit tests for portfolio storage functionality.
"""

import json
import pytest
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from src.portfolio.storage import FileBasedStorage, PortfolioEncoder, PortfolioDecoder
from src.portfolio.models import Portfolio, PortfolioSnapshot, Currency


class TestPortfolioEncoder:
    """Test cases for PortfolioEncoder."""

    def test_encode_decimal(self):
        """Test encoding Decimal values."""
        encoder = PortfolioEncoder()
        data = {"price": Decimal("123.45")}
        encoded = json.dumps(data, cls=PortfolioEncoder)
        
        # Should convert Decimal to float
        assert "123.45" in encoded
        
        # Decode back and verify
        decoded = json.loads(encoded)
        assert decoded["price"] == 123.45

    def test_encode_datetime(self):
        """Test encoding datetime values."""
        encoder = PortfolioEncoder()
        dt = datetime(2024, 1, 15, 10, 30, 0)
        data = {"timestamp": dt}
        encoded = json.dumps(data, cls=PortfolioEncoder)
        
        # Should convert to ISO format
        assert "2024-01-15T10:30:00" in encoded

    def test_encode_date(self):
        """Test encoding date values."""
        encoder = PortfolioEncoder()
        d = date(2024, 1, 15)
        data = {"date": d}
        encoded = json.dumps(data, cls=PortfolioEncoder)
        
        # Should convert to ISO format
        assert "2024-01-15" in encoded

    def test_encode_pydantic_model(self, sample_portfolio):
        """Test encoding Pydantic models."""
        encoder = PortfolioEncoder()
        encoded = json.dumps(sample_portfolio, cls=PortfolioEncoder)
        
        # Should be able to encode without error
        assert isinstance(encoded, str)
        
        # Should contain expected fields
        assert "test-portfolio-123" in encoded
        assert "Test Portfolio" in encoded


class TestPortfolioDecoder:
    """Test cases for PortfolioDecoder."""

    def test_decimal_hook(self):
        """Test converting float values back to Decimal."""
        data = {
            "quantity": 100.0,
            "price": 150.5,
            "fees": 1.0,
            "other_field": "not a decimal field"
        }
        
        result = PortfolioDecoder.decimal_hook(data)
        
        assert isinstance(result["quantity"], Decimal)
        assert isinstance(result["price"], Decimal)
        assert isinstance(result["fees"], Decimal)
        assert isinstance(result["other_field"], str)
        
        assert result["quantity"] == Decimal("100.0")
        assert result["price"] == Decimal("150.5")


class TestFileBasedStorage:
    """Test cases for FileBasedStorage."""

    def test_init_storage(self, temp_data_dir):
        """Test initializing storage creates directories."""
        storage = FileBasedStorage(temp_data_dir)
        
        # Check that directories are created
        assert storage.portfolios_dir.exists()
        assert storage.snapshots_dir.exists()
        assert storage.portfolios_dir.is_dir()
        assert storage.snapshots_dir.is_dir()

    def test_save_and_load_portfolio(self, storage, sample_portfolio):
        """Test saving and loading a portfolio."""
        # Save portfolio
        storage.save_portfolio(sample_portfolio)
        
        # Check file was created
        portfolio_file = storage.portfolios_dir / f"{sample_portfolio.id}.json"
        assert portfolio_file.exists()
        
        # Load portfolio back
        loaded_portfolio = storage.load_portfolio(sample_portfolio.id)
        
        assert loaded_portfolio is not None
        assert loaded_portfolio.id == sample_portfolio.id
        assert loaded_portfolio.name == sample_portfolio.name
        assert loaded_portfolio.base_currency == sample_portfolio.base_currency

    def test_load_nonexistent_portfolio(self, storage):
        """Test loading a portfolio that doesn't exist."""
        result = storage.load_portfolio("nonexistent-id")
        assert result is None

    def test_list_portfolios(self, storage, sample_portfolio):
        """Test listing available portfolios."""
        # Initially empty
        portfolios = storage.list_portfolios()
        assert len(portfolios) == 0
        
        # Save a portfolio
        storage.save_portfolio(sample_portfolio)
        
        # Should now appear in list
        portfolios = storage.list_portfolios()
        assert len(portfolios) == 1
        assert sample_portfolio.id in portfolios

    def test_delete_portfolio(self, storage, sample_portfolio):
        """Test deleting a portfolio."""
        # Save portfolio first
        storage.save_portfolio(sample_portfolio)
        assert sample_portfolio.id in storage.list_portfolios()
        
        # Delete it
        result = storage.delete_portfolio(sample_portfolio.id)
        assert result is True
        
        # Should no longer exist
        assert sample_portfolio.id not in storage.list_portfolios()
        
        # Deleting again should return False
        result = storage.delete_portfolio(sample_portfolio.id)
        assert result is False

    def test_save_and_load_snapshot(self, storage):
        """Test saving and loading portfolio snapshots."""
        portfolio_id = "test-portfolio"
        snapshot = PortfolioSnapshot(
            date=date(2024, 1, 15),
            total_value=Decimal("25000.00"),
            cash_balance=Decimal("5000.00"),
            positions_value=Decimal("20000.00"),
            base_currency=Currency.USD,
        )
        
        # Save snapshot
        storage.save_snapshot(portfolio_id, snapshot)
        
        # Check directory and file were created
        snapshot_dir = storage.snapshots_dir / portfolio_id
        assert snapshot_dir.exists()
        
        snapshot_file = snapshot_dir / "2024-01-15.json"
        assert snapshot_file.exists()
        
        # Load snapshots
        snapshots = storage.load_snapshots(portfolio_id)
        assert len(snapshots) == 1
        
        loaded_snapshot = snapshots[0]
        assert loaded_snapshot.date == snapshot.date
        assert loaded_snapshot.total_value == snapshot.total_value
        assert loaded_snapshot.cash_balance == snapshot.cash_balance

    def test_load_snapshots_with_date_filter(self, storage):
        """Test loading snapshots with date range filtering."""
        portfolio_id = "test-portfolio"
        
        # Create multiple snapshots
        dates = [date(2024, 1, 10), date(2024, 1, 15), date(2024, 1, 20)]
        for d in dates:
            snapshot = PortfolioSnapshot(
                date=d,
                total_value=Decimal("25000.00"),
                cash_balance=Decimal("5000.00"),
                positions_value=Decimal("20000.00"),
                base_currency=Currency.USD,
            )
            storage.save_snapshot(portfolio_id, snapshot)
        
        # Load all snapshots
        all_snapshots = storage.load_snapshots(portfolio_id)
        assert len(all_snapshots) == 3
        
        # Load with date filter
        filtered_snapshots = storage.load_snapshots(
            portfolio_id,
            start_date=date(2024, 1, 12),
            end_date=date(2024, 1, 18)
        )
        assert len(filtered_snapshots) == 1
        assert filtered_snapshots[0].date == date(2024, 1, 15)

    def test_get_latest_snapshot(self, storage):
        """Test getting the most recent snapshot."""
        portfolio_id = "test-portfolio"
        
        # No snapshots initially
        latest = storage.get_latest_snapshot(portfolio_id)
        assert latest is None
        
        # Add snapshots
        dates = [date(2024, 1, 10), date(2024, 1, 20), date(2024, 1, 15)]
        for d in dates:
            snapshot = PortfolioSnapshot(
                date=d,
                total_value=Decimal("25000.00"),
                cash_balance=Decimal("5000.00"),
                positions_value=Decimal("20000.00"),
                base_currency=Currency.USD,
            )
            storage.save_snapshot(portfolio_id, snapshot)
        
        # Should return the latest date
        latest = storage.get_latest_snapshot(portfolio_id)
        assert latest is not None
        assert latest.date == date(2024, 1, 20)

    def test_backup_portfolio(self, storage, sample_portfolio):
        """Test creating portfolio backup."""
        # Save portfolio first
        storage.save_portfolio(sample_portfolio)
        
        # Create backup
        backup_path = storage.backup_portfolio(sample_portfolio.id)
        
        assert backup_path != ""
        backup_file = Path(backup_path)
        assert backup_file.exists()
        assert sample_portfolio.id in backup_file.name

    def test_backup_nonexistent_portfolio(self, storage):
        """Test backing up a portfolio that doesn't exist."""
        backup_path = storage.backup_portfolio("nonexistent-id")
        assert backup_path == ""

    def test_export_transactions_csv(self, storage, portfolio_with_transactions):
        """Test exporting transactions to CSV format."""
        # Save portfolio first
        storage.save_portfolio(portfolio_with_transactions)
        
        # Export to CSV
        export_path = storage.export_transactions(portfolio_with_transactions.id, "csv")
        
        assert export_path != ""
        export_file = Path(export_path)
        assert export_file.exists()
        assert export_file.suffix == ".csv"
        
        # Check CSV content
        content = export_file.read_text()
        assert "id,timestamp,symbol" in content  # Header
        assert "AAPL" in content  # Transaction data

    def test_export_transactions_json(self, storage, portfolio_with_transactions):
        """Test exporting transactions to JSON format."""
        # Save portfolio first
        storage.save_portfolio(portfolio_with_transactions)
        
        # Export to JSON
        export_path = storage.export_transactions(portfolio_with_transactions.id, "json")
        
        assert export_path != ""
        export_file = Path(export_path)
        assert export_file.exists()
        assert export_file.suffix == ".json"
        
        # Check JSON content
        with open(export_file, 'r') as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "id" in data[0]

    def test_export_nonexistent_portfolio(self, storage):
        """Test exporting transactions for nonexistent portfolio."""
        export_path = storage.export_transactions("nonexistent-id")
        assert export_path == ""

    def test_portfolio_with_transactions_roundtrip(self, storage, portfolio_with_transactions):
        """Test saving and loading a portfolio with transactions."""
        # Save portfolio with transactions
        storage.save_portfolio(portfolio_with_transactions)
        
        # Load it back
        loaded_portfolio = storage.load_portfolio(portfolio_with_transactions.id)
        
        assert loaded_portfolio is not None
        assert len(loaded_portfolio.transactions) == len(portfolio_with_transactions.transactions)
        assert len(loaded_portfolio.positions) == len(portfolio_with_transactions.positions)
        assert len(loaded_portfolio.cash_balances) == len(portfolio_with_transactions.cash_balances)
        
        # Check transaction details
        original_txn = portfolio_with_transactions.transactions[0]
        loaded_txn = loaded_portfolio.transactions[0]
        
        assert loaded_txn.id == original_txn.id
        assert loaded_txn.instrument.symbol == original_txn.instrument.symbol
        assert loaded_txn.quantity == original_txn.quantity
        assert loaded_txn.price == original_txn.price

    def test_decimal_precision_preservation(self, storage, sample_portfolio):
        """Test that Decimal precision is preserved during save/load."""
        # Add a transaction with high precision
        from src.portfolio.models import Transaction, TransactionType, FinancialInstrument, InstrumentType
        
        instrument = FinancialInstrument(
            symbol="TEST",
            name="Test Stock",
            instrument_type=InstrumentType.STOCK,
            currency=Currency.USD,
        )
        
        transaction = Transaction(
            id="precision-test",
            timestamp=datetime.now(),
            instrument=instrument,
            transaction_type=TransactionType.BUY,
            quantity=Decimal("123.456789"),
            price=Decimal("987.654321"),
            fees=Decimal("0.123456789"),
            currency=Currency.USD,
        )
        
        sample_portfolio.add_transaction(transaction)
        
        # Save and load
        storage.save_portfolio(sample_portfolio)
        loaded_portfolio = storage.load_portfolio(sample_portfolio.id)
        
        loaded_txn = loaded_portfolio.transactions[0]
        assert loaded_txn.quantity == transaction.quantity
        assert loaded_txn.price == transaction.price
        assert loaded_txn.fees == transaction.fees

    def test_corrupted_portfolio_file(self, storage, temp_data_dir):
        """Test handling of corrupted portfolio files."""
        # Create a corrupted file
        portfolio_id = "corrupted-portfolio"
        portfolio_file = storage.portfolios_dir / f"{portfolio_id}.json"
        portfolio_file.write_text("{ invalid json content")
        
        # Should return None and not crash
        result = storage.load_portfolio(portfolio_id)
        assert result is None

    def test_portfolio_file_permissions(self, storage, sample_portfolio):
        """Test that portfolio files are created with proper permissions."""
        storage.save_portfolio(sample_portfolio)
        
        portfolio_file = storage.portfolios_dir / f"{sample_portfolio.id}.json"
        assert portfolio_file.exists()
        
        # File should be readable
        content = portfolio_file.read_text()
        assert len(content) > 0