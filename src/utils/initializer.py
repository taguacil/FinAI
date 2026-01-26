"""
Initialization system for the Portfolio Tracker application.
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from ..data_providers.manager import DataProviderManager
from ..portfolio.manager import PortfolioManager
from ..portfolio.models import Currency
from ..portfolio.storage import FileBasedStorage
from ..services.market_data_service import MarketDataService
from ..utils.metrics import FinancialMetricsCalculator


class PortfolioInitializer:
    """Handles application initialization and portfolio updates."""

    def __init__(self, data_dir: str = "data"):
        """Initialize the portfolio initializer."""
        self.data_dir = Path(data_dir)
        self.setup_logging()

        # Initialize components
        self.storage = FileBasedStorage(data_dir)
        self.data_manager = DataProviderManager()
        self.market_data_service = MarketDataService(self.data_manager)
        self.portfolio_manager = PortfolioManager(self.storage, self.market_data_service)
        self.metrics_calculator = FinancialMetricsCalculator(self.data_manager)

        logging.info("Portfolio initializer started")

    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = self.data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = (
            log_dir / f"portfolio_tracker_{datetime.now().strftime('%Y%m%d')}.log"
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def initialize_system(self) -> Dict[str, bool]:
        """Initialize the entire system and check components."""
        results = {}

        logging.info("Starting system initialization...")

        # Check data directories
        results["data_directories"] = self._create_data_directories()

        # Check data providers
        results["data_providers"] = self._check_data_providers()

        # Load existing portfolios
        results["portfolios_loaded"] = self._load_existing_portfolios()

        # Update portfolio prices if any portfolios exist
        results["prices_updated"] = self._update_all_portfolio_prices()

        # Update market data for portfolios
        results["market_data_updated"] = self._update_market_data()

        logging.info(f"System initialization completed: {results}")
        return results

    def _create_data_directories(self) -> bool:
        """Create necessary data directories."""
        try:
            directories = [
                self.data_dir,
                self.data_dir / "portfolios",
                self.data_dir / "snapshots",
                self.data_dir / "backups",
                self.data_dir / "exports",
                self.data_dir / "logs",
            ]

            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)

            logging.info("Data directories created successfully")
            return True

        except Exception as e:
            logging.error(f"Failed to create data directories: {e}")
            return False

    def _check_data_providers(self) -> bool:
        """Check if data providers are working."""
        try:
            provider_status = self.data_manager.get_provider_status()

            working_providers = [
                name for name, status in provider_status.items() if status
            ]

            if working_providers:
                logging.info(f"Working data providers: {working_providers}")
                return True
            else:
                logging.warning("No data providers are working")
                return False

        except Exception as e:
            logging.error(f"Failed to check data providers: {e}")
            return False

    def _load_existing_portfolios(self) -> bool:
        """Load and validate existing portfolios."""
        try:
            portfolio_ids = self.storage.list_portfolios()

            if not portfolio_ids:
                logging.info("No existing portfolios found")
                return True

            loaded_count = 0
            for portfolio_id in portfolio_ids:
                try:
                    portfolio = self.storage.load_portfolio(portfolio_id)
                    if portfolio:
                        loaded_count += 1
                        logging.debug(
                            f"Loaded portfolio: {portfolio.name} ({portfolio_id})"
                        )
                except Exception as e:
                    logging.error(f"Failed to load portfolio {portfolio_id}: {e}")

            logging.info(f"Loaded {loaded_count}/{len(portfolio_ids)} portfolios")
            return loaded_count > 0

        except Exception as e:
            logging.error(f"Failed to load existing portfolios: {e}")
            return False

    def _update_all_portfolio_prices(self) -> bool:
        """Load existing portfolio data without updating prices."""
        try:
            portfolio_ids = self.storage.list_portfolios()

            if not portfolio_ids:
                logging.info("No portfolios found to load")
                return True

            loaded_count = 0
            for portfolio_id in portfolio_ids:
                try:
                    portfolio = self.portfolio_manager.load_portfolio(portfolio_id)
                    if portfolio:
                        loaded_count += 1
                        logging.debug(
                            f"Loaded portfolio {portfolio.name} with {len(portfolio.positions)} positions"
                        )

                except Exception as e:
                    logging.error(f"Failed to load portfolio {portfolio_id}: {e}")

            logging.info(f"Loaded {loaded_count} portfolios")
            return loaded_count > 0

        except Exception as e:
            logging.error(f"Failed to load portfolios: {e}")
            return False

    def _update_market_data(self) -> bool:
        """Update market data for all portfolios."""
        try:
            portfolio_ids = self.storage.list_portfolios()

            if not portfolio_ids:
                return True

            today = date.today()
            updated_count = 0

            for portfolio_id in portfolio_ids:
                try:
                    # Load portfolio and update market data
                    portfolio = self.portfolio_manager.load_portfolio(portfolio_id)
                    if portfolio:
                        # Update market data for today
                        self.portfolio_manager.update_market_data(today, today)
                        updated_count += 1
                        logging.debug(
                            f"Updated market data for portfolio {portfolio.name}"
                        )

                except Exception as e:
                    logging.error(
                        f"Failed to update market data for portfolio {portfolio_id}: {e}"
                    )

            logging.info(f"Updated market data for {updated_count} portfolios")
            return updated_count >= 0

        except Exception as e:
            logging.error(f"Failed to update market data: {e}")
            return False

    def update_portfolio_since_last_run(self, portfolio_id: str) -> Dict[str, any]:
        """Load existing portfolio data and update market data."""
        try:
            portfolio = self.portfolio_manager.load_portfolio(portfolio_id)
            if not portfolio:
                return {"error": "Portfolio not found"}

            results = {
                "portfolio_name": portfolio.name,
                "market_data_updated": False,
            }

            # Note: Prices are not automatically updated - use UI update button instead
            logging.info(f"Loaded portfolio {portfolio.name} with existing data")

            # Update market data for recent period
            today = date.today()
            start_date = today - timedelta(days=30)
            try:
                self.portfolio_manager.update_market_data(start_date, today)
                results["market_data_updated"] = True
                logging.info(
                    f"Updated market data from {start_date} to {today}"
                )
            except Exception as e:
                logging.warning(f"Could not update market data: {e}")

            logging.info(f"Portfolio update completed for {portfolio.name}: {results}")
            return results

        except Exception as e:
            logging.error(f"Failed to update portfolio {portfolio_id}: {e}")
            return {"error": str(e)}

    def create_sample_portfolio(self, name: str = "Sample Portfolio") -> Optional[str]:
        """Create a sample portfolio with some demo data."""
        try:
            # Create portfolio
            portfolio = self.portfolio_manager.create_portfolio(name, Currency.USD)

            # Add sample cash deposit (external)
            self.portfolio_manager.deposit_cash(
                amount=100000,
                timestamp=datetime.now() - timedelta(days=60),
                notes="Initial deposit",
            )

            # Add diversified purchases across categories (days_ago staggered)
            # Short Term (T-bills ETF)
            sample_transactions = [
                ("BIL", 200, 100.00, 45),  # Short-term treasury ETF
                # Bonds
                ("TLT", 50, 90.00, 40),
                # Equities
                ("AAPL", 30, 150.00, 30),
                ("MSFT", 20, 300.00, 28),
                ("GOOGL", 10, 2500.00, 25),
                # Alternatives
                ("GLD", 15, 180.00, 22),
                ("BTC-USD", 0.5, 60000.00, 20),
            ]

            for symbol, quantity, price, days_ago in sample_transactions:
                try:
                    self.portfolio_manager.buy_shares(
                        symbol=symbol,
                        quantity=quantity,
                        price=price,
                        timestamp=datetime.now() - timedelta(days=days_ago),
                        notes=f"Sample purchase of {symbol}",
                    )
                    logging.debug(f"Added sample transaction: {quantity} {symbol}")
                except Exception as e:
                    logging.error(f"Failed to add sample transaction {symbol}: {e}")

            # Update current market prices
            logging.info("Updating current market prices for sample portfolio...")
            price_update_results = self.portfolio_manager.update_current_prices()
            logging.info(f"Price update results: {price_update_results}")

            # Update market data for the past 60 days
            logging.info("Updating historical market data for sample portfolio...")
            end_date = date.today()
            start_date = end_date - timedelta(days=60)
            self.portfolio_manager.update_market_data(start_date, end_date)
            logging.info(
                f"Updated market data for sample portfolio: {name} ({portfolio.id})"
            )
            return portfolio.id

        except Exception as e:
            logging.error(f"Failed to create sample portfolio: {e}")
            return None

    def cleanup_old_data(self, days_to_keep: int = 365) -> Dict[str, int]:
        """Clean up old data files."""
        try:
            cutoff_date = date.today() - timedelta(days=days_to_keep)
            results = {
                "old_snapshots_removed": 0,
                "old_backups_removed": 0,
                "old_logs_removed": 0,
            }

            # Clean old snapshots
            snapshots_dir = self.data_dir / "snapshots"
            if snapshots_dir.exists():
                for portfolio_dir in snapshots_dir.iterdir():
                    if portfolio_dir.is_dir():
                        for snapshot_file in portfolio_dir.glob("*.json"):
                            try:
                                snapshot_date = date.fromisoformat(snapshot_file.stem)
                                if snapshot_date < cutoff_date:
                                    snapshot_file.unlink()
                                    results["old_snapshots_removed"] += 1
                            except (ValueError, OSError):
                                continue

            # Clean old backups
            backups_dir = self.data_dir / "backups"
            if backups_dir.exists():
                for backup_file in backups_dir.rglob("*.json"):
                    try:
                        if backup_file.stat().st_mtime < cutoff_date.timestamp():
                            backup_file.unlink()
                            results["old_backups_removed"] += 1
                    except OSError:
                        continue

            # Clean old logs
            logs_dir = self.data_dir / "logs"
            if logs_dir.exists():
                for log_file in logs_dir.glob("*.log"):
                    try:
                        if log_file.stat().st_mtime < cutoff_date.timestamp():
                            log_file.unlink()
                            results["old_logs_removed"] += 1
                    except OSError:
                        continue

            logging.info(f"Cleanup completed: {results}")
            return results

        except Exception as e:
            logging.error(f"Failed to cleanup old data: {e}")
            return {"error": str(e)}

    def get_system_status(self) -> Dict[str, any]:
        """Get comprehensive system status."""
        try:
            portfolio_ids = self.storage.list_portfolios()

            status = {
                "timestamp": datetime.now().isoformat(),
                "data_providers": self.data_manager.get_provider_status(),
                "portfolios": {"total_count": len(portfolio_ids), "portfolio_list": []},
                "storage": {
                    "data_directory": str(self.data_dir),
                    "directories_exist": self.data_dir.exists(),
                },
            }

            # Get details for each portfolio
            for portfolio_id in portfolio_ids[:10]:  # Limit to first 10
                try:
                    portfolio = self.storage.load_portfolio(portfolio_id)
                    if portfolio:
                        status["portfolios"]["portfolio_list"].append(
                            {
                                "id": portfolio_id,
                                "name": portfolio.name,
                                "created": portfolio.created_at.isoformat(),
                                "positions": len(portfolio.positions),
                                "transactions": len(portfolio.transactions),
                            }
                        )
                except Exception as e:
                    logging.error(
                        f"Error getting status for portfolio {portfolio_id}: {e}"
                    )

            return status

        except Exception as e:
            logging.error(f"Failed to get system status: {e}")
            return {"error": str(e)}

    def update_portfolio_market_data(
        self, portfolio_id: str, days: int = 60
    ) -> Dict[str, any]:
        """Update market data for a specific portfolio."""
        try:
            # Load the portfolio
            portfolio = self.storage.load_portfolio(portfolio_id)
            if not portfolio:
                return {"error": f"Portfolio {portfolio_id} not found"}

            # Load portfolio into manager
            self.portfolio_manager.load_portfolio(portfolio_id)

            # Update current prices
            logging.info(f"Updating prices for portfolio {portfolio_id}")
            price_results = self.portfolio_manager.update_current_prices()

            # Update market data for the specified number of days
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            self.portfolio_manager.update_market_data(start_date, end_date)

            return {
                "success": True,
                "portfolio_id": portfolio_id,
                "portfolio_name": portfolio.name,
                "market_data_updated": True,
                "price_update_results": price_results,
                "current_value": float(self.portfolio_manager.get_portfolio_value() or 0),
            }

        except Exception as e:
            logging.error(f"Error updating market data for portfolio {portfolio_id}: {e}")
            return {"error": str(e)}
