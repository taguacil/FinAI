"""
Initialization system for the Portfolio Tracker application.
"""

import os
import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple
from pathlib import Path

from ..portfolio.manager import PortfolioManager
from ..portfolio.storage import FileBasedStorage
from ..data_providers.manager import DataProviderManager
from ..utils.metrics import FinancialMetricsCalculator
from ..portfolio.models import Currency


class PortfolioInitializer:
    """Handles application initialization and portfolio updates."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the portfolio initializer."""
        self.data_dir = Path(data_dir)
        self.setup_logging()
        
        # Initialize components
        self.storage = FileBasedStorage(data_dir)
        self.data_manager = DataProviderManager()
        self.portfolio_manager = PortfolioManager(self.storage, self.data_manager)
        self.metrics_calculator = FinancialMetricsCalculator(self.data_manager)
        
        logging.info("Portfolio initializer started")
    
    def setup_logging(self):
        """Set up logging configuration."""
        log_dir = self.data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"portfolio_tracker_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def initialize_system(self) -> Dict[str, bool]:
        """Initialize the entire system and check components."""
        results = {}
        
        logging.info("Starting system initialization...")
        
        # Check data directories
        results['data_directories'] = self._create_data_directories()
        
        # Check data providers
        results['data_providers'] = self._check_data_providers()
        
        # Load existing portfolios
        results['portfolios_loaded'] = self._load_existing_portfolios()
        
        # Update portfolio prices if any portfolios exist
        results['prices_updated'] = self._update_all_portfolio_prices()
        
        # Create snapshots for today
        results['snapshots_created'] = self._create_daily_snapshots()
        
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
                self.data_dir / "logs"
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
            
            working_providers = [name for name, status in provider_status.items() if status]
            
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
                        logging.debug(f"Loaded portfolio: {portfolio.name} ({portfolio_id})")
                except Exception as e:
                    logging.error(f"Failed to load portfolio {portfolio_id}: {e}")
            
            logging.info(f"Loaded {loaded_count}/{len(portfolio_ids)} portfolios")
            return loaded_count > 0
            
        except Exception as e:
            logging.error(f"Failed to load existing portfolios: {e}")
            return False
    
    def _update_all_portfolio_prices(self) -> bool:
        """Update prices for all portfolios."""
        try:
            portfolio_ids = self.storage.list_portfolios()
            
            if not portfolio_ids:
                return True
            
            updated_count = 0
            for portfolio_id in portfolio_ids:
                try:
                    portfolio = self.portfolio_manager.load_portfolio(portfolio_id)
                    if portfolio and portfolio.positions:
                        results = self.portfolio_manager.update_current_prices()
                        success_count = sum(results.values()) if results else 0
                        
                        if success_count > 0:
                            updated_count += 1
                            logging.debug(f"Updated {success_count} prices for portfolio {portfolio.name}")
                
                except Exception as e:
                    logging.error(f"Failed to update prices for portfolio {portfolio_id}: {e}")
            
            logging.info(f"Updated prices for {updated_count} portfolios")
            return updated_count > 0
            
        except Exception as e:
            logging.error(f"Failed to update portfolio prices: {e}")
            return False
    
    def _create_daily_snapshots(self) -> bool:
        """Create daily snapshots for all portfolios."""
        try:
            portfolio_ids = self.storage.list_portfolios()
            
            if not portfolio_ids:
                return True
            
            today = date.today()
            created_count = 0
            
            for portfolio_id in portfolio_ids:
                try:
                    # Check if snapshot already exists for today
                    existing_snapshots = self.storage.load_snapshots(portfolio_id, today, today)
                    
                    if existing_snapshots:
                        logging.debug(f"Snapshot already exists for portfolio {portfolio_id} on {today}")
                        continue
                    
                    # Load portfolio and create snapshot
                    portfolio = self.portfolio_manager.load_portfolio(portfolio_id)
                    if portfolio:
                        snapshot = self.portfolio_manager.create_snapshot(today)
                        created_count += 1
                        logging.debug(f"Created snapshot for portfolio {portfolio.name}")
                
                except Exception as e:
                    logging.error(f"Failed to create snapshot for portfolio {portfolio_id}: {e}")
            
            logging.info(f"Created {created_count} daily snapshots")
            return created_count >= 0  # Return True even if no snapshots needed
            
        except Exception as e:
            logging.error(f"Failed to create daily snapshots: {e}")
            return False
    
    def update_portfolio_since_last_run(self, portfolio_id: str) -> Dict[str, any]:
        """Update a specific portfolio since the last run."""
        try:
            portfolio = self.portfolio_manager.load_portfolio(portfolio_id)
            if not portfolio:
                return {'error': 'Portfolio not found'}
            
            results = {
                'portfolio_name': portfolio.name,
                'prices_updated': False,
                'snapshots_created': 0,
                'last_snapshot_date': None
            }
            
            # Update current prices
            if portfolio.positions:
                price_results = self.portfolio_manager.update_current_prices()
                results['prices_updated'] = any(price_results.values()) if price_results else False
                logging.info(f"Updated prices for portfolio {portfolio.name}")
            
            # Find the last snapshot date
            snapshots = self.storage.load_snapshots(portfolio_id)
            last_snapshot_date = snapshots[-1].date if snapshots else portfolio.created_at.date()
            results['last_snapshot_date'] = last_snapshot_date
            
            # Create snapshots for missing days
            today = date.today()
            current_date = last_snapshot_date + timedelta(days=1)
            
            while current_date <= today:
                try:
                    snapshot = self.portfolio_manager.create_snapshot(current_date)
                    results['snapshots_created'] += 1
                    logging.debug(f"Created snapshot for {current_date}")
                except Exception as e:
                    logging.error(f"Failed to create snapshot for {current_date}: {e}")
                
                current_date += timedelta(days=1)
            
            logging.info(f"Portfolio update completed for {portfolio.name}: {results}")
            return results
            
        except Exception as e:
            logging.error(f"Failed to update portfolio {portfolio_id}: {e}")
            return {'error': str(e)}
    
    def create_sample_portfolio(self, name: str = "Sample Portfolio") -> Optional[str]:
        """Create a sample portfolio with some demo data."""
        try:
            # Create portfolio
            portfolio = self.portfolio_manager.create_portfolio(name, Currency.USD)
            
            # Add sample cash deposit
            self.portfolio_manager.deposit_cash(
                amount=10000,
                timestamp=datetime.now() - timedelta(days=30),
                notes="Initial deposit"
            )
            
            # Add sample stock purchases
            sample_transactions = [
                ("AAPL", 50, 150.00, 25),
                ("GOOGL", 10, 2500.00, 20),
                ("MSFT", 30, 300.00, 15),
                ("TSLA", 15, 200.00, 10),
                ("SPY", 20, 400.00, 5)
            ]
            
            for symbol, quantity, price, days_ago in sample_transactions:
                try:
                    self.portfolio_manager.buy_shares(
                        symbol=symbol,
                        quantity=quantity,
                        price=price,
                        timestamp=datetime.now() - timedelta(days=days_ago),
                        notes=f"Sample purchase of {symbol}"
                    )
                    logging.debug(f"Added sample transaction: {quantity} {symbol}")
                except Exception as e:
                    logging.error(f"Failed to add sample transaction {symbol}: {e}")
            
            # Update current prices
            self.portfolio_manager.update_current_prices()
            
            # Create snapshots for the past 30 days
            for days_back in range(30, -1, -1):
                snapshot_date = date.today() - timedelta(days=days_back)
                try:
                    self.portfolio_manager.create_snapshot(snapshot_date)
                except Exception as e:
                    logging.error(f"Failed to create sample snapshot for {snapshot_date}: {e}")
            
            logging.info(f"Created sample portfolio: {name} ({portfolio.id})")
            return portfolio.id
            
        except Exception as e:
            logging.error(f"Failed to create sample portfolio: {e}")
            return None
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> Dict[str, int]:
        """Clean up old data files."""
        try:
            cutoff_date = date.today() - timedelta(days=days_to_keep)
            results = {
                'old_snapshots_removed': 0,
                'old_backups_removed': 0,
                'old_logs_removed': 0
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
                                    results['old_snapshots_removed'] += 1
                            except (ValueError, OSError):
                                continue
            
            # Clean old backups
            backups_dir = self.data_dir / "backups"
            if backups_dir.exists():
                for backup_file in backups_dir.rglob("*.json"):
                    try:
                        if backup_file.stat().st_mtime < cutoff_date.timestamp():
                            backup_file.unlink()
                            results['old_backups_removed'] += 1
                    except OSError:
                        continue
            
            # Clean old logs
            logs_dir = self.data_dir / "logs"
            if logs_dir.exists():
                for log_file in logs_dir.glob("*.log"):
                    try:
                        if log_file.stat().st_mtime < cutoff_date.timestamp():
                            log_file.unlink()
                            results['old_logs_removed'] += 1
                    except OSError:
                        continue
            
            logging.info(f"Cleanup completed: {results}")
            return results
            
        except Exception as e:
            logging.error(f"Failed to cleanup old data: {e}")
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict[str, any]:
        """Get comprehensive system status."""
        try:
            portfolio_ids = self.storage.list_portfolios()
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'data_providers': self.data_manager.get_provider_status(),
                'portfolios': {
                    'total_count': len(portfolio_ids),
                    'portfolio_list': []
                },
                'storage': {
                    'data_directory': str(self.data_dir),
                    'directories_exist': self.data_dir.exists()
                }
            }
            
            # Get details for each portfolio
            for portfolio_id in portfolio_ids[:10]:  # Limit to first 10
                try:
                    portfolio = self.storage.load_portfolio(portfolio_id)
                    if portfolio:
                        snapshots = self.storage.load_snapshots(portfolio_id)
                        status['portfolios']['portfolio_list'].append({
                            'id': portfolio_id,
                            'name': portfolio.name,
                            'created': portfolio.created_at.isoformat(),
                            'positions': len(portfolio.positions),
                            'snapshots': len(snapshots),
                            'last_snapshot': snapshots[-1].date.isoformat() if snapshots else None
                        })
                except Exception as e:
                    logging.error(f"Error getting status for portfolio {portfolio_id}: {e}")
            
            return status
            
        except Exception as e:
            logging.error(f"Failed to get system status: {e}")
            return {'error': str(e)}