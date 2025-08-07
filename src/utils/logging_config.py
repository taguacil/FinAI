"""
Logging configuration for Portfolio Tracker.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "data/logs",
    app_name: str = "portfolio-tracker",
) -> logging.Logger:
    """
    Set up comprehensive logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Specific log file name (optional)
        log_dir: Directory for log files
        app_name: Application name for log files

    Returns:
        Configured logger instance
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Configure log filename
    if not log_file:
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = f"{app_name}_{timestamp}.log"

    log_filepath = log_path / log_file

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )

    simple_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    root_logger.handlers.clear()

    # File handler (detailed logging)
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)

    # Console handler (simple logging)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    # Configure specific loggers
    configure_module_loggers()

    # Log the configuration
    app_logger = logging.getLogger(app_name)
    app_logger.info(f"Logging configured - Level: {log_level}, File: {log_filepath}")

    return app_logger


def configure_module_loggers():
    """Configure logging for specific modules."""

    # Data providers - more verbose for debugging API issues
    logging.getLogger("src.data_providers").setLevel(logging.DEBUG)

    # Portfolio operations - important for audit trail
    logging.getLogger("src.portfolio").setLevel(logging.INFO)

    # Metrics calculations - moderate verbosity
    logging.getLogger("src.utils.metrics").setLevel(logging.INFO)

    # External libraries - reduce noise
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("alpha_vantage").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(name)


class PortfolioLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds portfolio context to log messages.
    """

    def __init__(self, logger: logging.Logger, portfolio_id: str):
        self.portfolio_id = portfolio_id
        super().__init__(logger, {"portfolio_id": portfolio_id})

    def process(self, msg, kwargs):
        return f"[Portfolio: {self.portfolio_id}] {msg}", kwargs


class TransactionLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds transaction context to log messages.
    """

    def __init__(
        self, logger: logging.Logger, transaction_id: str, portfolio_id: str = None
    ):
        self.transaction_id = transaction_id
        self.portfolio_id = portfolio_id
        context = {"transaction_id": transaction_id}
        if portfolio_id:
            context["portfolio_id"] = portfolio_id
        super().__init__(logger, context)

    def process(self, msg, kwargs):
        prefix = f"[Transaction: {self.transaction_id}"
        if self.portfolio_id:
            prefix += f", Portfolio: {self.portfolio_id}"
        prefix += "] "
        return f"{prefix}{msg}", kwargs


# Performance logging helpers
def log_performance(func):
    """Decorator to log function execution time."""
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} completed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise

    return wrapper


def log_api_call(provider: str, endpoint: str, symbol: str = None):
    """Log API call details."""
    logger = logging.getLogger("src.data_providers")
    symbol_info = f" for {symbol}" if symbol else ""
    logger.debug(f"API call to {provider} - {endpoint}{symbol_info}")


def log_portfolio_change(portfolio_id: str, change_type: str, details: str):
    """Log portfolio changes for audit trail."""
    logger = PortfolioLoggerAdapter(logging.getLogger("src.portfolio"), portfolio_id)
    logger.info(f"{change_type}: {details}")


def log_metric_calculation(metric_name: str, portfolio_id: str, result: dict):
    """Log financial metric calculations."""
    logger = PortfolioLoggerAdapter(
        logging.getLogger("src.utils.metrics"), portfolio_id
    )
    logger.info(f"Calculated {metric_name}: {result}")


# Error logging helpers
def log_data_provider_error(
    provider: str, operation: str, symbol: str, error: Exception
):
    """Log data provider errors with context."""
    logger = logging.getLogger("src.data_providers")
    logger.error(
        f"Data provider error - Provider: {provider}, Operation: {operation}, "
        f"Symbol: {symbol}, Error: {error}"
    )


def log_storage_error(operation: str, portfolio_id: str, error: Exception):
    """Log storage errors with context."""
    logger = logging.getLogger("src.portfolio.storage")
    logger.error(
        f"Storage error - Operation: {operation}, Portfolio: {portfolio_id}, "
        f"Error: {error}"
    )


def log_validation_error(model: str, field: str, value: str, error: Exception):
    """Log validation errors with context."""
    logger = logging.getLogger("src.portfolio.models")
    logger.warning(
        f"Validation error - Model: {model}, Field: {field}, "
        f"Value: {value}, Error: {error}"
    )
