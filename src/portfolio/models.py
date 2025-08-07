"""
Portfolio data models for tracking financial instruments and transactions.
"""

from datetime import datetime
from datetime import date as date_type
from enum import Enum
from typing import Dict, List, Optional
from decimal import Decimal

from pydantic import BaseModel, Field


class InstrumentType(str, Enum):
    """Types of financial instruments."""
    STOCK = "stock"
    BOND = "bond"
    CRYPTO = "crypto"
    CASH = "cash"
    ETF = "etf"
    MUTUAL_FUND = "mutual_fund"
    OPTION = "option"
    FUTURE = "future"


class TransactionType(str, Enum):
    """Types of portfolio transactions."""
    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"
    INTEREST = "interest"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    SPLIT = "split"
    MERGER = "merger"


class Currency(str, Enum):
    """Supported currencies."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    CHF = "CHF"
    CAD = "CAD"
    AUD = "AUD"
    BTC = "BTC"
    ETH = "ETH"


class FinancialInstrument(BaseModel):
    """Represents a financial instrument (stock, bond, crypto, etc.)."""

    symbol: str = Field(..., description="Trading symbol (e.g., AAPL, TSLA)", min_length=1, max_length=50)
    isin: Optional[str] = Field(None, description="International Securities Identification Number", max_length=12)
    name: str = Field(..., description="Full name of the instrument", min_length=1, max_length=200)
    instrument_type: InstrumentType = Field(..., description="Type of financial instrument")
    currency: Currency = Field(..., description="Trading currency")
    exchange: Optional[str] = Field(None, description="Exchange where traded", max_length=100)

    class Config:
        use_enum_values = False


class Transaction(BaseModel):
    """Represents a single portfolio transaction."""

    id: str = Field(..., description="Unique transaction identifier", min_length=1, max_length=100)
    timestamp: datetime = Field(..., description="When the transaction occurred")
    instrument: FinancialInstrument = Field(..., description="The financial instrument")
    transaction_type: TransactionType = Field(..., description="Type of transaction")
    quantity: Decimal = Field(..., description="Number of shares/units", gt=0)
    price: Decimal = Field(..., description="Price per unit", ge=0)
    fees: Decimal = Field(default=Decimal("0"), description="Transaction fees", ge=0)
    currency: Currency = Field(..., description="Transaction currency")
    notes: Optional[str] = Field(None, description="Additional notes", max_length=500)

    @property
    def total_value(self) -> Decimal:
        """Calculate total transaction value including fees."""
        base_value = self.quantity * self.price
        if self.transaction_type == TransactionType.BUY:
            return base_value + self.fees
        elif self.transaction_type == TransactionType.SELL:
            return base_value - self.fees
        return base_value

    class Config:
        use_enum_values = False


class Position(BaseModel):
    """Represents a current position in the portfolio."""

    instrument: FinancialInstrument = Field(..., description="The financial instrument")
    quantity: Decimal = Field(..., description="Current quantity held")
    average_cost: Decimal = Field(..., description="Average cost basis per unit")
    current_price: Optional[Decimal] = Field(None, description="Current market price")
    last_updated: Optional[datetime] = Field(None, description="When price was last updated")

    @property
    def market_value(self) -> Optional[Decimal]:
        """Calculate current market value of the position."""
        if self.current_price is not None:
            return self.quantity * self.current_price
        return None

    @property
    def cost_basis(self) -> Decimal:
        """Calculate total cost basis of the position."""
        return self.quantity * self.average_cost

    @property
    def unrealized_pnl(self) -> Optional[Decimal]:
        """Calculate unrealized profit/loss."""
        if self.current_price is not None:
            return self.market_value - self.cost_basis
        return None

    @property
    def unrealized_pnl_percent(self) -> Optional[Decimal]:
        """Calculate unrealized profit/loss percentage."""
        if self.current_price is not None and self.cost_basis != 0:
            return (self.unrealized_pnl / self.cost_basis) * 100
        return None


class PortfolioSnapshot(BaseModel):
    """Represents a portfolio state at a specific point in time."""

    date: date_type = Field(..., description="Date of the snapshot")
    total_value: Decimal = Field(..., description="Total portfolio value")
    cash_balance: Decimal = Field(..., description="Cash balance")
    positions_value: Decimal = Field(..., description="Total value of positions")
    base_currency: Currency = Field(..., description="Base currency for calculations")

    class Config:
        use_enum_values = False


class Portfolio(BaseModel):
    """Main portfolio class containing all positions and transactions."""

    id: str = Field(..., description="Unique portfolio identifier")
    name: str = Field(..., description="Portfolio name")
    base_currency: Currency = Field(default=Currency.USD, description="Base currency")
    created_at: datetime = Field(default_factory=datetime.now)
    transactions: List[Transaction] = Field(default_factory=list)
    positions: Dict[str, Position] = Field(default_factory=dict)  # Key: instrument symbol
    cash_balances: Dict[Currency, Decimal] = Field(default_factory=dict)

    def add_transaction(self, transaction: Transaction) -> None:
        """Add a new transaction to the portfolio."""
        self.transactions.append(transaction)
        self._update_position_from_transaction(transaction)

    def _update_position_from_transaction(self, transaction: Transaction) -> None:
        """Update position based on a new transaction."""
        symbol = transaction.instrument.symbol

        if transaction.transaction_type == TransactionType.BUY:
            if symbol in self.positions:
                # Update existing position
                pos = self.positions[symbol]
                total_cost = (pos.quantity * pos.average_cost) + transaction.total_value
                total_quantity = pos.quantity + transaction.quantity
                pos.average_cost = total_cost / total_quantity if total_quantity > 0 else Decimal("0")
                pos.quantity = total_quantity
            else:
                # Create new position
                self.positions[symbol] = Position(
                    instrument=transaction.instrument,
                    quantity=transaction.quantity,
                    average_cost=transaction.price
                )

        elif transaction.transaction_type == TransactionType.SELL:
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.quantity -= transaction.quantity
                if pos.quantity <= 0:
                    del self.positions[symbol]

        elif transaction.transaction_type in [TransactionType.DEPOSIT, TransactionType.WITHDRAWAL]:
            # Handle cash transactions
            currency = transaction.currency
            if currency not in self.cash_balances:
                self.cash_balances[currency] = Decimal("0")

            if transaction.transaction_type == TransactionType.DEPOSIT:
                self.cash_balances[currency] += transaction.total_value
            else:
                self.cash_balances[currency] -= transaction.total_value

    def get_total_value(self, exchange_rates: Optional[Dict[str, Decimal]] = None) -> Decimal:
        """Calculate total portfolio value in base currency."""
        total = Decimal("0")

        # Add cash balances
        for currency, amount in self.cash_balances.items():
            # Handle both Currency enum and string keys
            currency_code = currency.value if hasattr(currency, 'value') else currency
            base_currency_code = self.base_currency.value if hasattr(self.base_currency, 'value') else self.base_currency

            if currency_code == base_currency_code:
                total += amount
            elif exchange_rates and currency_code in exchange_rates:
                total += amount * exchange_rates[currency_code]

        # Add position values
        for position in self.positions.values():
            if position.market_value:
                position_currency_code = position.instrument.currency.value if hasattr(position.instrument.currency, 'value') else position.instrument.currency

                if position_currency_code == base_currency_code:
                    total += position.market_value
                elif exchange_rates and position_currency_code in exchange_rates:
                    total += position.market_value * exchange_rates[position_currency_code]

        return total

    def get_positions_by_type(self, instrument_type: InstrumentType) -> List[Position]:
        """Get all positions of a specific instrument type."""
        return [pos for pos in self.positions.values()
                if pos.instrument.instrument_type == instrument_type]

    class Config:
        use_enum_values = False