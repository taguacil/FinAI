"""
Instrument resolver for discovering and creating instrument information.

Extracted from PortfolioManager to keep instrument discovery logic separate
from portfolio management concerns.
"""

import logging
from typing import Dict, Optional

from ..data_providers.manager import DataProviderManager
from .models import Currency, InstrumentType


class InstrumentResolver:
    """Resolves instrument information from symbols, ISINs, and company names.

    Uses DataProviderManager for lookups and provides intelligent fallbacks
    when provider data is unavailable.
    """

    def __init__(self, data_manager: DataProviderManager):
        self.data_manager = data_manager

    def discover_instrument_info(
        self,
        symbol: Optional[str],
        isin: Optional[str],
        currency: Optional[Currency],
        notes: Optional[str],
        instrument_type: Optional[str] = None,
    ) -> Optional[Dict]:
        """Enhanced instrument information discovery with intelligent fallbacks.

        Note: ISIN is optional. If provided, use it directly. If not provided,
        work with available symbol/name information without searching for ISIN.
        Company names can be automatically converted to symbols.
        """

        # Case 1: Both symbol and ISIN provided
        if symbol and isin:
            return self._handle_symbol_and_isin(symbol, isin, currency, notes, instrument_type)

        # Case 2: Only ISIN provided (symbol will be discovered or created)
        elif isin and not symbol:
            return self._handle_isin_only(isin, currency, notes, instrument_type)

        # Case 3: Only symbol provided (work with what we have, don't search for ISIN)
        elif symbol and not isin:
            return self._handle_symbol_only(symbol, currency, notes, instrument_type)

        # Case 4: Neither provided (invalid)
        else:
            logging.error("Neither symbol nor ISIN provided")
            return None

    def _handle_symbol_and_isin(
        self, symbol: str, isin: str, currency: Optional[Currency], notes: Optional[str], instrument_type: Optional[str] = None
    ) -> Dict:
        """Handle case where both symbol and ISIN are provided."""
        normalized_symbol = symbol.strip().lstrip("$").upper()

        # Try to get comprehensive info from data providers
        instrument_info = self.data_manager.get_instrument_info(normalized_symbol)

        if instrument_info:
            # Use provider data but ensure ISIN matches
            return {
                "symbol": instrument_info.symbol,
                "name": instrument_info.name,
                "instrument_type": self._get_instrument_type(instrument_type, normalized_symbol, isin),
                "currency": currency or instrument_info.currency,
                "exchange": instrument_info.exchange,
                "isin": isin.upper(),  # Use provided ISIN
            }
        else:
            # Fallback: create basic info
            return self._create_basic_instrument_info(
                normalized_symbol, isin, currency, notes, instrument_type
            )

    def _handle_isin_only(
        self, isin: str, currency: Optional[Currency], notes: Optional[str], instrument_type: Optional[str] = None
    ) -> Dict:
        """Handle case where only ISIN is provided (symbol will be discovered or created)."""
        isin = isin.upper().strip()

        # Try to find instrument by ISIN from our known mappings
        instrument_info = self.data_manager.search_by_isin(isin)

        if instrument_info:
            # Found the instrument - use its data
            return {
                "symbol": instrument_info.symbol,
                "name": instrument_info.name,
                "instrument_type": self._get_instrument_type(instrument_type, instrument_info.symbol, isin),
                "currency": currency or instrument_info.currency,
                "exchange": instrument_info.exchange,
                "isin": isin,
            }
        else:
            # Not found in known mappings - create placeholder with intelligent defaults
            return self._create_placeholder_from_isin(isin, currency, notes, instrument_type)

    def _handle_symbol_only(
        self, symbol: str, currency: Optional[Currency], notes: Optional[str], instrument_type: Optional[str] = None
    ) -> Dict:
        """Handle case where only symbol is provided."""
        normalized_symbol = symbol.strip().lstrip("$").upper()

        # Check if this might be a company name rather than a symbol
        if self._is_likely_company_name(normalized_symbol):
            # Try to find the symbol for this company name
            found_symbol = self._find_symbol_from_company_name(normalized_symbol)
            if found_symbol:
                normalized_symbol = found_symbol
                logging.info(
                    f"Converted company name '{symbol}' to symbol '{found_symbol}'"
                )

        # Get instrument info from data providers
        instrument_info = self.data_manager.get_instrument_info(normalized_symbol)

        if instrument_info:
            # Use provider data
            return {
                "symbol": instrument_info.symbol,
                "name": instrument_info.name,
                "instrument_type": self._get_instrument_type(instrument_type, normalized_symbol, None),
                "currency": currency or instrument_info.currency,
                "exchange": instrument_info.exchange,
                "isin": instrument_info.isin,
            }
        else:
            # Create basic instrument info if not found
            return self._create_basic_instrument_info(
                normalized_symbol, None, currency, notes, instrument_type
            )

    def _is_likely_company_name(self, text: str) -> bool:
        """Check if text is likely a company name rather than a stock symbol."""
        # Common patterns that suggest company names
        company_indicators = [
            "inc",
            "corp",
            "corporation",
            "company",
            "co",
            "ltd",
            "limited",
            "plc",
            "ag",
            "sa",
            "nv",
            "holdings",
            "group",
            "technologies",
            "systems",
            "solutions",
            "services",
            "international",
            "global",
        ]

        text_lower = text.lower()

        # Check for company suffixes
        for indicator in company_indicators:
            if indicator in text_lower:
                return True

        # Check if it's all lowercase (likely company name)
        if text.islower() and len(text) > 3:
            return True

        # Check if it contains spaces (likely company name)
        if " " in text:
            return True

        # Check if it's not all uppercase (likely company name)
        if not text.isupper():
            return True

        # Check if it's a known company name that should be converted
        known_companies = [
            "apple",
            "microsoft",
            "google",
            "alphabet",
            "tesla",
            "amazon",
            "meta",
            "facebook",
            "nvidia",
            "netflix",
            "berkshire",
            "hathaway",
            "jpmorgan",
            "chase",
            "johnson",
            "visa",
            "procter",
            "gamble",
            "unitedhealth",
            "home",
            "depot",
            "mastercard",
            "disney",
            "walt",
            "paypal",
            "asml",
            "holding",
            "holdings",
        ]

        if text_lower in known_companies:
            return True

        return False

    def _find_symbol_from_company_name(self, company_name: str) -> Optional[str]:
        """Find stock symbol from company name using known mappings."""
        # Common company name to symbol mappings
        company_to_symbol = {
            "apple": "AAPL",
            "apple inc": "AAPL",
            "apple inc.": "AAPL",
            "apple computer": "AAPL",
            "microsoft": "MSFT",
            "microsoft corporation": "MSFT",
            "microsoft corp": "MSFT",
            "google": "GOOGL",
            "alphabet": "GOOGL",
            "alphabet inc": "GOOGL",
            "alphabet inc.": "GOOGL",
            "tesla": "TSLA",
            "tesla inc": "TSLA",
            "tesla inc.": "TSLA",
            "amazon": "AMZN",
            "amazon.com": "AMZN",
            "amazon.com inc": "AMZN",
            "amazon.com inc.": "AMZN",
            "meta": "META",
            "meta platforms": "META",
            "meta platforms inc": "META",
            "meta platforms inc.": "META",
            "facebook": "META",
            "facebook inc": "META",
            "facebook inc.": "META",
            "nvidia": "NVDA",
            "nvidia corporation": "NVDA",
            "nvidia corp": "NVDA",
            "netflix": "NFLX",
            "netflix inc": "NFLX",
            "netflix inc.": "NFLX",
            "berkshire hathaway": "BRK.A",
            "berkshire hathaway inc": "BRK.A",
            "berkshire hathaway inc.": "BRK.A",
            "jpmorgan": "JPM",
            "jpmorgan chase": "JPM",
            "jpmorgan chase & co": "JPM",
            "jpmorgan chase & co.": "JPM",
            "johnson & johnson": "JNJ",
            "johnson and johnson": "JNJ",
            "visa": "V",
            "visa inc": "V",
            "visa inc.": "V",
            "procter & gamble": "PG",
            "procter and gamble": "PG",
            "procter & gamble co": "PG",
            "procter & gamble co.": "PG",
            "unitedhealth": "UNH",
            "unitedhealth group": "UNH",
            "unitedhealth group inc": "UNH",
            "unitedhealth group inc.": "UNH",
            "home depot": "HD",
            "the home depot": "HD",
            "the home depot inc": "HD",
            "the home depot inc.": "HD",
            "mastercard": "MA",
            "mastercard inc": "MA",
            "mastercard inc.": "MA",
            "disney": "DIS",
            "the walt disney company": "DIS",
            "walt disney": "DIS",
            "paypal": "PYPL",
            "paypal holdings": "PYPL",
            "paypal holdings inc": "PYPL",
            "paypal holdings inc.": "PYPL",
            "asml": "ASML",
            "asml holding": "ASML",
            "asml holding nv": "ASML",
            "asml holding n.v.": "ASML",
            "asml holdings": "ASML",
            "asml holdings nv": "ASML",
            "asml holdings n.v.": "ASML",
        }

        # Try exact match first
        company_lower = company_name.lower().strip()
        if company_lower in company_to_symbol:
            return company_to_symbol[company_lower]

        # Try partial matches
        for company, symbol in company_to_symbol.items():
            if company_lower in company or company in company_lower:
                return symbol

        # If no match found, return None (will use company name as symbol)
        return None

    def _create_basic_instrument_info(
        self,
        symbol: str,
        isin: Optional[str],
        currency: Optional[Currency],
        notes: Optional[str],
        instrument_type: Optional[str] = None,
    ) -> Dict:
        """Create basic instrument info when provider data is unavailable."""
        # Use provided instrument type if available, otherwise infer from symbol
        if instrument_type:
            try:
                inferred_type = InstrumentType(instrument_type.lower())
            except ValueError:
                # If invalid instrument type provided, fall back to inference
                inferred_type = self._infer_instrument_type(symbol, isin)
        else:
            inferred_type = self._infer_instrument_type(symbol, isin)

        # Try to find a better name than the symbol
        instrument_name = self._find_instrument_name(symbol, isin, notes)

        return {
            "symbol": symbol.upper(),
            "name": instrument_name,
            "instrument_type": inferred_type,
            "currency": currency or Currency.USD,
            "exchange": None,
            "isin": isin,
        }

    def _create_placeholder_from_isin(
        self, isin: str, currency: Optional[Currency], notes: Optional[str], instrument_type: Optional[str] = None
    ) -> Dict:
        """Create placeholder instrument info when ISIN lookup fails."""
        # Use provided instrument type if available, otherwise infer from ISIN prefix
        if instrument_type:
            try:
                inferred_type = InstrumentType(instrument_type.lower())
            except ValueError:
                # If invalid instrument type provided, fall back to inference
                inferred_type = self._infer_instrument_type_from_isin(isin)
        else:
            inferred_type = self._infer_instrument_type_from_isin(isin)

        # Create a descriptive name
        if notes and len(notes) > 10:
            instrument_name = notes
        else:
            instrument_name = f"Instrument {isin}"

        # Create placeholder symbol
        if isin.upper().startswith("XS"):
            # Bonds - use ISIN prefix
            placeholder_symbol = f"BOND_{isin[:8]}"
        elif isin.upper().startswith("US"):
            # US instruments - use ISIN prefix
            placeholder_symbol = f"US_{isin[:8]}"
        elif isin.upper().startswith("CH"):
            # Swiss instruments - use ISIN prefix
            placeholder_symbol = f"CH_{isin[:8]}"
        else:
            # Generic - use ISIN prefix
            placeholder_symbol = f"ISIN_{isin[:8]}"

        return {
            "symbol": placeholder_symbol,
            "name": instrument_name,
            "instrument_type": inferred_type,
            "currency": currency or Currency.USD,
            "exchange": None,
            "isin": isin,
        }

    def _infer_instrument_type(
        self, symbol: str, isin: Optional[str]
    ) -> InstrumentType:
        """Infer instrument type from symbol and ISIN."""
        if isin:
            return self._infer_instrument_type_from_isin(isin)

        # Infer from symbol
        symbol_upper = symbol.upper()

        # Check for CASH
        if symbol_upper == "CASH":
            return InstrumentType.CASH

        # Common ETF symbols (check ETFs first to avoid conflicts)
        etf_symbols = {
            "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VEA", "VWO", "VXUS", "VT",
            "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "DBB", "DJP", "ARKK", "ARKW",
            "ARKF", "ARKG", "ARKQ", "ARKX", "SOXL", "SOXS", "TQQQ", "SQQQ", "LABU",
            "LABD", "FAS", "FAZ", "ERX", "ERY", "DPST", "KRE", "XOP", "XLE", "XLF",
            "XLK", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC"
        }
        if symbol_upper in etf_symbols:
            return InstrumentType.ETF

        # Common bond ETFs and bond-related symbols
        bond_symbols = {
            "TLT", "IEF", "TIP", "LQD", "HYG", "EMB", "SHY", "SHV", "BND", "AGG", "BIL",
            "VCIT", "VCSH", "VGIT", "VGSH", "VTIP", "VGLT", "VCLT", "VWOB", "VWITX",
            "TMF", "TMV", "TBT", "TLH", "TLO", "TENZ", "TAN", "TZA"
        }
        if symbol_upper in bond_symbols:
            return InstrumentType.BOND

        # Crypto symbols
        crypto_symbols = {"BTC", "ETH", "ADA", "DOT", "LINK", "LTC", "BCH", "XRP", "SOL", "AVAX"}
        if symbol_upper in crypto_symbols:
            return InstrumentType.CRYPTO

        # Common stock symbols (well-known companies) - check these before pattern matching
        stock_symbols = {
            "AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "META", "NVDA", "NFLX",
            "BRK.A", "BRK.B", "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL",
            "ASML", "NVO", "LLY", "PFE", "ABBV", "TMO", "AVGO", "COST", "PEP", "KO",
            "WMT", "MRK", "ABT", "ACN", "CVX", "XOM", "BAC", "WFC", "GS", "MS", "BLK"
        }
        if symbol_upper in stock_symbols:
            return InstrumentType.STOCK

        # Check symbol patterns for better inference (only for unknown symbols)
        if self._is_likely_etf_symbol(symbol_upper):
            return InstrumentType.ETF
        elif self._is_likely_bond_symbol(symbol_upper):
            return InstrumentType.BOND
        elif self._is_likely_crypto_symbol(symbol_upper):
            return InstrumentType.CRYPTO

        # Default to stock for unknown symbols
        return InstrumentType.STOCK

    def _is_likely_etf_symbol(self, symbol: str) -> bool:
        """Check if symbol is likely an ETF based on patterns."""
        if len(symbol) >= 3:
            # Common ETF suffixes
            etf_suffixes = ["ETF", "FND", "TR", "FD", "IX", "EX", "AX", "RX", "TX"]
            for suffix in etf_suffixes:
                if symbol.endswith(suffix):
                    return True

            # Sector ETFs
            sector_etfs = ["XLF", "XLK", "XLV", "XLI", "XLP", "XLY", "XLB", "XLU", "XLRE", "XLC"]
            if symbol in sector_etfs:
                return True

            # Leveraged/inverse ETFs
            leveraged_patterns = ["TQQQ", "SQQQ", "SOXL", "SOXS", "LABU", "LABD", "FAS", "FAZ"]
            if symbol in leveraged_patterns:
                return True

        return False

    def _is_likely_bond_symbol(self, symbol: str) -> bool:
        """Check if symbol is likely a bond based on patterns."""
        if len(symbol) >= 3:
            # Treasury-related (exact matches or starts with)
            treasury_patterns = ["T", "TB", "TN", "TL", "TS", "TIPS", "TBILL", "TBOND"]
            for pattern in treasury_patterns:
                if symbol.startswith(pattern) and len(pattern) >= 2:
                    return True

            # Corporate bond patterns (exact matches)
            if symbol in ["CORP", "BOND", "DEBT"]:
                return True

            # Municipal bond patterns (exact matches)
            if symbol in ["MUNI", "MUN"]:
                return True

        return False

    def _is_likely_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is likely a cryptocurrency based on patterns."""
        if 3 <= len(symbol) <= 5:
            # Common crypto prefixes/suffixes
            crypto_patterns = ["BTC", "ETH", "ADA", "DOT", "LINK", "LTC", "BCH", "XRP", "SOL", "AVAX"]
            if symbol in crypto_patterns:
                return True

            # Check for common crypto naming patterns (only for short symbols)
            if len(symbol) <= 4 and (symbol.endswith("X") or symbol.endswith("T") or symbol.endswith("N")):
                return True

        return False

    def _infer_instrument_type_from_isin(self, isin: str) -> InstrumentType:
        """Infer instrument type from ISIN prefix with enhanced logic."""
        isin_upper = isin.upper()

        # Bond ISINs
        if isin_upper.startswith("XS"):  # International bonds
            return InstrumentType.BOND
        elif isin_upper.startswith("US") and len(isin_upper) >= 12:
            # US ISINs - check for bond patterns
            if any(pattern in isin_upper for pattern in ["BOND", "DEBT", "TREAS", "CORP"]):
                return InstrumentType.BOND
            # Check if it's a known bond ISIN
            elif isin_upper in ["US4642876555", "US78464A7353", "US78464A7353"]:  # TLT, BIL examples
                return InstrumentType.BOND
            else:
                return InstrumentType.STOCK  # Default for US ISINs

        # European ISINs
        elif isin_upper.startswith("IE"):  # Ireland (ETFs)
            return InstrumentType.ETF
        elif isin_upper.startswith("LU"):  # Luxembourg (ETFs)
            return InstrumentType.ETF
        elif isin_upper.startswith("DE"):  # Germany
            return InstrumentType.STOCK
        elif isin_upper.startswith("FR"):  # France
            return InstrumentType.STOCK
        elif isin_upper.startswith("GB"):  # UK
            return InstrumentType.STOCK

        # Swiss ISINs
        elif isin_upper.startswith("CH"):
            return InstrumentType.STOCK

        # Japanese ISINs
        elif isin_upper.startswith("JP"):
            return InstrumentType.STOCK

        # Canadian ISINs
        elif isin_upper.startswith("CA"):
            return InstrumentType.STOCK

        # Australian ISINs
        elif isin_upper.startswith("AU"):
            return InstrumentType.STOCK

        # Default to stock for unknown ISIN patterns
        return InstrumentType.STOCK

    def _get_instrument_type(
        self,
        user_provided_type: Optional[str],
        symbol: Optional[str],
        isin: Optional[str]
    ) -> InstrumentType:
        """Get instrument type, prioritizing user-provided type over automatic inference."""
        if user_provided_type:
            try:
                return InstrumentType(user_provided_type.lower())
            except ValueError:
                # If invalid instrument type provided, fall back to inference
                pass

        # Fall back to automatic inference
        if symbol:
            return self._infer_instrument_type(symbol, isin)
        elif isin:
            return self._infer_instrument_type_from_isin(isin)
        else:
            return InstrumentType.STOCK

    def _find_instrument_name(
        self, symbol: str, isin: Optional[str], notes: Optional[str]
    ) -> str:
        """Find a better instrument name than just the symbol."""
        symbol_upper = symbol.upper()

        if symbol_upper == "CASH":
            return "Cash"

        # Use notes if provided and meaningful
        if notes and len(notes) > 5 and not notes.upper().startswith(symbol_upper):
            return notes

        # Common stock symbols with known names
        symbol_to_name = {
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corporation",
            "GOOGL": "Alphabet Inc.",
            "TSLA": "Tesla Inc.",
            "AMZN": "Amazon.com Inc.",
            "META": "Meta Platforms Inc.",
            "NVDA": "NVIDIA Corporation",
            "NFLX": "Netflix Inc.",
            "BRK.A": "Berkshire Hathaway Inc.",
            "BRK.B": "Berkshire Hathaway Inc.",
            "JPM": "JPMorgan Chase & Co.",
            "JNJ": "Johnson & Johnson",
            "V": "Visa Inc.",
            "PG": "Procter & Gamble Co.",
            "UNH": "UnitedHealth Group Inc.",
            "HD": "The Home Depot Inc.",
            "MA": "Mastercard Inc.",
            "DIS": "The Walt Disney Company",
            "PYPL": "PayPal Holdings Inc.",
        }

        if symbol_upper in symbol_to_name:
            return symbol_to_name[symbol_upper]

        # Common bond ETFs
        bond_symbols = {
            "TLT": "iShares 20+ Year Treasury Bond ETF",
            "IEF": "iShares 7-10 Year Treasury Bond ETF",
            "TIP": "iShares TIPS Bond ETF",
            "LQD": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
            "HYG": "iShares iBoxx $ High Yield Corporate Bond ETF",
            "EMB": "iShares J.P. Morgan USD Emerging Markets Bond ETF",
            "SHY": "iShares 1-3 Year Treasury Bond ETF",
            "SHV": "iShares Short Treasury Bond ETF",
            "BND": "Vanguard Total Bond Market ETF",
            "AGG": "iShares Core U.S. Aggregate Bond ETF",
            "BIL": "SPDR Bloomberg 1-3 Month T-Bill ETF",
        }

        if symbol_upper in bond_symbols:
            return bond_symbols[symbol_upper]

        # If no better name found, return the symbol
        return symbol_upper
