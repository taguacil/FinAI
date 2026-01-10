"""
Agent tools package.

Contains specialized tools for different agent types.
"""

# Re-export all tools from portfolio_tools
from ..portfolio_tools import (
    AddTransactionTool,
    AdvancedWhatIfTool,
    CalculatorTool,
    DeleteTransactionTool,
    GetCurrentPriceTool,
    GetPortfolioMetricsTool,
    GetPortfolioSummaryTool,
    GetTransactionHistoryTool,
    GetTransactionsTool,
    HypotheticalPositionTool,
    IngestPdfTool,
    ModifyTransactionTool,
    OptimizePortfolioTool,
    ResolveInstrumentTool,
    SearchCompanyTool,
    SearchInstrumentTool,
    SetMarketPriceTool,
    SimulateWhatIfTool,
)

# Re-export market data tools
from .market_data_tools import (
    GetBatchPricesTool,
    GetDataFreshnessTool,
    GetFXRateTool,
    GetPriceHistoryTool,
    RefreshDataTool,
    create_market_data_tools,
)

__all__ = [
    # Portfolio tools
    "AddTransactionTool",
    "AdvancedWhatIfTool",
    "CalculatorTool",
    "DeleteTransactionTool",
    "GetCurrentPriceTool",
    "GetPortfolioMetricsTool",
    "GetPortfolioSummaryTool",
    "GetTransactionHistoryTool",
    "GetTransactionsTool",
    "HypotheticalPositionTool",
    "IngestPdfTool",
    "ModifyTransactionTool",
    "OptimizePortfolioTool",
    "ResolveInstrumentTool",
    "SearchCompanyTool",
    "SearchInstrumentTool",
    "SetMarketPriceTool",
    "SimulateWhatIfTool",
    # Market data tools
    "GetPriceHistoryTool",
    "GetFXRateTool",
    "GetBatchPricesTool",
    "GetDataFreshnessTool",
    "RefreshDataTool",
    "create_market_data_tools",
]
