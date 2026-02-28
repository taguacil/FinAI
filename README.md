# FinAI - AI Portfolio Tracker

An AI-powered portfolio management system built with Python, LangChain, and MCP (Model Context Protocol). Features multi-agent architecture, portfolio optimization, and natural language interaction.

## Features

### Portfolio Management
- **Multi-currency support** - Track investments in USD, EUR, GBP, JPY, CHF, CAD, AUD, BTC, ETH
- **Transaction tracking** - Buy, sell, dividend, deposit, withdrawal, fees, splits, mergers
- **Position management** - Automatic cost basis calculation and P&L tracking
- **Historical snapshots** - Daily portfolio value tracking for performance analysis
- **Manual price control** - Fetch prices only when explicitly requested

### Multi-Agent AI System
- **Orchestrator Agent** - Routes queries to specialized agents
- **Transaction Agent** - Handles buy/sell/transaction parsing from natural language
- **Analytics Agent** - Portfolio analysis and insights
- **Natural language interface** - "I bought 50 shares of AAPL at $150"

### Advanced Analytics
- **Performance metrics** - Sharpe ratio, volatility, max drawdown, alpha, beta, CAGR
- **Risk analysis** - Value at Risk (VaR), Conditional VaR, Sortino ratio, downside deviation
- **Portfolio optimization** - Hierarchical Risk Parity (HRP) and Mean-Variance (Markowitz)
- **What-if scenarios** - Monte Carlo simulations and stress testing
- **Technical analysis** - Moving average crossover signals (50/200 day)
- **Benchmark comparison** - Compare against S&P 500 and custom indices

### MCP Server Integration
- **Model Context Protocol** - Exposes 30+ tools for external AI systems
- **Multiple transports** - SSE (HTTP) for Cursor, stdio for Claude Desktop
- **Full portfolio operations** - Transactions, analytics, market data, optimization

### Web Interface
- **Streamlit UI** - Clean, responsive web interface
- **Interactive charts** - Plotly-powered visualizations
- **Real-time chat** - AI conversation for portfolio management
- **Market data management** - Control price updates and snapshots

## Quick Start

### Prerequisites
- Python 3.11 or higher
- uv package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/taguacil/FinAI.git
   cd FinAI
   ```

2. **Install dependencies**
   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install project dependencies
   uv sync
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env to add your API keys
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

The application will start the web interface at `http://localhost:8501`

## Usage

### Command Line Interface

```bash
# Start the web UI (default)
python main.py

# Initialize system only
python main.py --mode init

# Create a sample portfolio
python main.py --mode sample --sample-name "My Portfolio"

# Check system status
python main.py --mode status

# Use custom data directory
python main.py --data-dir /path/to/data
```

### MCP Server

Run as an MCP server for integration with Claude Desktop or other AI tools:

```bash
# SSE transport (for Cursor)
uv run python -m src.mcp_server

# stdio transport (for Claude Desktop)
uv run python -m src.mcp_server --stdio
```

#### Claude Desktop Configuration

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "finai": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.mcp_server", "--stdio"],
      "cwd": "/path/to/FinAI"
    }
  }
}
```

### Web Interface

1. **Portfolio Management** - Create portfolios, add transactions, view positions
2. **AI Chat** - Natural language portfolio management
3. **Analytics Dashboard** - Performance metrics and visualizations
4. **Market Data** - Update prices and create historical snapshots

### Example Interactions

```
User: "I bought 100 shares of Microsoft at $300 yesterday"
AI: "I'll add that transaction for you..."

User: "What's my portfolio performance this year?"
AI: "Based on your portfolio data, here's your YTD performance..."

User: "Optimize my portfolio for maximum Sharpe ratio"
AI: "Running HRP optimization..."
```

## Architecture

```
FinAI/
├── src/
│   ├── portfolio/           # Core portfolio logic
│   │   ├── models.py        # Pydantic data models
│   │   ├── manager.py       # Portfolio operations
│   │   ├── storage.py       # File-based persistence
│   │   ├── analyzer.py      # Portfolio analysis
│   │   ├── optimizer.py     # HRP and Markowitz optimization
│   │   ├── scenarios.py     # What-if and Monte Carlo
│   │   ├── portfolio_history.py  # Historical tracking
│   │   └── market_data_store.py  # Price data management
│   │
│   ├── agents/              # Multi-agent AI system
│   │   ├── orchestrator_agent.py  # Query routing
│   │   ├── transaction_agent.py   # Transaction handling
│   │   ├── analytics_agent.py     # Analysis queries
│   │   ├── portfolio_tools.py     # LangChain tools
│   │   ├── llm_config.py          # LLM provider config
│   │   └── tools/                 # Market data tools
│   │
│   ├── data_providers/      # Financial data sources
│   │   ├── yahoo_finance.py # Yahoo Finance API
│   │   ├── manager.py       # Provider coordination
│   │   └── fx_cache.py      # FX rate caching
│   │
│   ├── services/
│   │   └── market_data_service.py  # Market data operations
│   │
│   ├── ui/
│   │   ├── streamlit_app.py    # Main web interface
│   │   └── market_data_page.py # Market data UI
│   │
│   ├── utils/
│   │   ├── metrics.py       # Financial calculations
│   │   ├── initializer.py   # System setup
│   │   ├── health_check.py  # System monitoring
│   │   └── logging_config.py
│   │
│   └── mcp_server.py        # MCP server (30+ tools)
│
├── data/                    # Portfolio data (gitignored)
├── config/                  # Configuration files
└── main.py                  # Application entry point
```

## Configuration

### Environment Variables

```bash
# LLM Providers (choose one or configure multiple)

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_key

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_key

# Google Vertex AI
GOOGLE_VERTEX_PROJECT=your-project-id
GOOGLE_VERTEX_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json

# OpenAI (fallback)
OPENAI_API_KEY=your_openai_key

# App config
DATA_DIR=custom/data/path
LOG_LEVEL=INFO
```

## Supported Assets

| Type | Description |
|------|-------------|
| Stocks | Individual company shares |
| ETFs | Exchange-traded funds |
| Mutual Funds | Mutual fund investments |
| Crypto | Bitcoin, Ethereum, etc. |
| Bonds | Government and corporate (with ISIN support) |
| Cash | Multi-currency positions |
| Options/Futures | Derivatives (limited support) |

## Financial Metrics

### Returns & Performance
- Total Return, Annualized Return, CAGR
- Time-weighted returns, YTD performance

### Risk Metrics
- Volatility, Maximum Drawdown
- Value at Risk (VaR), Conditional VaR
- Downside deviation

### Risk-Adjusted Returns
- Sharpe Ratio, Sortino Ratio
- Calmar Ratio, Information Ratio

### Benchmark Analysis
- Alpha, Beta, Correlation
- Tracking error

## MCP Tools

The MCP server exposes 30+ tools:

- **Portfolio**: `get_portfolio_summary`, `get_portfolio_snapshot`, `get_ytd_performance`
- **Transactions**: `add_transaction`, `bulk_add_transactions`, `modify_transaction`, `delete_transaction`
- **Market Data**: `get_current_price`, `get_price_history`, `refresh_data`, `set_market_price`
- **Analytics**: `get_portfolio_metrics`, `optimize_portfolio`, `advanced_what_if`
- **Instruments**: `search_instrument`, `resolve_instrument`, `check_market_data_availability`

## Development

### Running Tests

```bash
uv run pytest
```

### Code Quality

```bash
# Format code
uv run black src/

# Sort imports
uv run isort src/

# Lint
uv run flake8 src/

# Type checking
uv run mypy src/
```

## Data Storage

```
data/
├── portfolios/      # Portfolio JSON files
├── snapshots/       # Daily portfolio snapshots
├── market_data/     # Cached price data
├── backups/         # Automatic backups
└── logs/            # Application logs
```

Data is stored in human-readable JSON format with Decimal precision for financial accuracy.

## Troubleshooting

**Import Errors**
```bash
cd FinAI
uv sync
```

**UI Not Starting**
```bash
python main.py --mode status
# Check if port 8501 is available
```

**Missing Prices**
```bash
# Use the MCP tool or UI to refresh market data
```

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

**Disclaimer**: This software is for educational and informational purposes only. It is not intended as financial advice. Always consult with qualified financial professionals before making investment decisions.
