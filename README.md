# 🤖 AI Portfolio Tracker

A comprehensive portfolio management system with AI-powered investment advice, built with Python, LangChain, and Streamlit.

## ✨ Features

### 📊 Portfolio Management
- **Multi-currency support** - Track investments in USD, EUR, GBP, JPY, CHF, CAD, AUD, BTC, ETH
- **Manual price updates** - Fetch latest prices only when explicitly requested
- **Transaction tracking** - Buy, sell, dividend, deposit, withdrawal transactions
- **Position management** - Automatic cost basis calculation and P&L tracking
- **Historical snapshots** - Daily portfolio value tracking for performance analysis

### 🤖 AI Financial Advisor
- **Natural language interface** - Chat with your AI financial advisor
- **Portfolio analysis** - Get insights on your holdings and performance
- **Investment advice** - AI-powered recommendations based on market conditions
- **Transaction parsing** - Add transactions using natural language
- **Market research** - Access to current market news and data

### 📈 Advanced Analytics
- **Performance metrics** - Sharpe ratio, volatility, max drawdown, alpha, beta
- **Risk analysis** - Value at Risk (VaR), Conditional VaR, Sortino ratio
- **Benchmark comparison** - Compare against S&P 500, Dow Jones, and custom indices
- **Interactive charts** - Portfolio performance, allocation, and returns visualization
- **Sector analysis** - Portfolio breakdown by sector and currency

### 🎨 Modern Web Interface
- **Streamlit-powered UI** - Clean, responsive web interface
- **Real-time chat** - Interactive AI conversation
- **Visual analytics** - Charts and graphs powered by Plotly
- **Portfolio dashboard** - Comprehensive overview of your investments
- **Transaction management** - Easy-to-use forms for adding transactions

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- uv package manager (automatically installed during setup)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd portfolio-tracker
   ```

2. **Install dependencies**
   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install project dependencies
   uv sync
   ```

3. **Set up environment variables** (optional)
   ```bash
   cp .env.example .env
   # Edit .env to add your API keys
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

The application will automatically:
- Initialize the system
- Create a sample portfolio if none exists
- Start the web interface at `http://localhost:8501`

## 📖 Usage

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

### Web Interface

1. **Portfolio Management**
   - Create new portfolios or load existing ones
   - Add transactions through forms or AI chat
   - View real-time positions and P&L

2. **Data Management**
   - **Manual Price Updates**: Click "Update Current Prices" in the sidebar to fetch latest market data
   - **Snapshot Creation**: Use "Create Snapshots Since Last" to generate historical snapshots
   - **Data Freshness**: The UI shows when prices were last updated and warns about stale data
   - **API Control**: This approach helps control API usage and ensures you only get fresh data when needed

3. **AI Chat**
   - Natural language portfolio management
   - Investment advice and market insights
   - Transaction parsing: "I bought 50 shares of AAPL at $150"

4. **Analytics Dashboard**
   - Performance metrics and risk analysis
   - Interactive charts and visualizations
   - Benchmark comparisons

### Example Interactions

**Adding transactions via chat:**
```
User: "I bought 100 shares of Microsoft at $300 yesterday"
AI: "I'll add that transaction for you..."

User: "What's my portfolio performance this year?"
AI: "Based on your portfolio data, here's your performance analysis..."

User: "Should I invest in Tesla?"
AI: "Let me analyze Tesla and your current portfolio allocation..."
```

## 🏗️ Architecture

```
portfolio-tracker/
├── src/
│   ├── portfolio/          # Portfolio models and management
│   ├── data_providers/     # Financial data APIs
│   ├── agents/            # AI agent and tools
│   ├── ui/                # Streamlit web interface
│   └── utils/             # Utilities and metrics
├── data/                  # Portfolio data storage
├── config/               # Configuration files
└── main.py              # Application entry point
```

### Core Components

- **Portfolio Manager** - Handles transactions and position tracking
- **Data Providers** - Yahoo Finance and Alpha Vantage integration
- **AI Agent** - LangChain-powered financial advisor
- **Metrics Calculator** - Advanced financial analysis
- **Storage System** - JSON-based file storage
- **Web UI** - Streamlit interface with Plotly charts

## 🔑 API Keys

The system works without API keys but with limited functionality:

### Required for AI Features
- **OpenAI API Key** - Enables AI chat functionality
  - Get from: https://platform.openai.com/api-keys
  - Set as: `OPENAI_API_KEY=your_key_here`



## 📊 Supported Assets

### Instrument Types
- **Stocks** - Individual company shares
- **ETFs** - Exchange-traded funds
- **Mutual Funds** - Mutual fund investments
- **Cryptocurrencies** - Bitcoin, Ethereum, etc.
- **Cash** - Multi-currency cash positions
- **Bonds** - Government and corporate bonds (limited support)

### Data Sources
- **Yahoo Finance** - Free real-time and historical data
- **Extensible** - Easy to add new data providers

## 🧮 Financial Metrics

### Returns & Performance
- Total Return
- Annualized Return
- Compound Annual Growth Rate (CAGR)
- Time-weighted returns

### Risk Metrics
- Volatility (standard deviation)
- Maximum Drawdown
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Downside deviation

### Risk-Adjusted Returns
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Information Ratio

### Benchmark Analysis
- Alpha (excess return)
- Beta (market sensitivity)
- Correlation with indices
- Tracking error

## 🎯 Examples

### Sample Portfolio Creation
```python
from src.utils.initializer import PortfolioInitializer

# Initialize system
initializer = PortfolioInitializer()

# Create sample portfolio with demo data
portfolio_id = initializer.create_sample_portfolio("My Portfolio")
```

### Adding Transactions
```python
from src.portfolio.manager import PortfolioManager
from decimal import Decimal

manager = PortfolioManager()
manager.load_portfolio(portfolio_id)

# Buy stocks
manager.buy_shares("AAPL", Decimal("50"), Decimal("150.00"))

# Add dividend
manager.add_dividend("AAPL", Decimal("25.00"))

# Deposit cash
manager.deposit_cash(Decimal("5000.00"))
```

### Performance Analysis
```python
from src.utils.metrics import FinancialMetricsCalculator

calculator = FinancialMetricsCalculator()
snapshots = storage.load_snapshots(portfolio_id)

# Calculate comprehensive metrics
metrics = calculator.calculate_portfolio_metrics(
    snapshots,
    benchmark_symbol="SPY"
)

print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
print(f"Alpha: {metrics['alpha']*100:.2f}%")
```

## 🔧 Configuration

### Environment Variables
```bash
# AI Providers (choose one or configure multiple and switch in the UI)

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://kallamai.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_key
# Optional: override API version (defaults to 2025-01-01-preview in code)
# AZURE_OPENAI_API_VERSION=2025-01-01-preview

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_key

# Google Vertex AI
GOOGLE_VERTEX_PROJECT=mystic-fountain-415918
GOOGLE_VERTEX_LOCATION=us-central1
# Service account JSON file for ADC
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service_account.json

# OpenAI (fallback only)
OPENAI_API_KEY=your_openai_key

# Financial data providers
# Add additional providers as needed

# App config
DATA_DIR=custom/data/path
LOG_LEVEL=INFO
```

### Streamlit Configuration
Edit `config/streamlit_config.toml` to customize the web interface:
- Port and server settings
- Theme colors and styling
- Browser behavior

## 📁 Data Storage

The system uses a file-based storage system:

```
data/
├── portfolios/           # Portfolio data files
├── snapshots/           # Daily portfolio snapshots
├── backups/            # Automatic backups
├── exports/            # Data exports
└── logs/              # Application logs
```

### Data Format
- **JSON-based** - Human-readable portfolio data
- **Decimal precision** - Accurate financial calculations
- **Automatic backups** - Data safety and recovery
- **Export options** - CSV and JSON export formats

## 🧪 Development

### Project Structure
```
src/
├── portfolio/
│   ├── models.py          # Data models
│   ├── manager.py         # Portfolio operations
│   └── storage.py         # File storage system
├── data_providers/
│   ├── base.py           # Provider interface
│   ├── yahoo_finance.py  # Yahoo Finance API
│   └── manager.py        # Provider coordination
├── agents/
│   ├── tools.py          # LangChain tools
│   └── portfolio_agent.py # AI agent
├── ui/
│   └── streamlit_app.py  # Web interface
└── utils/
    ├── metrics.py        # Financial calculations
    └── initializer.py    # System initialization
```

### Adding New Features

1. **New Data Provider**
   ```python
   # Inherit from BaseDataProvider
   class NewProvider(BaseDataProvider):
       def get_current_price(self, symbol):
           # Implementation
   ```

2. **New Financial Metric**
   ```python
   # Add to FinancialMetricsCalculator
   def calculate_new_metric(self, returns):
       # Implementation
   ```

3. **New AI Tool**
   ```python
   # Create LangChain tool
   class NewTool(BaseTool):
       name = "new_tool"
       description = "Description"
       # Implementation
   ```

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure you're in the project directory
cd portfolio-tracker
# Reinstall dependencies
uv sync
```

**UI Not Starting**
```bash
# Check if port 8501 is available
python main.py --mode status
# Try different port in streamlit_config.toml
```

**Data Provider Issues**
```bash
# Check provider status
python main.py --mode status
# Verify API keys in .env file
```

**Missing Prices**
```bash
# Update prices manually
python -c "
from src.utils.initializer import PortfolioInitializer
init = PortfolioInitializer()
init._update_all_portfolio_prices()
"
```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the logs in `data/logs/`

---

**Disclaimer**: This software is for educational and informational purposes only. It is not intended as financial advice. Always consult with qualified financial professionals before making investment decisions.
