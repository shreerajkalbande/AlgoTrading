# 📈 Quantitative Trading System

A professional algorithmic trading system with multi-factor strategies, machine learning, and comprehensive risk management.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Features

- **Multi-Factor Strategy**: RSI, MACD, Moving Averages with signal strength weighting
- **Machine Learning**: Ensemble models (Random Forest, Gradient Boosting, Logistic Regression)
- **Risk Management**: Volatility-adjusted position sizing, stop-loss, take-profit
- **Performance Analytics**: Sharpe ratio, drawdown analysis, comprehensive metrics
- **Real-time Integration**: Google Sheets logging, Telegram alerts

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/AlgoTrading.git
cd AlgoTrading

# Setup environment
python -m venv trading_env
source trading_env/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Run system
python main.py
```

## 📊 Architecture

```
AlgoTrading/
├── config/           # Settings and parameters
├── strategy/         # Trading strategies
├── utils/           # Performance analysis
├── data/            # Market data
├── models/          # ML models
├── main.py          # Main execution
└── ml_model.py      # ML pipeline
```

## 🔧 Configuration

### Basic Setup
Edit `config/settings.py`:
```python
STOCK_UNIVERSE = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
TRADING_CONFIG = {
    "initial_capital": 1_000_000,
    "max_position_size": 0.2,
    "stop_loss": 0.05,
    "take_profit": 0.10
}
```

### Optional Integrations

**Telegram Alerts** - Create `config.yaml`:
```yaml
telegram:
  token: "YOUR_BOT_TOKEN"
  chat_id: "YOUR_CHAT_ID"
```

**Google Sheets** - Add `service_account.json` for trade logging

## 📈 Strategy Logic

### Entry Signals
- **Long**: RSI < 30 + SMA20 > SMA50 + MACD bullish
- **Short**: RSI > 70 + SMA20 < SMA50 + MACD bearish

### Risk Management
- **Position Size**: 20% max per position, volatility-adjusted
- **Stop Loss**: 5% maximum loss
- **Take Profit**: 10% profit target

### ML Features
- 20+ technical indicators
- Time series cross-validation
- Feature selection with statistical tests
- Model ensemble with performance tracking

## 📊 Performance Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Sharpe Ratio | > 1.0 | Risk-adjusted returns |
| Max Drawdown | < 15% | Largest decline |
| Win Rate | > 55% | Profitable trades % |

## 🛠️ Usage Examples

### Run Complete System
```bash
python main.py
```

### Individual Components
```bash
python ml_model.py      # ML pipeline only
python data_ingest.py   # Fetch data only
```

### Custom Strategy
```python
from strategy.quant_strategy import QuantMomentumStrategy

strategy = QuantMomentumStrategy(initial_capital=500000)
trades, performance = backtest_strategy(data, strategy)
```

## 📋 Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- yfinance, ta (technical analysis)
- matplotlib, seaborn (visualization)

## ⚠️ Disclaimer

**Educational purposes only.** Past performance doesn't guarantee future results. Always:
- Test with paper trading first
- Start with small positions
- Monitor risk continuously
- Understand strategies before deployment

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-strategy`)
3. Commit changes (`git commit -am 'Add new strategy'`)
4. Push to branch (`git push origin feature/new-strategy`)
5. Create Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Author**: Shreeraj Kalbande  
**Version**: 2.0  
**Last Updated**: 2024