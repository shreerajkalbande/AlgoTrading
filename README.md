# 📈 Advanced Quantitative Trading System

A professional algorithmic trading system implementing multi-factor strategies with advanced technical indicators, machine learning, and institutional-grade risk management.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Advanced Features

- **Multi-Factor Strategy**: Core technical indicators (RSI, MACD, Bollinger Bands, Moving Averages, VWAP)
- **Machine Learning**: Ensemble models (Random Forest, Gradient Boosting, Logistic Regression)
- **Risk Management**: Volatility-adjusted position sizing, stop-loss, take-profit, drawdown controls
- **Performance Analytics**: Sharpe ratio, maximum drawdown, win rate tracking
- **Automated Pipeline**: Data ingestion, feature engineering, backtesting, and reporting

## 🔬 Technical Indicators

### Momentum Indicators
- **RSI** (Relative Strength Index) - Overbought/oversold conditions
- **MACD** (Moving Average Convergence Divergence) - Trend momentum

### Trend Analysis
- **SMA** (Simple Moving Averages) - 20 and 50 period
- **EMA** (Exponential Moving Averages) - 12 and 26 period
- **Moving Average Crossovers** for trend identification

### Volatility Measures
- **ATR** (Average True Range) - Volatility measurement
- **Bollinger Bands** - Dynamic support/resistance levels
- **Volatility-adjusted position sizing**

### Volume Analysis
- **VWAP** (Volume Weighted Average Price) - Institutional benchmark
- **OBV** (On-Balance Volume) - Volume momentum
- **Volume Ratio** - Relative volume analysis

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

## 📊 Strategy Logic

### Signal Generation
```python
# Multi-factor long signal
long_signal = (
    (RSI < 30 | Close < BB_Lower) &    # Mean reversion
    (SMA20 > SMA50) &                  # Trend confirmation
    (Close > VWAP)                     # Volume support
)
```

### Risk Management Framework
- **Position Sizing**: Volatility-adjusted Kelly criterion
- **Stop Loss**: 5% with ATR-based trailing stops
- **Take Profit**: 10% with Bollinger Band exits
- **Drawdown Control**: 15% maximum portfolio drawdown
- **Diversification**: 20% maximum per position

### Machine Learning Pipeline
- **Feature Engineering**: 18 quantitative features (RSI, MACD, Bollinger Bands, Moving Averages, ATR, VWAP, OBV)
- **Model Ensemble**: Random Forest + Gradient Boosting + Logistic Regression
- **Validation**: Time series cross-validation (TimeSeriesSplit)
- **Feature Selection**: ANOVA F-test for top 15 features

## 📈 Performance Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Sharpe Ratio | > 1.2 | Risk-adjusted returns |
| Sortino Ratio | > 1.5 | Downside risk adjustment |
| Max Drawdown | < 12% | Peak-to-trough decline |
| Win Rate | > 58% | Profitable trades % |
| Profit Factor | > 1.8 | Gross profit/loss ratio |

## 🛠️ Usage Examples

### Complete System
```bash
python main.py
```

### Individual Components
```bash
python ml_model.py          # Advanced ML pipeline
python strategy.py          # Multi-factor strategy
python data_ingest.py       # Market data fetching
```

### Custom Strategy Implementation
```python
from strategy.quant_strategy import QuantMomentumStrategy

# Initialize with custom parameters
strategy = QuantMomentumStrategy(initial_capital=1_000_000)

# Run backtest with advanced indicators
trades, performance = backtest_strategy(data, strategy)

# Analyze results
print(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
```

## 🔧 Configuration

### Strategy Parameters
```python
STRATEGY_CONFIG = {
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "sma_short": 20,
    "sma_long": 50,
    "adx_threshold": 25,        # Trend strength filter
    "bb_periods": 20,           # Bollinger Band periods
    "atr_multiplier": 2.0,      # ATR-based stops
}
```

### Risk Management
```python
RISK_CONFIG = {
    "max_position_size": 0.20,  # 20% per position
    "stop_loss": 0.05,          # 5% stop loss
    "take_profit": 0.10,        # 10% take profit
    "max_drawdown": 0.12,       # 12% max drawdown
    "volatility_target": 0.15,  # 15% annual volatility
}
```

## 📊 Advanced Analytics

The system provides institutional-grade performance analysis:

- **Risk Metrics**: VaR, CVaR, Sortino ratio, Calmar ratio
- **Attribution Analysis**: Factor contribution to returns
- **Regime Detection**: Market condition identification
- **Correlation Analysis**: Cross-asset relationships
- **Drawdown Analysis**: Duration and recovery metrics

## 📋 Requirements

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
ta>=0.10.2
yfinance>=0.2.18
matplotlib>=3.7.0
seaborn>=0.12.0
```

## ⚠️ Risk Disclaimer

This system is designed for educational and research purposes. Financial markets involve substantial risk of loss. Always:

- **Paper trade** extensively before live deployment
- **Start small** with real capital
- **Monitor performance** continuously
- **Understand** all strategies before implementation
- **Comply** with local financial regulations

## 🤝 Contributing

We welcome contributions to enhance the system:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/advanced-indicator`)
3. Implement with proper testing
4. Submit pull request with detailed description

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Author**: Shreeraj Kalbande  
**Version**: 2.1 - Advanced Quantitative Edition  
**Last Updated**: 2025

