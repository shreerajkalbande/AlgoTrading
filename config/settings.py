"""
Configuration settings for the quantitative trading system.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure directories exist
for dir_path in [DATA_DIR, LOGS_DIR, MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Trading parameters
TRADING_CONFIG = {
    "initial_capital": 1_000_000,  # ₹10 Lakh
    "max_position_size": 0.2,      # 20% per position
    "risk_free_rate": 0.06,        # 6% annual
    "transaction_cost": 0.001,     # 0.1% per trade
    "slippage": 0.0005,           # 0.05% slippage
}

# Strategy parameters
STRATEGY_CONFIG = {
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "sma_short": 20,
    "sma_long": 50,
    "lookback_period": 252,  # 1 year
    "rebalance_freq": "daily",
}

# ML model parameters
ML_CONFIG = {
    "test_size": 0.2,
    "cv_folds": 5,
    "random_state": 42,
    "feature_selection_threshold": 0.01,
}

# Universe of stocks
STOCK_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ICICIBANK.NS", "KOTAKBANK.NS", "BHARTIARTL.NS", "ITC.NS", "SBIN.NS"
]

# Risk management
RISK_CONFIG = {
    "max_drawdown": 0.15,     # 15% max drawdown
    "var_confidence": 0.05,    # 95% VaR
    "stop_loss": 0.05,        # 5% stop loss
    "take_profit": 0.10,      # 10% take profit
}