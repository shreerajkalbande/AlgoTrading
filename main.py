"""
Main execution script for the Quantitative Trading System.
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.settings import DATA_DIR, STOCK_UNIVERSE, TRADING_CONFIG
from data_ingest import fetch_data
from strategy.quant_strategy import QuantMomentumStrategy, backtest_strategy
from utils.performance import PerformanceAnalyzer
from ml_model import main as run_ml_pipeline

def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Load and clean stock data from CSV."""
    df = pd.read_csv(csv_path)
    df = df[df['Date'].notna()].copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['Close', 'Volume'], inplace=True)
    
    return df

def run_strategy_backtest():
    """Run comprehensive strategy backtesting."""
    print("=== Quantitative Trading System ===")
    print("Running strategy backtests...\n")
    
    results = {}
    all_returns = []
    
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith('.csv'):
            continue
            
        symbol = fname.replace('.csv', '').replace('_', '.')
        print(f"Backtesting {symbol}...")
        
        # Load data
        df = load_and_clean_data(DATA_DIR / fname)
        
        # Run backtest
        strategy = QuantMomentumStrategy()
        trades, performance = backtest_strategy(df, strategy)
        
        # Calculate returns for performance analysis
        equity_curve = pd.Series(strategy.equity_curve, index=df.index[:len(strategy.equity_curve)])
        returns = equity_curve.pct_change().dropna()
        
        # Store results
        results[symbol] = {
            'trades': trades,
            'performance': performance,
            'returns': returns,
            'final_value': equity_curve.iloc[-1] if len(equity_curve) > 0 else TRADING_CONFIG['initial_capital']
        }
        
        all_returns.extend(returns.tolist())
        
        # Print summary
        print(f"  Final Portfolio Value: ₹{results[symbol]['final_value']:,.0f}")
        print(f"  Total Return: {performance.get('total_return', 0):.2%}")
        print(f"  Sharpe Ratio: {performance.get('sharpe_ratio', 0):.3f}")
        print(f"  Max Drawdown: {performance.get('max_drawdown', 0):.2%}")
        print(f"  Total Trades: {performance.get('total_trades', 0)}\n")
    
    # Portfolio-level analysis
    if all_returns:
        portfolio_returns = pd.Series(all_returns)
        analyzer = PerformanceAnalyzer(portfolio_returns)
        
        print("=== Portfolio Summary ===")
        print(analyzer.generate_report("Portfolio"))
        
        # Calculate total portfolio value
        total_value = sum(r['final_value'] for r in results.values())
        total_invested = TRADING_CONFIG['initial_capital'] * len(results)
        portfolio_return = (total_value - total_invested) / total_invested
        
        print(f"Total Portfolio Value: ₹{total_value:,.0f}")
        print(f"Total Invested: ₹{total_invested:,.0f}")
        print(f"Portfolio Return: {portfolio_return:.2%}")
    
    return results

def main():
    """Main execution function."""
    print("Quantitative Trading System")
    print("=" * 50)
    
    # 1. Data ingestion
    print("\n1. Fetching latest market data...")
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    try:
        fetch_data(tickers, period="1y", save_csv=True)
        print("✓ Data fetch completed")
    except Exception as e:
        print(f"⚠ Data fetch failed: {e}")
        print("Using existing data...")
    
    # 2. Strategy backtesting
    print("\n2. Running strategy backtests...")
    strategy_results = run_strategy_backtest()
    
    # 3. ML model training (optional)
    print("\n3. Training ML models...")
    try:
        run_ml_pipeline()
        print("✓ ML pipeline completed")
    except Exception as e:
        print(f"⚠ ML pipeline failed: {e}")
    
    print("\n" + "=" * 50)
    print("System execution completed!")
    print(f"Results for {len(strategy_results)} symbols processed")
    
    return strategy_results

if __name__ == "__main__":
    results = main()