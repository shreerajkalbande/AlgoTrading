"""
Quantitative Trading Strategy Implementation

Implements a multi-factor momentum and mean-reversion strategy with proper
risk management and position sizing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import VolumeWeightedAveragePrice

from config.settings import TRADING_CONFIG, STRATEGY_CONFIG, RISK_CONFIG
from strategy.base import BaseStrategy, Trade, Position

class QuantMomentumStrategy(BaseStrategy):
    """Multi-factor quantitative momentum strategy."""
    
    def __init__(self, initial_capital: float = TRADING_CONFIG['initial_capital']):
        super().__init__(initial_capital)
        self.max_position_size = TRADING_CONFIG['max_position_size']
        self.transaction_cost = TRADING_CONFIG['transaction_cost']
        
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute advanced technical indicators for strategy."""
        # Momentum indicators
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        
        stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        
        # CCI for momentum confirmation
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # Trend indicators
        df['SMA20'] = df['Close'].rolling(window=STRATEGY_CONFIG['sma_short']).mean()
        df['SMA50'] = df['Close'].rolling(window=STRATEGY_CONFIG['sma_long']).mean()
        df['ADX'] = ADXIndicator(df['High'], df['Low'], df['Close']).adx()
        
        # Volatility indicators
        df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        bb = BollingerBands(df['Close'])
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        
        # Volume indicators
        df['VWAP'] = VolumeWeightedAveragePrice(
            df['High'], df['Low'], df['Close'], df['Volume'], window=20
        ).volume_weighted_average_price()
        
        # Statistical indicators
        sma_20 = df['Close'].rolling(20).mean()
        std_20 = df['Close'].rolling(20).std()
        df['Z_Score'] = (df['Close'] - sma_20) / std_20
        
        df['Volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        df['Return_5d'] = df['Close'].pct_change(5)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using advanced multi-factor approach."""
        df = self.compute_indicators(df)
        
        # Initialize signals
        df['Signal'] = 0.0
        df['Signal_Strength'] = 0.0
        
        # Multi-factor signal components
        
        # 1. Mean reversion signals
        rsi_oversold = df['RSI'] < STRATEGY_CONFIG['rsi_oversold']
        rsi_overbought = df['RSI'] > STRATEGY_CONFIG['rsi_overbought']
        bb_oversold = df['Close'] < df['BB_Lower']
        bb_overbought = df['Close'] > df['BB_Upper']
        zscore_oversold = df['Z_Score'] < -2
        zscore_overbought = df['Z_Score'] > 2
        
        # 2. Trend following signals
        trend_bullish = (df['SMA20'] > df['SMA50']) & (df['ADX'] > 25)
        trend_bearish = (df['SMA20'] < df['SMA50']) & (df['ADX'] > 25)
        
        # 3. Momentum confirmation
        stoch_oversold = df['Stoch_K'] < 20
        stoch_overbought = df['Stoch_K'] > 80
        cci_oversold = df['CCI'] < -100
        cci_overbought = df['CCI'] > 100
        
        # 4. Volume confirmation
        volume_support = df['Close'] > df['VWAP']
        
        # Combined long signals (mean reversion + trend + momentum)
        long_signal = (
            (rsi_oversold | bb_oversold | zscore_oversold) &
            trend_bullish &
            (stoch_oversold | cci_oversold) &
            volume_support
        )
        
        # Combined short signals
        short_signal = (
            (rsi_overbought | bb_overbought | zscore_overbought) &
            trend_bearish &
            (stoch_overbought | cci_overbought) &
            ~volume_support
        )
        
        # Assign signals
        df.loc[long_signal, 'Signal'] = 1.0
        df.loc[short_signal, 'Signal'] = -1.0
        
        # Calculate signal strength using multiple factors
        df['Signal_Strength'] = abs(df['Signal']) * (
            0.25 * abs(df['RSI'] - 50) / 50 +  # RSI deviation from neutral
            0.25 * abs(df['Z_Score']) / 3 +    # Z-score strength
            0.25 * df['ADX'] / 100 +           # Trend strength
            0.25 * abs(df['Return_5d'])        # Recent momentum
        )
        
        return df
    
    def calculate_position_size(self, signal: float, price: float, 
                              volatility: float) -> int:
        """Calculate position size using volatility-adjusted sizing."""
        if abs(signal) < 0.1:
            return 0
            
        # Base position size as percentage of portfolio
        base_size = self.max_position_size * abs(signal)
        
        # Volatility adjustment (reduce size for high volatility)
        vol_adjustment = min(1.0, 0.15 / max(volatility, 0.05))
        adjusted_size = base_size * vol_adjustment
        
        # Calculate number of shares
        portfolio_value = self.cash + sum(pos.market_value for pos in self.positions.values())
        position_value = portfolio_value * adjusted_size
        
        # Account for transaction costs
        effective_price = price * (1 + self.transaction_cost)
        shares = int(position_value / effective_price)
        
        return shares

def backtest_strategy(df: pd.DataFrame, strategy: QuantMomentumStrategy = None) -> Tuple[List[Dict], Dict[str, float]]:
    """Backtest the quantitative strategy with simplified logic."""
    if strategy is None:
        strategy = QuantMomentumStrategy()
    
    df = strategy.generate_signals(df)
    trade_log = []
    
    cash = strategy.initial_capital
    position = 0
    entry_price = 0
    equity_curve = []
    
    for date, row in df.iterrows():
        # Exit logic
        if position > 0:
            stop_loss_hit = row['Close'] < entry_price * (1 - RISK_CONFIG['stop_loss'])
            take_profit_hit = row['Close'] > entry_price * (1 + RISK_CONFIG['take_profit'])
            rsi_exit = row['RSI'] > STRATEGY_CONFIG['rsi_overbought']
            trend_exit = row['SMA20'] < row['SMA50']
            
            if stop_loss_hit or take_profit_hit or rsi_exit or trend_exit:
                proceeds = position * row['Close']
                pnl = position * (row['Close'] - entry_price)
                cash += proceeds
                trade_log.append({
                    'Date': date,
                    'Action': 'SELL',
                    'Price': row['Close'],
                    'Shares': position,
                    'P&L': pnl
                })
                position = 0
        
        # Entry logic
        if position == 0 and row['Signal'] > 0.5:
            shares = int((cash * strategy.max_position_size) / row['Close'])
            if shares > 0:
                cost = shares * row['Close']
                cash -= cost
                position = shares
                entry_price = row['Close']
                trade_log.append({
                    'Date': date,
                    'Action': 'BUY',
                    'Price': row['Close'],
                    'Shares': shares
                })
        
        # Track equity
        portfolio_value = cash + (position * row['Close'])
        equity_curve.append(portfolio_value)
    
    # Calculate performance
    returns = pd.Series(equity_curve).pct_change().dropna()
    total_return = (equity_curve[-1] - strategy.initial_capital) / strategy.initial_capital
    volatility = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0
    
    cumulative = pd.Series(equity_curve)
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    completed_trades = [t for t in trade_log if 'P&L' in t]
    win_rate = len([t for t in completed_trades if t['P&L'] > 0]) / len(completed_trades) if completed_trades else 0
    
    performance = {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_trades': len(completed_trades),
        'final_portfolio_value': equity_curve[-1] if equity_curve else strategy.initial_capital
    }
    
    return trade_log, performance