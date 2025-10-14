"""
Quantitative Trading Strategy Implementation

Implements a multi-factor momentum and mean-reversion strategy with proper
risk management and position sizing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange

from config.settings import TRADING_CONFIG, STRATEGY_CONFIG, RISK_CONFIG
from strategy.base import BaseStrategy, Trade, Position

class QuantMomentumStrategy(BaseStrategy):
    """Multi-factor quantitative momentum strategy."""
    
    def __init__(self, initial_capital: float = TRADING_CONFIG['initial_capital']):
        super().__init__(initial_capital)
        self.max_position_size = TRADING_CONFIG['max_position_size']
        self.transaction_cost = TRADING_CONFIG['transaction_cost']
        
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators for strategy."""
        # Momentum indicators
        df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
        
        # Trend indicators
        df['SMA20'] = df['Close'].rolling(window=STRATEGY_CONFIG['sma_short']).mean()
        df['SMA50'] = df['Close'].rolling(window=STRATEGY_CONFIG['sma_long']).mean()
        df['EMA12'] = EMAIndicator(df['Close'], window=12).ema_indicator()
        df['EMA26'] = EMAIndicator(df['Close'], window=26).ema_indicator()
        
        # MACD
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Volatility
        df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['Volatility'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
        
        # Price momentum
        df['Return_5d'] = df['Close'].pct_change(5)
        df['Return_20d'] = df['Close'].pct_change(20)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on multiple factors."""
        df = self.compute_indicators(df)
        
        # Initialize signals
        df['Signal'] = 0.0
        df['Signal_Strength'] = 0.0
        
        # Mean reversion component (RSI)
        rsi_oversold = df['RSI'] < STRATEGY_CONFIG['rsi_oversold']
        rsi_overbought = df['RSI'] > STRATEGY_CONFIG['rsi_overbought']
        
        # Trend following component
        trend_bullish = df['SMA20'] > df['SMA50']
        trend_bearish = df['SMA20'] < df['SMA50']
        
        # Momentum component
        momentum_positive = (df['MACD'] > df['MACD_Signal']) & (df['MACD_Histogram'] > 0)
        momentum_negative = (df['MACD'] < df['MACD_Signal']) & (df['MACD_Histogram'] < 0)
        
        # Combined signals
        long_signal = rsi_oversold & trend_bullish & momentum_positive
        short_signal = rsi_overbought & trend_bearish & momentum_negative
        
        # Signal strength based on multiple factors
        df.loc[long_signal, 'Signal'] = 1.0
        df.loc[short_signal, 'Signal'] = -1.0
        
        # Calculate signal strength (0-1)
        df['Signal_Strength'] = abs(df['Signal']) * (
            0.3 * (100 - df['RSI']) / 100 +  # RSI component
            0.3 * abs(df['MACD_Histogram']) / df['ATR'] +  # MACD strength
            0.4 * abs(df['Return_5d'])  # Recent momentum
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
    """Backtest the quantitative strategy with proper risk management."""
    if strategy is None:
        strategy = QuantMomentumStrategy()
    
    df = strategy.generate_signals(df)
    trade_log = []
    
    for date, row in df.iterrows():
        current_prices = {'symbol': row['Close']}
        
        # Update portfolio value
        portfolio_value = strategy.update_portfolio_value(current_prices)
        
        # Check for exit signals on existing positions
        positions_to_close = []
        for symbol, pos in strategy.positions.items():
            # Exit conditions
            stop_loss_hit = (row['Close'] < pos.avg_price * (1 - RISK_CONFIG['stop_loss']))
            take_profit_hit = (row['Close'] > pos.avg_price * (1 + RISK_CONFIG['take_profit']))
            rsi_exit = (pos.quantity > 0 and row['RSI'] > STRATEGY_CONFIG['rsi_overbought']) or \
                      (pos.quantity < 0 and row['RSI'] < STRATEGY_CONFIG['rsi_oversold'])
            trend_exit = (pos.quantity > 0 and row['SMA20'] < row['SMA50']) or \
                        (pos.quantity < 0 and row['SMA20'] > row['SMA50'])
            
            if stop_loss_hit or take_profit_hit or rsi_exit or trend_exit:
                positions_to_close.append(symbol)
        
        # Close positions
        for symbol in positions_to_close:
            pos = strategy.positions[symbol]
            # Find corresponding open trade and close it
            for trade in reversed(strategy.trades):
                if trade.symbol == symbol and trade.is_open:
                    trade.close_trade(date, row['Close'])
                    trade_log.append({
                        'Date': date,
                        'Action': 'SELL' if trade.side == 'long' else 'COVER',
                        'Price': row['Close'],
                        'Shares': abs(trade.quantity),
                        'P&L': trade.pnl
                    })
                    break
            
            # Update cash
            strategy.cash += pos.quantity * row['Close']
            del strategy.positions[symbol]
        
        # Generate new entry signals
        if abs(row['Signal']) > 0.1:
            trade = strategy.execute_trade(
                date=date,
                symbol='symbol',
                signal=row['Signal'],
                price=row['Close'],
                volatility=row.get('Volatility', 0.2)
            )
            
            if trade:
                trade_log.append({
                    'Date': date,
                    'Action': 'BUY' if trade.side == 'long' else 'SHORT',
                    'Price': trade.entry_price,
                    'Shares': trade.quantity
                })
    
    # Calculate performance metrics
    performance = strategy.get_performance_metrics()
    
    return trade_log, performance