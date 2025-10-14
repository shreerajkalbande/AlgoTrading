"""
Base classes for trading strategies.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: datetime
    exit_date: Optional[datetime]
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    quantity: int
    pnl: Optional[float] = None
    
    @property
    def is_open(self) -> bool:
        return self.exit_date is None
    
    def close_trade(self, exit_date: datetime, exit_price: float) -> None:
        """Close the trade and calculate P&L."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        
        if self.side == 'long':
            self.pnl = (exit_price - self.entry_price) * self.quantity
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.quantity

@dataclass
class Position:
    """Represents a position in a security."""
    symbol: str
    quantity: int
    avg_price: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    
    def update_market_value(self, current_price: float) -> None:
        """Update market value and unrealized P&L."""
        self.market_value = self.quantity * current_price
        self.unrealized_pnl = (current_price - self.avg_price) * self.quantity

class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, initial_capital: float = 1_000_000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals for the given data."""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: float, price: float, 
                              volatility: float) -> int:
        """Calculate position size based on signal strength and risk management."""
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        if len(self.equity_curve) < 2:
            return {}
            
        returns = pd.Series(self.equity_curve).pct_change().dropna()
        
        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (np.array(self.equity_curve) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Win rate
        completed_trades = [t for t in self.trades if t.pnl is not None]
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(completed_trades),
            'final_portfolio_value': self.equity_curve[-1] if self.equity_curve else self.initial_capital
        }