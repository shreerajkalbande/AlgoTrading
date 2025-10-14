"""
Performance analysis and risk metrics for quantitative trading.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path

from config.settings import REPORTS_DIR

class PerformanceAnalyzer:
    """Comprehensive performance analysis for trading strategies."""
    
    def __init__(self, returns: pd.Series, benchmark_returns: pd.Series = None):
        self.returns = returns
        self.benchmark_returns = benchmark_returns
        
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        # Basic return metrics
        metrics['total_return'] = (1 + self.returns).prod() - 1
        metrics['annualized_return'] = (1 + self.returns.mean()) ** 252 - 1
        metrics['volatility'] = self.returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0
        metrics['max_drawdown'] = self._calculate_max_drawdown()
        
        # Win/Loss metrics
        winning_returns = self.returns[self.returns > 0]
        losing_returns = self.returns[self.returns < 0]
        
        metrics['win_rate'] = len(winning_returns) / len(self.returns)
        metrics['avg_win'] = winning_returns.mean() if len(winning_returns) > 0 else 0
        metrics['avg_loss'] = losing_returns.mean() if len(losing_returns) > 0 else 0
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + self.returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()
    
    def generate_report(self, strategy_name: str = "Strategy") -> str:
        """Generate a comprehensive performance report."""
        metrics = self.calculate_metrics()
        
        report = f"""
=== {strategy_name} Performance Report ===

Return Metrics:
  Total Return:        {metrics['total_return']:8.2%}
  Annualized Return:   {metrics['annualized_return']:8.2%}
  Volatility:          {metrics['volatility']:8.2%}

Risk-Adjusted Metrics:
  Sharpe Ratio:        {metrics['sharpe_ratio']:8.3f}
  Max Drawdown:        {metrics['max_drawdown']:8.2%}

Win/Loss Analysis:
  Win Rate:            {metrics['win_rate']:8.2%}
  Avg Win:             {metrics['avg_win']:8.2%}
  Avg Loss:            {metrics['avg_loss']:8.2%}
"""
        return report