"""
Shared data containers for the ML pipeline.

Each dataclass is a typed result object returned by one of the ML classes.
Using dataclasses keeps results structured and easy to pass between agents.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class TimeSeriesForecast:
    """Result from TimeSeriesForecaster."""
    forecast_values: np.ndarray
    forecast_dates: pd.DatetimeIndex
    confidence_interval_lower: np.ndarray
    confidence_interval_upper: np.ndarray
    mape: float           # Mean Absolute Percentage Error on validation set
    model_type: str       # 'ARIMA' or 'Prophet'

    @property
    def uncertainty_range(self) -> np.ndarray:
        """Width of confidence interval at each forecast step."""
        return self.confidence_interval_upper - self.confidence_interval_lower


@dataclass
class RiskMetrics:
    """Result from PortfolioRiskAnalyzer."""
    var_95: float           # Max loss at 95% confidence (daily)
    cvar_95: float          # Average loss when VaR is breached
    sharpe_ratio: float     # (Annual return - risk-free rate) / annual vol
    sortino_ratio: float    # Like Sharpe but only penalises downside vol
    max_drawdown: float     # Worst peak-to-trough loss in the period
    correlation_matrix: pd.DataFrame
    beta: Dict[str, float]  # Systematic risk per asset vs S&P 500
    volatility: float       # Annualised portfolio volatility
    skewness: float         # Return distribution skew (negative = left tail)
    kurtosis: float         # Fat-tailedness (>3 = heavier tails than normal)


@dataclass
class PortfolioAllocation:
    """Result from PortfolioOptimizer."""
    weights: Dict[str, float]       # Asset → allocation %
    expected_return: float          # Annualised expected return
    expected_volatility: float      # Annualised expected volatility
    sharpe_ratio: float
    efficient_frontier_data: Dict[str, List[float]]  # For plotting
