"""
Markowitz Modern Portfolio Theory optimization.

Finds the portfolio allocation that:
  - Maximises Sharpe ratio (best risk-adjusted return), or
  - Minimises volatility (safest portfolio)
  - Generates the efficient frontier (all optimal risk/return tradeoffs)

The core idea: diversification reduces risk without reducing expected return.
The math: treat allocation weights as variables, optimise via scipy.
"""

import logging
import numpy as np
import pandas as pd
from typing import List
from datetime import datetime, timedelta

import yfinance as yf
from scipy.optimize import minimize

from .models import PortfolioAllocation

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Find optimal asset weights using Markowitz mean-variance framework.

    Typical usage:
        optimizer = PortfolioOptimizer(['AAPL', 'MSFT', 'JPM', 'JNJ'])
        optimizer.fetch_returns()
        result = optimizer.optimize_for_max_sharpe()
        print(result.weights)   # {'AAPL': 0.31, 'MSFT': 0.69, ...}
    """

    def __init__(self, symbols: List[str], lookback_days: int = 252):
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.returns_df: pd.DataFrame = None
        self.expected_returns: pd.Series = None   # Annualised mean returns
        self.cov_matrix: pd.DataFrame = None      # Annualised covariance matrix

    def fetch_returns(self) -> pd.DataFrame:
        """Fetch prices, compute daily returns, and pre-calculate statistics."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)

        prices = yf.download(self.symbols, start=start_date, end=end_date, progress=False)['Close']
        self.returns_df = prices.pct_change().dropna()

        # Annualise (×252 trading days) for use in optimisation objective
        self.expected_returns = self.returns_df.mean() * 252
        self.cov_matrix = self.returns_df.cov() * 252
        return self.returns_df

    def optimize_for_max_sharpe(self, risk_free_rate: float = 0.04) -> PortfolioAllocation:
        """
        Find weights that maximise the Sharpe ratio.

        Constraints:
            - Weights sum to 1.0 (fully invested)
            - No short selling (each weight ∈ [0, 1])

        The optimiser minimises *negative* Sharpe (scipy only minimises).
        Starting point: equal weights. SLSQP handles the constraints.
        """
        if self.returns_df is None:
            self.fetch_returns()

        n = len(self.symbols)

        def negative_sharpe(w):
            ret = np.dot(w, self.expected_returns)
            vol = np.sqrt(np.dot(w, np.dot(self.cov_matrix, w)))
            return -(ret - risk_free_rate) / vol

        result = minimize(
            negative_sharpe,
            x0=np.full(n, 1 / n),
            method='SLSQP',
            bounds=[(0, 1)] * n,
            constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1}
        )

        w = result.x
        ret = float(np.dot(w, self.expected_returns))
        vol = float(np.sqrt(np.dot(w, np.dot(self.cov_matrix, w))))
        sharpe = (ret - risk_free_rate) / vol

        logger.info(f"Max Sharpe portfolio — Return: {ret:.2%}, Vol: {vol:.2%}, Sharpe: {sharpe:.4f}")

        return PortfolioAllocation(
            weights={sym: float(wt) for sym, wt in zip(self.symbols, w)},
            expected_return=ret,
            expected_volatility=vol,
            sharpe_ratio=sharpe,
            efficient_frontier_data={}
        )

    def optimize_for_min_volatility(self) -> PortfolioAllocation:
        """
        Find weights that minimise portfolio volatility (the safest portfolio).

        Useful for conservative investors who prioritise capital preservation.
        Usually concentrates in low-volatility, low-correlation assets.
        """
        if self.returns_df is None:
            self.fetch_returns()

        n = len(self.symbols)

        def portfolio_vol(w):
            return float(np.sqrt(np.dot(w, np.dot(self.cov_matrix, w))))

        result = minimize(
            portfolio_vol,
            x0=np.full(n, 1 / n),
            method='SLSQP',
            bounds=[(0, 1)] * n,
            constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1}
        )

        w = result.x
        ret = float(np.dot(w, self.expected_returns))
        vol = portfolio_vol(w)

        logger.info(f"Min Variance portfolio — Return: {ret:.2%}, Vol: {vol:.2%}")

        return PortfolioAllocation(
            weights={sym: float(wt) for sym, wt in zip(self.symbols, w)},
            expected_return=ret,
            expected_volatility=vol,
            sharpe_ratio=(ret - 0.04) / vol,
            efficient_frontier_data={}
        )

    def generate_efficient_frontier(self, n_portfolios: int = 100) -> PortfolioAllocation:
        """
        Generate the efficient frontier: all risk/return optimal portfolios.

        For each target return level (from min to max possible return),
        find the portfolio with the lowest possible volatility.
        Plotting volatility vs return traces out a curved "frontier".

        Any portfolio below the frontier is suboptimal — you could get
        the same return with less risk by moving onto the frontier.
        """
        if self.returns_df is None:
            self.fetch_returns()

        n = len(self.symbols)
        target_returns = np.linspace(
            self.expected_returns.min(),
            self.expected_returns.max(),
            n_portfolios
        )

        frontier_vols, frontier_rets = [], []

        for target in target_returns:
            def portfolio_vol(w):
                return float(np.sqrt(np.dot(w, np.dot(self.cov_matrix, w))))

            result = minimize(
                portfolio_vol,
                x0=np.full(n, 1 / n),
                method='SLSQP',
                bounds=[(0, 1)] * n,
                constraints=[
                    {'type': 'eq', 'fun': lambda w: w.sum() - 1},
                    {'type': 'eq', 'fun': lambda w, t=target: np.dot(w, self.expected_returns) - t}
                ]
            )
            if result.success:
                frontier_vols.append(portfolio_vol(result.x))
                frontier_rets.append(float(target))

        return PortfolioAllocation(
            weights={},
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=0.0,
            efficient_frontier_data={
                'volatilities': frontier_vols,
                'returns': frontier_rets
            }
        )
