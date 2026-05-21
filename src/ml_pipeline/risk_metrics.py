"""
Portfolio risk analysis: VaR, CVaR, Sharpe, Sortino, Max Drawdown, Beta.

All metrics are computed from historical daily returns fetched via yfinance.
Weights are passed in so the same class works for any portfolio composition.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List

from scipy.stats import norm

from .data_utils import download_close_prices

logger = logging.getLogger(__name__)


class PortfolioRiskAnalyzer:
    """
    Compute risk metrics for a multi-asset portfolio.

    Typical usage:
        analyzer = PortfolioRiskAnalyzer(['AAPL', 'MSFT', 'JPM'])
        analyzer.fetch_returns()
        var = analyzer.calculate_var({'AAPL': 0.4, 'MSFT': 0.4, 'JPM': 0.2})
    """

    def __init__(self, symbols: List[str], lookback_days: int = 252):
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.returns_df: pd.DataFrame = None
        self.price_df: pd.DataFrame = None

    def fetch_returns(self) -> pd.DataFrame:
        """
        Fetch closing prices and compute daily percentage returns.

        Uses :func:`download_close_prices` for rate-limit-safe downloading
        with exponential-backoff retries.

        Returns:
            DataFrame of daily returns, one column per symbol.

        Raises:
            RuntimeError: if data is still empty after all retries.
        """
        prices = download_close_prices(self.symbols, self.lookback_days)
        self.price_df = prices
        self.returns_df = prices.pct_change().dropna()
        logger.info(
            "Fetched returns for %d assets over %d days",
            len(self.symbols), len(self.returns_df),
        )
        return self.returns_df

    def _portfolio_returns(self, weights: Dict[str, float]) -> pd.Series:
        """Compute daily portfolio returns given asset weights."""
        w = np.array([weights[s] for s in self.returns_df.columns])
        return (self.returns_df * w).sum(axis=1)

    def calculate_var(self, weights: Dict[str, float],
                      confidence: float = 0.95,
                      method: str = 'parametric') -> float:
        """
        Value at Risk: the most you can expect to lose on a bad day.

        Args:
            weights: {symbol: allocation}, must sum to 1.0
            confidence: 0.95 means "95% of days, loss won't exceed this"
            method: 'historical' | 'parametric' | 'monte_carlo'

        Returns:
            Negative float, e.g. -0.012 means max daily loss is 1.2%

        Three methods:
            historical   — just take the 5th percentile of actual past returns
            parametric   — assume returns are normally distributed, use z-score
            monte_carlo  — simulate 10,000 random days from the same distribution
        """
        if self.returns_df is None:
            self.fetch_returns()

        port_returns = self._portfolio_returns(weights)

        if method == 'historical':
            var = float(np.percentile(port_returns, (1 - confidence) * 100))

        elif method == 'parametric':
            mu, sigma = port_returns.mean(), port_returns.std()
            var = float(mu + norm.ppf(1 - confidence) * sigma)

        elif method == 'monte_carlo':
            mu, sigma = port_returns.mean(), port_returns.std()
            simulated = np.random.normal(mu, sigma, 10_000)
            var = float(np.percentile(simulated, (1 - confidence) * 100))

        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"VaR ({confidence*100:.0f}%, {method}): {var:.4f} ({var*100:.2f}%)")
        return var

    def calculate_cvar(self, weights: Dict[str, float], confidence: float = 0.95) -> float:
        """
        Conditional VaR (Expected Shortfall): average loss on your worst days.

        Always worse than VaR — it's the mean of everything beyond the VaR threshold.
        Banks prefer CVaR because VaR ignores how bad the tail actually is.
        """
        if self.returns_df is None:
            self.fetch_returns()

        port_returns = self._portfolio_returns(weights)
        var = np.percentile(port_returns, (1 - confidence) * 100)
        cvar = float(port_returns[port_returns <= var].mean())

        logger.info(f"CVaR ({confidence*100:.0f}%): {cvar:.4f} ({cvar*100:.2f}%)")
        return cvar

    def calculate_sharpe_ratio(self, weights: Dict[str, float],
                               risk_free_rate: float = 0.04) -> float:
        """
        Sharpe ratio: how much return you get per unit of total risk.

        Formula: (Annual Return - Risk-Free Rate) / Annual Volatility

        Interpretation:
            > 1.0  — excellent
            0.5–1.0 — good
            < 0.5  — poor (not worth the risk)
        """
        if self.returns_df is None:
            self.fetch_returns()

        port_returns = self._portfolio_returns(weights)
        annual_return = port_returns.mean() * 252
        annual_vol = port_returns.std() * np.sqrt(252)
        sharpe = float((annual_return - risk_free_rate) / annual_vol)

        logger.info(f"Sharpe Ratio: {sharpe:.4f}")
        return sharpe

    def calculate_sortino_ratio(self, weights: Dict[str, float],
                                risk_free_rate: float = 0.04) -> float:
        """
        Sortino ratio: like Sharpe but only penalises downside volatility.

        Better for portfolios with upside-skewed returns — why punish upside swings?
        """
        if self.returns_df is None:
            self.fetch_returns()

        port_returns = self._portfolio_returns(weights)
        annual_return = port_returns.mean() * 252
        downside_vol = port_returns[port_returns < 0].std() * np.sqrt(252)
        sortino = float((annual_return - risk_free_rate) / downside_vol)

        logger.info(f"Sortino Ratio: {sortino:.4f}")
        return sortino

    def calculate_max_drawdown(self, weights: Dict[str, float]) -> float:
        """
        Maximum drawdown: worst peak-to-trough loss over the entire period.

        Returns a negative number, e.g. -0.35 means the portfolio fell 35%
        from its highest point before recovering (or not).
        """
        if self.returns_df is None:
            self.fetch_returns()

        port_returns = self._portfolio_returns(weights)
        cumulative = (1 + port_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = float(drawdown.min())

        logger.info(f"Max Drawdown: {max_dd:.4f} ({max_dd*100:.2f}%)")
        return max_dd

    def calculate_correlation_matrix(self) -> pd.DataFrame:
        """
        Pearson correlation matrix of daily returns.

        Low correlations mean better diversification — if one asset falls,
        others don't necessarily follow. Target: avoid pairs above 0.8.
        """
        if self.returns_df is None:
            self.fetch_returns()

        corr = self.returns_df.corr()
        upper = corr.values[np.triu_indices_from(corr.values, k=1)]
        logger.info(f"Average pairwise correlation: {upper.mean():.4f}")
        return corr

    def calculate_beta(self, symbol: str, benchmark: str = '^GSPC') -> float:
        """
        Beta: how much the stock moves relative to the market (S&P 500).

        Beta > 1 → amplifies market moves (aggressive)
        Beta = 1 → moves with the market
        Beta < 1 → more stable than market (defensive)
        Beta < 0 → moves opposite to market (hedge)
        """
        if self.returns_df is None:
            self.fetch_returns()

        bench_prices = download_close_prices(
            [benchmark],
            lookback_days=self.lookback_days,
        )
        bench_returns = bench_prices.squeeze().pct_change().dropna()

        common = self.returns_df.index.intersection(bench_returns.index)
        cov = np.cov(self.returns_df.loc[common, symbol], bench_returns.loc[common])[0, 1]
        beta = float(cov / np.var(bench_returns.loc[common]))

        logger.info(f"Beta for {symbol} vs {benchmark}: {beta:.4f}")
        return beta
