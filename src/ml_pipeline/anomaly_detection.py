"""
Anomaly detection for securities and trading patterns.

Two methods:
  1. Isolation Forest — multivariate, learns what "normal" looks like across
     several features (momentum, volatility, skew, kurtosis) and flags outliers.
  2. Z-score — univariate, flags any feature value more than N std deviations
     from its mean. Simpler and more interpretable.

Use cases:
  - Spot mispriced assets before the market corrects
  - Detect unusual trading activity (potential manipulation or news catalyst)
  - Identify regime changes (e.g. volatility spiking across the portfolio)
"""

import logging
import numpy as np
import pandas as pd
from typing import List
from datetime import datetime, timedelta

import yfinance as yf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Detect anomalous market behaviour across a set of securities.

    Typical usage:
        detector = AnomalyDetector(['AAPL', 'MSFT', 'JPM'])
        detector.fetch_data()
        detector.engineer_features()
        anomalies = detector.detect_anomalies_isolation_forest()
        # anomalies: Series of -1 (anomaly) or 1 (normal) per date
    """

    def __init__(self, symbols: List[str], lookback_days: int = 252):
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.returns_df: pd.DataFrame = None
        self.features_df: pd.DataFrame = None
        self.iso_forest: IsolationForest = None

    def fetch_data(self) -> pd.DataFrame:
        """Download historical prices and compute daily returns."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)

        prices = yf.download(self.symbols, start=start_date, end=end_date, progress=False)['Close']
        self.returns_df = prices.pct_change().dropna()
        return self.returns_df

    def engineer_features(self, window: int = 30) -> pd.DataFrame:
        """
        Build features from return history for each symbol.

        Features per symbol (over a rolling `window`-day window):
            momentum   — cumulative return (positive = recent uptrend)
            volatility — rolling std dev (higher = more risk)
            skewness   — return distribution shape (negative = left tail)
            kurtosis   — tail fatness (>3 = more extreme moves than normal)

        Why rolling features?
            A single day's return is noisy. Rolling statistics capture the
            *regime* the stock is in, which is what we want to flag as anomalous.
        """
        if self.returns_df is None:
            self.fetch_data()

        features = pd.DataFrame(index=self.returns_df.index)

        for symbol in self.symbols:
            r = self.returns_df[symbol]
            features[f'{symbol}_momentum'] = r.rolling(window).sum()
            features[f'{symbol}_volatility'] = r.rolling(window).std()
            features[f'{symbol}_skewness'] = r.rolling(window).skew()
            features[f'{symbol}_kurtosis'] = r.rolling(window).kurt()

        self.features_df = features.bfill().dropna()
        return self.features_df

    def detect_anomalies_isolation_forest(self, contamination: float = 0.05) -> pd.Series:
        """
        Flag anomalous dates using Isolation Forest.

        Args:
            contamination: Expected proportion of anomalies (0.05 = flag ~5% of days)

        Returns:
            Series indexed by date: -1 = anomaly, 1 = normal

        How Isolation Forest works:
            Randomly partition features using decision trees. Anomalies are easier
            to isolate (fewer splits needed) because they're far from the dense
            normal cluster. It's O(log n) and doesn't assume a distribution.
        """
        if self.features_df is None:
            self.engineer_features()

        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.features_df)

        self.iso_forest = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            random_state=42
        )
        labels = self.iso_forest.fit_predict(scaled)

        n_anomalies = (labels == -1).sum()
        logger.info(f"Isolation Forest: {n_anomalies} anomalies ({n_anomalies/len(labels)*100:.1f}%)")

        return pd.Series(labels, index=self.features_df.index, name='anomaly')

    def detect_outliers_zscore(self, threshold: float = 3.0) -> pd.DataFrame:
        """
        Flag features more than `threshold` standard deviations from the mean.

        Args:
            threshold: 3.0 catches the extreme 0.3% of a normal distribution

        Returns:
            Boolean DataFrame — True where a feature is an outlier on that date

        Simpler than Isolation Forest and easier to explain:
            "AAPL's 30-day volatility jumped 4 standard deviations above normal."
        """
        if self.features_df is None:
            self.engineer_features()

        z_scores = (self.features_df - self.features_df.mean()) / self.features_df.std()
        outliers = z_scores.abs() > threshold

        logger.info(f"Z-score outliers (threshold={threshold}): {outliers.sum().sum()} total")
        return outliers
