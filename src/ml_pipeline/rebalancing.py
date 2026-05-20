"""
Rebalancing drift detector.

Uses a Gradient Boosting classifier to decide: does this portfolio need
rebalancing right now, or can it wait?

Features:
    - How far have current weights drifted from target? (L1 drift)
    - How long since the last rebalance?
    - What volatility regime are we in? (low / medium / high)

Why ML instead of a simple threshold rule?
    A rule like "rebalance if any weight drifts >5%" is easy but ignores context.
    In a high-volatility regime, small drift matters more. In a low-vol regime,
    you can tolerate more drift before transaction costs eat into the benefit.

Note on training data:
    The model needs labelled examples (drift state → should_rebalance: yes/no).
    In production, this comes from historical portfolio management records.
    Here we generate synthetic training data that encodes sensible heuristics,
    which can later be replaced with real labelled data.
"""

import logging
import numpy as np
from typing import Dict, Tuple

from sklearn.ensemble import GradientBoostingClassifier

logger = logging.getLogger(__name__)

# Volatility regime encoding used consistently across the class
VOL_ENCODING = {'low': 0, 'medium': 1, 'high': 2}


class RebalancingDetector:
    """
    Predict whether a portfolio needs rebalancing.

    Typical usage:
        detector = RebalancingDetector()
        detector.fit_on_synthetic_data()
        features = detector.build_features(
            current={'AAPL': 0.45, 'MSFT': 0.35, 'JPM': 0.20},
            target={'AAPL': 0.33, 'MSFT': 0.33, 'JPM': 0.34},
            days_since_rebalance=45,
            volatility_regime='high'
        )
        rebalance, confidence = detector.predict(features)
    """

    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.is_fitted = False

    def build_features(self,
                       current: Dict[str, float],
                       target: Dict[str, float],
                       days_since_rebalance: int,
                       volatility_regime: str) -> np.ndarray:
        """
        Encode the current portfolio state into a feature vector.

        Args:
            current: Current portfolio weights (must share keys with target)
            target: Target portfolio weights
            days_since_rebalance: Days since last rebalancing event
            volatility_regime: 'low', 'medium', or 'high'

        Returns:
            Shape (1, 3) numpy array for the classifier
        """
        drift = float(np.sum(np.abs(
            np.array(list(current.values())) - np.array(list(target.values()))
        )))

        return np.array([[
            drift,
            days_since_rebalance,
            VOL_ENCODING[volatility_regime]
        ]])

    def fit_on_synthetic_data(self, n_samples: int = 1000):
        """
        Train on synthetically generated labelled examples.

        Label heuristic (encodes sensible domain knowledge):
            Rebalance = YES if:
                drift > 0.10 (weights off by more than 10 percentage points), OR
                drift > 0.05 AND it's been more than 90 days, OR
                high-vol regime AND drift > 0.03

        This is a placeholder — replace with real historical data
        when available (e.g. from a portfolio management system).
        """
        rng = np.random.default_rng(42)

        drift = rng.uniform(0, 0.30, n_samples)
        days = rng.integers(1, 365, n_samples)
        regime = rng.integers(0, 3, n_samples)  # 0=low, 1=medium, 2=high

        labels = (
            (drift > 0.10) |
            ((drift > 0.05) & (days > 90)) |
            ((regime == 2) & (drift > 0.03))
        ).astype(int)

        X = np.column_stack([drift, days, regime])
        self.model.fit(X, labels)
        self.is_fitted = True
        logger.info(f"RebalancingDetector fitted on {n_samples} synthetic samples")

    def predict(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        Predict if rebalancing is needed.

        Args:
            features: Output of build_features()

        Returns:
            (rebalance_needed: bool, confidence: float 0–1)
        """
        if not self.is_fitted:
            logger.warning("Model not fitted — call fit_on_synthetic_data() first")
            return False, 0.5

        prediction = bool(self.model.predict(features)[0])
        confidence = float(self.model.predict_proba(features).max())
        return prediction, confidence
