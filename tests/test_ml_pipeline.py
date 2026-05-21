"""
Unit tests for the classical ML pipeline.

All tests inject synthetic data directly into class instances so no
network calls are made — tests run fast and deterministically.

Fixtures are scoped to 'module' so synthetic data is generated once
and shared across all test classes.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml_pipeline import (
    TimeSeriesForecaster,
    PortfolioRiskAnalyzer,
    PortfolioOptimizer,
    AnomalyDetector,
)
from src.ml_pipeline.rebalancing import RebalancingDetector


# ============================================================================
# SHARED SYNTHETIC DATA
# ============================================================================

N_DAYS  = 200
SYMBOLS = ["AAPL", "MSFT", "JPM"]
DATES   = pd.date_range("2024-01-01", periods=N_DAYS, freq="B")


@pytest.fixture(scope="module")
def synthetic_returns():
    """
    200 days of daily returns for 3 assets.
    Deterministic (seed=42) so tests never flake.
    """
    np.random.seed(42)
    data = {
        "AAPL": np.random.normal(0.0008, 0.015, N_DAYS),
        "MSFT": np.random.normal(0.0005, 0.013, N_DAYS),
        "JPM":  np.random.normal(0.0002, 0.017, N_DAYS),
    }
    return pd.DataFrame(data, index=DATES)


@pytest.fixture(scope="module")
def synthetic_prices(synthetic_returns):
    """Prices derived from returns (AAPL=150, MSFT=300, JPM=120 at t=0)."""
    starts = {"AAPL": 150.0, "MSFT": 300.0, "JPM": 120.0}
    prices = {
        sym: start * (1 + synthetic_returns[sym]).cumprod()
        for sym, start in starts.items()
    }
    return pd.DataFrame(prices, index=DATES)


@pytest.fixture(scope="module")
def equal_weights():
    return {s: 1 / len(SYMBOLS) for s in SYMBOLS}


# ============================================================================
# PORTFOLIO RISK ANALYZER
# ============================================================================

class TestPortfolioRiskAnalyzer:

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_returns, equal_weights):
        """Inject synthetic data — no yfinance network call."""
        self.analyzer = PortfolioRiskAnalyzer(SYMBOLS)
        self.analyzer.returns_df = synthetic_returns
        self.weights = equal_weights

    # --- VaR ---

    def test_var_is_negative(self):
        """VaR represents a loss, so it must be negative."""
        var = self.analyzer.calculate_var(self.weights, method="parametric")
        assert var < 0

    def test_var_is_within_realistic_range(self):
        """Daily VaR for a diversified 3-stock portfolio should not exceed 20%."""
        var = self.analyzer.calculate_var(self.weights, method="parametric")
        assert var > -0.20

    def test_all_var_methods_agree_within_tolerance(self):
        """Historical, parametric, and Monte Carlo VaR should be in the same ballpark."""
        hist = self.analyzer.calculate_var(self.weights, method="historical")
        para = self.analyzer.calculate_var(self.weights, method="parametric")
        mc   = self.analyzer.calculate_var(self.weights, method="monte_carlo")
        assert abs(hist - para) < 0.015, "Historical vs parametric VaR diverged"
        assert abs(hist - mc)   < 0.015, "Historical vs Monte Carlo VaR diverged"

    # --- CVaR ---

    def test_cvar_is_worse_than_var(self):
        """CVaR is the average loss *beyond* VaR, so it must be <= VaR."""
        var  = self.analyzer.calculate_var(self.weights, method="parametric")
        cvar = self.analyzer.calculate_cvar(self.weights)
        assert cvar <= var, f"CVaR {cvar:.4f} should be <= VaR {var:.4f}"

    # --- Sharpe / Sortino ---

    def test_sharpe_is_a_sane_float(self):
        sharpe = self.analyzer.calculate_sharpe_ratio(self.weights)
        assert isinstance(sharpe, float)
        assert -5.0 < sharpe < 10.0

    def test_sortino_is_greater_than_or_equal_to_sharpe(self):
        """
        Sortino only penalises downside vol, so it should be >= Sharpe
        when returns are positively skewed (our synthetic data has positive drift).
        """
        sharpe  = self.analyzer.calculate_sharpe_ratio(self.weights)
        sortino = self.analyzer.calculate_sortino_ratio(self.weights)
        assert sortino >= sharpe - 0.1  # small tolerance for edge cases

    # --- Max Drawdown ---

    def test_max_drawdown_is_negative(self):
        dd = self.analyzer.calculate_max_drawdown(self.weights)
        assert dd < 0

    def test_max_drawdown_within_range(self):
        dd = self.analyzer.calculate_max_drawdown(self.weights)
        assert -1.0 <= dd <= 0.0

    # --- Correlation ---

    def test_correlation_matrix_shape(self):
        corr = self.analyzer.calculate_correlation_matrix()
        assert corr.shape == (len(SYMBOLS), len(SYMBOLS))

    def test_correlation_diagonal_is_one(self):
        corr = self.analyzer.calculate_correlation_matrix()
        np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-10)

    def test_correlation_is_symmetric(self):
        corr = self.analyzer.calculate_correlation_matrix()
        np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-10)

    def test_correlation_values_in_valid_range(self):
        corr = self.analyzer.calculate_correlation_matrix()
        assert corr.values.min() >= -1.0 - 1e-10
        assert corr.values.max() <= 1.0  + 1e-10


# ============================================================================
# PORTFOLIO OPTIMIZER
# ============================================================================

class TestPortfolioOptimizer:

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_returns):
        """Inject pre-computed expected returns and covariance — no yfinance."""
        self.opt = PortfolioOptimizer(SYMBOLS)
        self.opt.returns_df       = synthetic_returns
        self.opt.expected_returns = synthetic_returns.mean() * 252
        self.opt.cov_matrix       = synthetic_returns.cov()  * 252

    # --- Max Sharpe ---

    def test_max_sharpe_weights_sum_to_one(self):
        alloc = self.opt.optimize_for_max_sharpe()
        total = sum(alloc.weights.values())
        assert abs(total - 1.0) < 1e-5, f"Weights sum to {total:.6f}, expected 1.0"

    def test_max_sharpe_no_short_selling(self):
        alloc = self.opt.optimize_for_max_sharpe()
        for sym, w in alloc.weights.items():
            assert w >= -1e-6, f"{sym} weight {w:.6f} violates no-short-selling"

    def test_max_sharpe_ratio_is_positive(self):
        """With positive expected returns, max-Sharpe portfolio should have Sharpe > 0."""
        alloc = self.opt.optimize_for_max_sharpe()
        assert alloc.sharpe_ratio > 0

    # --- Min Volatility ---

    def test_min_vol_weights_sum_to_one(self):
        alloc = self.opt.optimize_for_min_volatility()
        total = sum(alloc.weights.values())
        assert abs(total - 1.0) < 1e-5

    def test_min_vol_lower_than_equal_weight(self):
        """Optimised min-vol portfolio should be <= equal-weight volatility."""
        n = len(SYMBOLS)
        eq_w   = np.array([1 / n] * n)
        eq_vol = float(np.sqrt(eq_w @ self.opt.cov_matrix.values @ eq_w))

        alloc = self.opt.optimize_for_min_volatility()
        assert alloc.expected_volatility <= eq_vol + 1e-4, (
            f"Min-vol {alloc.expected_volatility:.4f} > equal-weight {eq_vol:.4f}"
        )


# ============================================================================
# ANOMALY DETECTOR
# ============================================================================

class TestAnomalyDetector:

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_returns):
        """Inject synthetic returns — no yfinance."""
        self.detector = AnomalyDetector(SYMBOLS)
        self.detector.returns_df = synthetic_returns

    def test_features_contain_momentum_and_volatility_columns(self):
        features = self.detector.engineer_features()
        for sym in SYMBOLS:
            assert f"{sym}_momentum"   in features.columns, f"Missing {sym}_momentum"
            assert f"{sym}_volatility" in features.columns, f"Missing {sym}_volatility"

    def test_isolation_forest_labels_are_plus_minus_one(self):
        self.detector.engineer_features()
        labels = self.detector.detect_anomalies_isolation_forest(contamination=0.05)
        unique = set(labels.unique())
        assert unique.issubset({-1, 1}), f"Unexpected label values: {unique}"

    def test_anomaly_rate_close_to_contamination_param(self):
        """Isolation Forest should flag ~5% of observations as anomalies."""
        self.detector.engineer_features()
        labels = self.detector.detect_anomalies_isolation_forest(contamination=0.05)
        anomaly_rate = (labels == -1).mean()
        assert 0.03 < anomaly_rate < 0.08, (
            f"Anomaly rate {anomaly_rate:.2%} is far from 5% contamination target"
        )

    def test_zscore_outlier_returns_boolean_dataframe(self):
        self.detector.engineer_features()
        outliers = self.detector.detect_outliers_zscore(threshold=3.0)
        assert outliers.dtypes.eq(bool).all(), "Z-score outlier flags should all be bool"


# ============================================================================
# TIME SERIES FORECASTER  (no network — synthetic OHLCV injected)
# ============================================================================

class TestTimeSeriesForecaster:

    @pytest.fixture(autouse=True)
    def setup(self, synthetic_prices):
        """Build a minimal OHLCV DataFrame and inject into the forecaster."""
        price = synthetic_prices["AAPL"]
        np.random.seed(0)
        ohlcv = pd.DataFrame({
            "Open":   price * 0.99,
            "High":   price * 1.01,
            "Low":    price * 0.98,
            "Close":  price,
            "Volume": np.random.randint(1_000_000, 10_000_000, len(price)),
        }, index=price.index)
        ohlcv["Returns"] = ohlcv["Close"].pct_change()

        self.forecaster = TimeSeriesForecaster("AAPL")
        self.forecaster.data = ohlcv

    def test_stationarity_returns_bool_and_pvalue(self):
        is_stationary, p_value = self.forecaster.test_stationarity()
        # adfuller returns numpy.bool_, which is NOT a subclass of Python bool.
        # We check the value is truthy/falsy, not the exact type.
        assert is_stationary in (True, False)
        assert 0.0 <= p_value <= 1.0

    def test_technical_indicators_expected_columns(self):
        df = self.forecaster.calculate_technical_indicators()
        for col in ["RSI", "SMA_20", "MACD", "MACD_Signal"]:
            assert col in df.columns, f"Missing technical indicator: {col}"

    def test_rsi_values_in_valid_range(self):
        df = self.forecaster.calculate_technical_indicators().dropna()
        assert df["RSI"].min() >= 0.0,   "RSI below 0"
        assert df["RSI"].max() <= 100.0, "RSI above 100"

    def test_sma20_lags_price(self):
        """SMA_20 should be NaN for first 19 rows and valid after."""
        df = self.forecaster.calculate_technical_indicators()
        assert df["SMA_20"].iloc[:19].isna().all()
        assert df["SMA_20"].iloc[20:].notna().all()


# ============================================================================
# REBALANCING DETECTOR
# ============================================================================

class TestRebalancingDetector:

    def test_unfitted_returns_safe_defaults(self):
        """Unfitted model should not crash — returns default prediction."""
        detector = RebalancingDetector()
        features = detector.build_features(
            current={"AAPL": 0.4, "MSFT": 0.6},
            target={"AAPL": 0.5, "MSFT": 0.5},
            days_since_rebalance=45,
            volatility_regime="medium",
        )
        needs_rebalance, confidence = detector.predict(features)
        assert isinstance(needs_rebalance, bool)
        assert 0.0 <= confidence <= 1.0

    def test_fitted_model_on_high_drift(self):
        """After fitting, high drift + long interval should recommend rebalancing."""
        detector = RebalancingDetector()
        detector.fit_on_synthetic_data(n_samples=500)

        features = detector.build_features(
            current={"AAPL": 0.75, "MSFT": 0.25},
            target={"AAPL": 0.50, "MSFT": 0.50},
            days_since_rebalance=180,
            volatility_regime="high",
        )
        needs_rebalance, confidence = detector.predict(features)
        assert isinstance(needs_rebalance, bool)
        assert confidence >= 0.5

    def test_fitted_model_on_low_drift(self):
        """Tiny drift + recent rebalance + low vol should probably NOT rebalance."""
        detector = RebalancingDetector()
        detector.fit_on_synthetic_data(n_samples=500)

        features = detector.build_features(
            current={"AAPL": 0.505, "MSFT": 0.495},
            target={"AAPL": 0.500, "MSFT": 0.500},
            days_since_rebalance=3,
            volatility_regime="low",
        )
        needs_rebalance, _ = detector.predict(features)
        # With very low drift the model should lean toward not rebalancing
        assert needs_rebalance is False
