"""
ML Pipeline Evaluation
======================
Measures whether the quantitative models meet basic accuracy and
calibration standards using real historical data.

Three checks
------------
1. ARIMA directional accuracy  — walk-forward validation.
   Baseline: 50% (coin flip).  Target: >50% (any non-random edge).

2. VaR model calibration — Kupiec's Proportion of Failures (POF) test.
   At 95% confidence the model should breach ~5% of trading days.
   H₀: model is calibrated → we fail to reject if p-value > 0.05.

3. Optimizer improvement — max-Sharpe portfolio must beat equal-weight.
   Mathematical sanity check; failure means a bug in the optimizer.

Usage
-----
    python -m tests.eval_ml
    python -m tests.eval_ml --symbols AAPL MSFT GOOGL --years 2 --arima-windows 8

Note: makes real yfinance calls — requires an internet connection.
ARIMA fitting is slow; 10 windows ≈ 5–10 min depending on symbol.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ml_pipeline import TimeSeriesForecaster, PortfolioOptimizer

# Suppress noisy info logs from the ML pipeline during eval
logging.basicConfig(level=logging.WARNING)


# ── formatting helpers ────────────────────────────────────────────────────────

def _divider(title: str = "") -> None:
    w = 64
    if title:
        pad = (w - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * (w - pad - len(title) - 2)}")
    else:
        print("─" * w)


# ── 1. ARIMA directional accuracy ─────────────────────────────────────────────

def eval_arima(symbol: str, n_windows: int = 10) -> dict:
    """
    Walk-forward validation: evenly-spaced windows across 2 years of history.

    For each window we:
      - Train ARIMA on all data up to that point
      - Predict horizon days ahead
      - Check whether the final predicted price is on the correct side
        of the last observed price (direction correct = True)

    Why directional accuracy and not raw MAPE?
      An investor cares whether to buy or sell — getting the direction right
      is more actionable than minimising a point-forecast error metric.
    """
    _divider(f"ARIMA Directional Accuracy — {symbol}")

    horizon   = 5    # predict 5 trading days ahead
    min_train = 200  # minimum training observations

    # Fetch ~2 years once; inject slices into each window's forecaster
    master = TimeSeriesForecaster(symbol, lookback_days=730)
    master.fetch_data()
    full_data = master.data.copy()
    prices    = full_data["Close"].dropna()

    if len(prices) < min_train + horizon + 1:
        print(f"  Not enough data ({len(prices)} rows). Skipping.")
        return {"symbol": symbol, "accuracy": float("nan"), "passed": False}

    # Evenly space evaluation indices after the minimum training cutoff
    eval_indices = np.linspace(
        min_train,
        len(prices) - horizon - 1,
        n_windows,
        dtype=int,
    )

    correct, total, mapes = 0, 0, []

    for i, idx in enumerate(eval_indices):
        f      = TimeSeriesForecaster(symbol)
        f.data = full_data.iloc[:idx].copy()

        try:
            fc = f.fit_arima(forecast_horizon=horizon)
        except Exception as exc:
            print(f"  window {i + 1:2d} (@day {idx:4d}): ARIMA failed — {exc}")
            continue

        last_actual  = float(prices.iloc[idx - 1])
        pred_dir     = fc.forecast_values[-1] > last_actual
        actual_dir   = float(prices.iloc[idx + horizon - 1]) > last_actual

        correct += int(pred_dir == actual_dir)
        total   += 1
        mapes.append(fc.mape)

        symbol_str = "✓" if pred_dir == actual_dir else "✗"
        print(
            f"  window {i + 1:2d} (@day {idx:4d}): "
            f"pred={'↑' if pred_dir else '↓'}  "
            f"actual={'↑' if actual_dir else '↓'}  "
            f"MAPE={fc.mape:5.1f}%  {symbol_str}"
        )

    accuracy  = correct / total if total else float("nan")
    mean_mape = float(np.mean(mapes)) if mapes else float("nan")

    print(f"\n  Windows evaluated   : {total}")
    print(f"  Correct direction   : {correct}/{total}")
    print(f"  Directional accuracy: {accuracy:.1%}  (random baseline 50%)")
    print(f"  Mean MAPE           : {mean_mape:.2f}%")

    passed = accuracy > 0.50
    print(f"  Result : {'PASS ✓' if passed else 'BELOW BASELINE — no directional edge detected'}")

    return {
        "symbol":    symbol,
        "accuracy":  accuracy,
        "mean_mape": mean_mape,
        "windows":   total,
        "passed":    passed,
    }


# ── 2. VaR calibration — Kupiec's POF test ───────────────────────────────────

def eval_var_kupiec(symbols: list, weights: dict, years: int = 2) -> dict:
    """
    Kupiec's Proportion of Failures (POF) test.

    How it works
    ────────────
    We roll a 252-day window across the full history.  On each day t we:
      1. Compute 95% historical VaR from the past 252 daily returns
      2. Check if day t's actual return falls below that threshold (a "violation")
    We count total violations x and total observations n.

    The expected violation rate under a correct model is p = 0.05 (5%).
    Kupiec's likelihood-ratio statistic:

        LR = -2 × [ x·ln(p/p̂) + (n−x)·ln((1−p)/(1−p̂)) ]

    where p̂ = x/n.  Under H₀ (model calibrated), LR ~ χ²(1).
    Critical value at 95% significance: 3.841.
    If LR < 3.841 (p-value > 0.05) we fail to reject H₀ → model is calibrated.
    """
    _divider("VaR Calibration — Kupiec POF Test")
    print(f"  Portfolio : {symbols}")
    print(f"  Weights   : { {s: f'{weights[s]:.2f}' for s in symbols} }")
    print(f"  Lookback  : {years}y rolling 252-day window  |  Confidence: 95%")

    raw    = yf.download(symbols, period=f"{years}y", auto_adjust=True, progress=False)
    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(symbols[0])
    prices  = prices[symbols]           # enforce column order
    returns = prices.pct_change().dropna()

    p       = 0.05                      # expected violation rate
    w_vec   = np.array([weights[s] for s in symbols])
    lookback = 252

    if len(returns) < lookback + 30:
        print("  Not enough data for rolling VaR test. Need at least 282 days.")
        return {"passed": False, "calibrated": False}

    violations, n = 0, 0

    for t in range(lookback, len(returns)):
        window    = returns.iloc[t - lookback : t]
        port_r    = (window.values * w_vec).sum(axis=1)
        var_est   = float(np.percentile(port_r, 5))   # 95% historical VaR

        next_r    = float(returns.iloc[t].values @ w_vec)
        violations += int(next_r < var_est)
        n          += 1

    actual_rate = violations / n
    p_hat       = max(1e-10, min(1 - 1e-10, actual_rate))   # avoid log(0)

    lr      = -2 * (
        violations * np.log(p / p_hat) +
        (n - violations) * np.log((1 - p) / (1 - p_hat))
    )
    p_value    = float(1 - stats.chi2.cdf(lr, df=1))
    calibrated = p_value > 0.05          # fail to reject H₀

    print(f"\n  Trading days tested : {n}")
    print(f"  VaR violations      : {violations}  "
          f"({actual_rate:.2%} actual  vs  {p:.2%} expected)")
    print(f"  Kupiec LR statistic : {lr:.3f}  (χ²(1) critical = 3.841)")
    print(f"  p-value             : {p_value:.4f}")
    print(f"  H₀ (calibrated)     : {'fail to reject ✓' if calibrated else 'REJECTED ✗'}")
    print(f"  Result : {'CALIBRATED ✓' if calibrated else 'MISCALIBRATED ✗'}")

    return {
        "n":            n,
        "violations":   violations,
        "actual_rate":  actual_rate,
        "expected_rate": p,
        "lr_statistic": lr,
        "p_value":      p_value,
        "calibrated":   calibrated,
        "passed":       calibrated,
    }


# ── 3. Optimizer sanity check ────────────────────────────────────────────────

def eval_optimizer(symbols: list, years: int = 2) -> dict:
    """
    Verify the Markowitz optimizer improves upon the equal-weight baseline.

    Why this is a good sanity check
    ────────────────────────────────
    The SLSQP optimizer has the equal-weight portfolio as its starting point.
    If the optimizer is working correctly it must find a point at least as
    good as that start — if max-Sharpe < equal-weight Sharpe, the objective
    function or constraints contain a bug.

    We also verify:
    - Weights sum to 1.0  (fully invested constraint)
    - No short positions   (long-only constraint)
    """
    _divider("Markowitz Optimizer")
    print(f"  Symbols: {symbols}  |  Lookback: {years}y")

    raw    = yf.download(symbols, period=f"{years}y", auto_adjust=True, progress=False)
    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(symbols[0])
    prices  = prices[symbols]
    returns = prices.pct_change().dropna()

    opt = PortfolioOptimizer(symbols)
    opt.returns_df       = returns
    opt.expected_returns = returns.mean() * 252
    opt.cov_matrix       = returns.cov()  * 252

    alloc = opt.optimize_for_max_sharpe()

    # Equal-weight baseline
    n     = len(symbols)
    eq_w  = np.full(n, 1 / n)
    eq_r  = float(opt.expected_returns.values @ eq_w)
    eq_v  = float(np.sqrt(eq_w @ opt.cov_matrix.values @ eq_w))
    eq_sh = (eq_r - 0.04) / eq_v

    w_sum = sum(alloc.weights.values())
    w_neg = [s for s, v in alloc.weights.items() if v < -1e-6]

    improved = alloc.sharpe_ratio >= eq_sh - 1e-4

    print(f"\n  Equal-weight — Sharpe: {eq_sh:.4f}  "
          f"(ret {eq_r:.2%}, vol {eq_v:.2%})")
    print(f"  Max-Sharpe   — Sharpe: {alloc.sharpe_ratio:.4f}  "
          f"(ret {alloc.expected_return:.2%}, vol {alloc.expected_volatility:.2%})")
    print(f"  Improvement          : {alloc.sharpe_ratio - eq_sh:+.4f}")
    print(f"  Weights              : { {s: f'{v:.3f}' for s, v in alloc.weights.items()} }")
    print(f"  Weights sum to 1     : {w_sum:.6f}  "
          f"{'✓' if abs(w_sum - 1) < 1e-4 else '✗ (constraint violated)'}")
    print(f"  No short positions   : {'✓' if not w_neg else f'✗  shorts: {w_neg}'}")
    print(f"  Result : {'PASS ✓' if improved else 'FAIL ✗  optimizer degraded Sharpe'}")

    return {
        "eq_sharpe":     eq_sh,
        "opt_sharpe":    alloc.sharpe_ratio,
        "improvement":   alloc.sharpe_ratio - eq_sh,
        "weights_sum":   w_sum,
        "short_positions": w_neg,
        "passed":        improved and abs(w_sum - 1) < 1e-4 and not w_neg,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="ML pipeline evaluation suite")
    parser.add_argument("--symbols",       nargs="+", default=["AAPL", "MSFT", "JPM"])
    parser.add_argument("--years",         type=int,  default=2)
    parser.add_argument("--arima-windows", type=int,  default=10,
                        help="Walk-forward windows per symbol  (10 ≈ 5–10 min)")
    args = parser.parse_args()

    symbols = [s.upper() for s in args.symbols]
    weights = {s: 1 / len(symbols) for s in symbols}

    print("\n" + "═" * 64)
    print("   FINANCIAL ADVISOR — ML PIPELINE EVALUATION")
    print("═" * 64)
    print(f"   Symbols : {symbols}")
    print(f"   Lookback: {args.years} year(s)   ARIMA windows: {args.arima_windows}")

    results = {}
    # Run on first symbol only — ARIMA is slow; one symbol is representative
    results["arima"]     = eval_arima(symbols[0], n_windows=args.arima_windows)
    results["var"]       = eval_var_kupiec(symbols, weights, years=args.years)
    results["optimizer"] = eval_optimizer(symbols, years=args.years)

    _divider("SUMMARY")
    checks = {
        "ARIMA directional accuracy  (>50%)":   results["arima"]["passed"],
        "VaR Kupiec POF calibration":            results["var"]["passed"],
        "Optimizer beats equal-weight Sharpe":   results["optimizer"]["passed"],
    }
    for label, ok in checks.items():
        print(f"  {'✓' if ok else '✗'}  {label}")

    overall = all(checks.values())
    print(f"\n  {'ALL CHECKS PASSED ✓' if overall else 'SOME CHECKS FAILED ✗'}")
    _divider()

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
