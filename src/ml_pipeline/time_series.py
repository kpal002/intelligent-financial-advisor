"""
Time series forecasting for individual stocks.

Uses ARIMA (via statsmodels + pmdarima auto-selection) to:
- Test whether a price series is stationary (ADF test)
- Fit the best ARIMA(p,d,q) order automatically
- Generate a 30-day forecast with 95% confidence intervals
- Calculate MAPE on a held-out validation window
- Compute technical indicators (SMA, RSI, MACD)
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple
from datetime import datetime, timedelta

import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

from .models import TimeSeriesForecast

logger = logging.getLogger(__name__)


class TimeSeriesForecaster:
    """
    Forecast future prices for a single stock using ARIMA.

    Typical usage:
        forecaster = TimeSeriesForecaster('AAPL')
        forecaster.fetch_data()
        forecast = forecaster.fit_arima(forecast_horizon=30)
    """

    def __init__(self, symbol: str, lookback_days: int = 252):
        """
        Args:
            symbol: Stock ticker, e.g. 'AAPL'
            lookback_days: How many calendar days of history to fetch (default ~1 year)
        """
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.data: pd.DataFrame = None
        self.arima_model = None

    def fetch_data(self) -> pd.DataFrame:
        """
        Download OHLCV data from Yahoo Finance and add a Returns column.

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Returns
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)

        logger.info(f"Fetching {self.symbol} from {start_date.date()} to {end_date.date()}")
        self.data = yf.download(self.symbol, start=start_date, end=end_date, progress=False)
        self.data['Returns'] = self.data['Close'].pct_change()
        return self.data

    def test_stationarity(self) -> Tuple[bool, float]:
        """
        Augmented Dickey-Fuller test: is the price series stationary?

        Returns:
            (is_stationary, p_value)

        Why this matters:
            ARIMA requires a stationary series. Raw prices are usually NOT stationary
            (they trend). We need to difference them (d=1 or d=2) to make them stationary.
            The 'd' in ARIMA(p,d,q) is how many times we difference.
            Returns (% changes) ARE usually stationary, which is why risk models use returns.
        """
        if self.data is None:
            self.fetch_data()

        result = adfuller(self.data['Close'].dropna(), autolag='AIC')
        adf_statistic, p_value = result[0], result[1]
        is_stationary = p_value < 0.05

        logger.info(f"ADF Test for {self.symbol}:")
        logger.info(f"  ADF Statistic: {adf_statistic:.4f}")
        logger.info(f"  P-value: {p_value:.4f}")
        logger.info(f"  Stationary: {is_stationary}")
        return is_stationary, p_value

    def fit_arima(self, forecast_horizon: int = 30) -> TimeSeriesForecast:
        """
        Auto-select the best ARIMA order and generate a forecast.

        Args:
            forecast_horizon: Days ahead to forecast

        Returns:
            TimeSeriesForecast with values, dates, confidence intervals, and MAPE

        How it works:
            1. auto_arima tries many (p,d,q) combinations, picks lowest AIC
            2. Refit on the full series with the chosen order
            3. Hold out last 30 days to compute MAPE (how accurate the model is)
            4. Forecast `forecast_horizon` days beyond the last observed date
        """
        if self.data is None:
            self.fetch_data()

        close_prices = self.data['Close'].dropna()

        logger.info(f"Running auto_arima for {self.symbol}...")
        best_order = auto_arima(
            close_prices,
            seasonal=False,
            stepwise=True,
            max_p=5, max_d=2, max_q=5,
            information_criterion='aic',
            trace=False
        ).order
        logger.info(f"Best ARIMA order: {best_order}")

        # Pass .values (numpy) to avoid statsmodels DatetimeIndex frequency warnings
        self.arima_model = ARIMA(close_prices.values, order=best_order).fit()

        forecast_result = self.arima_model.get_forecast(steps=forecast_horizon)
        forecast_values = np.asarray(forecast_result.predicted_mean)
        ci = np.asarray(forecast_result.conf_int(alpha=0.05))  # 95% CI

        # Validation MAPE: train on all-but-last-30, forecast 30, compare to actuals
        test_size = 30
        val_model = ARIMA(close_prices.values[:-test_size], order=best_order).fit()
        predictions = np.asarray(val_model.get_forecast(steps=test_size).predicted_mean)
        actuals = close_prices.values[-test_size:]
        mape = float(np.mean(np.abs((actuals - predictions) / actuals)) * 100)

        last_date = self.data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_horizon,
            freq='D'
        )

        return TimeSeriesForecast(
            forecast_values=forecast_values,
            forecast_dates=forecast_dates,
            confidence_interval_lower=ci[:, 0],
            confidence_interval_upper=ci[:, 1],
            mape=mape,
            model_type='ARIMA'
        )

    def calculate_technical_indicators(self) -> pd.DataFrame:
        """
        Compute SMA, RSI, and MACD from closing prices.

        Returns:
            DataFrame with columns: SMA_20, SMA_50, RSI, MACD, MACD_Signal

        What each tells you:
            SMA_20/50: Is price above or below its moving average? (trend direction)
            RSI:       0-100 oscillator. >70 = overbought, <30 = oversold.
            MACD:      Difference between 12-day and 26-day EMAs. Positive = upward momentum.
        """
        df = self.data.copy()

        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()

        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + gain / loss))

        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

        return df[['SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal']]
