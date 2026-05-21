"""
Shared data-fetching utilities for the ML pipeline.

Centralises the yfinance download + retry logic so it isn't duplicated
across PortfolioRiskAnalyzer, AnomalyDetector, etc.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import List

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def download_close_prices(
    symbols: List[str],
    lookback_days: int = 252,
    max_attempts: int = 3,
) -> pd.DataFrame:
    """
    Download closing prices from Yahoo Finance with exponential-backoff retries.

    yfinance silently returns an empty DataFrame when rate-limited instead of
    raising — this wrapper detects that and retries before giving up.

    Args:
        symbols:      List of ticker symbols, e.g. ['AAPL', 'MSFT'].
        lookback_days: Calendar days of history to fetch (default ~1 year).
        max_attempts:  Number of download attempts before raising (default 3).

    Returns:
        DataFrame of closing prices, one column per symbol, indexed by date.

    Raises:
        RuntimeError: if data is still empty after all retries.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    prices: pd.DataFrame = pd.DataFrame()
    for attempt in range(max_attempts):
        raw = yf.download(symbols, start=start_date, end=end_date, progress=False)
        prices = raw["Close"] if "Close" in raw.columns else raw
        if not prices.empty:
            break
        if attempt < max_attempts - 1:
            wait = 2 ** (attempt + 1)   # 2 s, 4 s, …
            logger.warning(
                "Empty download for %s (attempt %d/%d) — "
                "Yahoo Finance may be rate-limiting. Retrying in %ds…",
                symbols, attempt + 1, max_attempts, wait,
            )
            time.sleep(wait)

    if prices.empty:
        raise RuntimeError(
            f"No price data returned for {symbols} after {max_attempts} attempts. "
            "Yahoo Finance may be rate-limiting this IP — wait a minute and retry."
        )

    logger.info("Downloaded %d rows for %s", len(prices), symbols)
    return prices
