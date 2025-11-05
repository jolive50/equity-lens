"""Price data fetcher using yfinance for real-time stock prices."""

import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class PriceFetcher:
    """Fetches real-time and historical price data using yfinance."""

    def __init__(self, cache_enabled: bool = True):
        """Initialize price fetcher.

        Args:
            cache_enabled: Whether to use database caching (delegated to caller)
        """
        self.cache_enabled = cache_enabled

    def fetch_prices(
        self,
        ticker: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch price data for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y')
            interval: Data interval ('1m', '5m', '1h', '1d', '1wk', '1mo')

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume

        Raises:
            ValueError: If ticker is invalid or no data returned
            RuntimeError: If download fails
        """
        if not ticker or not isinstance(ticker, str):
            raise ValueError(f"Invalid ticker: {ticker}")

        ticker = ticker.strip().upper()
        logger.info(f"Fetching price data for {ticker} (period={period}, interval={interval})")

        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download {ticker}: {e}")

        if not isinstance(df, pd.DataFrame):
            raise RuntimeError(f"Unexpected data type from yfinance: {type(df)}")

        if df.empty:
            raise ValueError(f"No price data returned for {ticker}")

        # Flatten multi-level columns if present
        if df.columns.nlevels > 1:
            df = df.droplevel(0, axis=1)

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Remove timezone if present
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            df.index = df.index.tz_convert(None)

        df = df.sort_index()

        logger.info(f"Fetched {len(df)} rows for {ticker}")
        return df

    def fetch_latest_price(self, ticker: str) -> Dict[str, float]:
        """Fetch most recent price data.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict with latest OHLCV data

        Raises:
            ValueError: If no data available
        """
        df = self.fetch_prices(ticker, period="5d", interval="1d")

        if df.empty:
            raise ValueError(f"No recent data for {ticker}")

        latest = df.iloc[-1]

        return {
            'ticker': ticker,
            'date': df.index[-1].strftime('%Y-%m-%d'),
            'open': float(latest['Open']),
            'high': float(latest['High']),
            'low': float(latest['Low']),
            'close': float(latest['Close']),
            'volume': int(latest['Volume'])
        }
