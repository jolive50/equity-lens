"""Price data fetcher using yfinance for real-time stock data.

PAM's Component - Price Data Fetching
This module fetches real-time price data from Yahoo Finance for training and inference.
"""
import logging
from typing import List, Dict, Any
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)


def get_historical_data(ticker: str, period: str = "3mo") -> List[Dict[str, Any]]:
    """
    Fetch historical price data for a ticker using yfinance.

    Args:
        ticker: Stock symbol (e.g., "AAPL")
        period: Time period (e.g., "3mo", "1y", "5y")

    Returns:
        List of dictionaries with daily price/volume data

    Raises:
        ValueError: If ticker is invalid or no data found
        RuntimeError: If API call fails
    """
    try:
        logger.info(f"Fetching {period} historical data for {ticker}")
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, auto_adjust=True)

        if hist.empty:
            raise ValueError(f"No historical data found for {ticker}")

        # Convert to list of dictionaries
        data = []
        for date, row in hist.iterrows():
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "close": round(float(row["Close"]), 2),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "volume": int(row["Volume"])
            })

        logger.info(f"Fetched {len(data)} days of data for {ticker}")
        return data

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {e}")
        raise RuntimeError(f"Unable to fetch market data for {ticker}: {str(e)}")


def get_fundamentals(ticker: str) -> Dict[str, float]:
    """
    Fetch fundamental metrics for a ticker from Yahoo Finance.

    Args:
        ticker: Stock symbol (e.g., "AAPL")

    Returns:
        Dictionary with fundamental metrics
    """
    try:
        logger.info(f"Fetching fundamentals for {ticker}")
        stock = yf.Ticker(ticker)
        info = stock.info

        # Extract key metrics with defaults
        fundamentals = {
            "pe_ratio": float(info.get("trailingPE", 0) or 0),
            "forward_pe": float(info.get("forwardPE", 0) or 0),
            "market_cap": float(info.get("marketCap", 0) or 0),
            "revenue_growth": float(info.get("revenueGrowth", 0) or 0),
            "profit_margin": float(info.get("profitMargins", 0) or 0),
            "debt_to_equity": float(info.get("debtToEquity", 0) or 0),
            "roe": float(info.get("returnOnEquity", 0) or 0),
            "beta": float(info.get("beta", 1.0) or 1.0),
        }

        logger.info(f"Fetched {len(fundamentals)} fundamental metrics for {ticker}")
        return fundamentals

    except Exception as e:
        logger.error(f"Failed to fetch fundamentals for {ticker}: {e}")
        raise RuntimeError(f"Unable to fetch fundamental data for {ticker}: {str(e)}")
