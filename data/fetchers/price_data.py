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
        logger.info(f"      📡 DATA FETCHER: Fetching {period} price history for {ticker}")
        logger.info(f"         → API: Yahoo Finance (yfinance)")
        logger.info(f"         → Period requested: {period}")

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

        # Log summary statistics
        first_date = data[0]["date"]
        last_date = data[-1]["date"]
        first_close = data[0]["close"]
        last_close = data[-1]["close"]
        price_change = ((last_close - first_close) / first_close) * 100
        avg_volume = sum(d["volume"] for d in data) / len(data)

        logger.info(f"      ✅ DATA FETCHER: Successfully fetched {len(data)} days of price data")
        logger.info(f"         → Date range: {first_date} to {last_date}")
        logger.info(f"         → Price range: ${min(d['close'] for d in data):.2f} - ${max(d['close'] for d in data):.2f}")
        logger.info(f"         → Period return: {price_change:+.2f}%")
        logger.info(f"         → Average volume: {avg_volume:,.0f}")
        logger.info(f"         → Latest close: ${last_close:.2f}")

        return data

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"      ❌ DATA FETCHER: Failed to fetch price data for {ticker}: {e}")
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
        logger.info(f"      📡 DATA FETCHER: Fetching fundamentals for {ticker}")
        logger.info(f"         → API: Yahoo Finance (yfinance)")

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

        logger.info(f"      ✅ DATA FETCHER: Successfully fetched {len(fundamentals)} fundamental metrics")
        logger.info(f"         → P/E Ratio: {fundamentals['pe_ratio']:.2f}")
        logger.info(f"         → Market Cap: ${fundamentals['market_cap']:,.0f}")
        logger.info(f"         → Profit Margin: {fundamentals['profit_margin']:.2%}")
        logger.info(f"         → Beta: {fundamentals['beta']:.2f}")

        return fundamentals

    except Exception as e:
        logger.error(f"      ❌ DATA FETCHER: Failed to fetch fundamentals for {ticker}: {e}")
        raise RuntimeError(f"Unable to fetch fundamental data for {ticker}: {str(e)}")
