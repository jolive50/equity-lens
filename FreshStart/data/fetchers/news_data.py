import os
import requests
from typing import Dict, List, Optional
from datetime import datetime
import time


class NewsDataFetcher:
    """Fetch financial news from Alpha Vantage NEWS_SENTIMENT API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize news fetcher with API key.

        Args:
            api_key: Alpha Vantage API key (defaults to ALPHA_VANTAGE_API_KEY env var)

        Raises:
            ValueError: If API key not provided
        """
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Alpha Vantage API key required")

        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12  # Alpha Vantage: 5 calls/min for free tier

    def fetch_news(
        self,
        ticker: str,
        limit: int = 50,
        time_from: Optional[str] = None,
        time_to: Optional[str] = None
    ) -> List[Dict]:
        """Fetch news articles for a specific ticker.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            limit: Maximum number of articles to return (max 1000)
            time_from: Start time in YYYYMMDDTHHMM format (optional)
            time_to: End time in YYYYMMDDTHHMM format (optional)

        Returns:
            List of article dictionaries with keys: title, content, source, timestamp, url

        Raises:
            RuntimeError: If API call fails
        """
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": self.api_key,
            "limit": min(limit, 1000)  # API max is 1000
        }

        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "Information" in data:
                raise RuntimeError(f"API rate limit: {data['Information']}")

            if "Error Message" in data:
                raise RuntimeError(f"API error: {data['Error Message']}")

            if "feed" not in data:
                raise RuntimeError(f"Unexpected API response: {data}")

            articles = self._parse_articles(data["feed"])

            # Rate limiting
            time.sleep(self.rate_limit_delay)

            return articles

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch news for {ticker}: {e}") from e

    def fetch_multiple_tickers(
        self,
        tickers: List[str],
        limit_per_ticker: int = 50
    ) -> Dict[str, List[Dict]]:
        """Fetch news for multiple tickers.

        Args:
            tickers: List of ticker symbols
            limit_per_ticker: Max articles per ticker

        Returns:
            Dictionary mapping ticker -> list of articles
        """
        results = {}
        for ticker in tickers:
            try:
                articles = self.fetch_news(ticker, limit=limit_per_ticker)
                results[ticker] = articles
            except Exception as e:
                print(f"Warning: Failed to fetch news for {ticker}: {e}")
                results[ticker] = []
        return results

    def _parse_articles(self, feed: List[Dict]) -> List[Dict]:
        """Parse raw Alpha Vantage feed into standardized format.

        Args:
            feed: Raw feed from Alpha Vantage API

        Returns:
            List of parsed article dictionaries
        """
        articles = []
        for item in feed:
            article = {
                "title": item.get("title", ""),
                "content": item.get("summary", ""),
                "source": item.get("source", "unknown"),
                "timestamp": item.get("time_published", datetime.now().isoformat()),
                "url": item.get("url", ""),
                "sentiment": {
                    "label": item.get("overall_sentiment_label", "Neutral"),
                    "score": float(item.get("overall_sentiment_score", 0.0))
                },
                "ticker_sentiment": item.get("ticker_sentiment", [])
            }
            articles.append(article)

        return articles
