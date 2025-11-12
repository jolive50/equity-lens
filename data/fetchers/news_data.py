import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class NewsDataFetcher:
    """Fetch financial news from Alpha Vantage NEWS_SENTIMENT API."""

    def __init__(self, api_key: Optional[str] = None, require_api_key: bool = False):
        """Initialize news fetcher with API key.

        Args:
            api_key: Alpha Vantage API key (defaults to ALPHA_VANTAGE_API_KEY env var)
            require_api_key: If True, raise when API key is missing

        Raises:
            ValueError: If API key not provided
        """
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.require_api_key = require_api_key
        self._api_available = bool(self.api_key)

        if not self.api_key and require_api_key:
            raise ValueError("Alpha Vantage API key required")
        if not self.api_key:
            logger.warning(
                "Alpha Vantage API key not configured; NewsDataFetcher will fall back "
                "to offline sample articles."
            )

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
        if not self.api_key:
            logger.info(f"      📡 DATA FETCHER: Using offline news sample for {ticker}")
            logger.info(f"         → Reason: API key not configured")
            articles = self._offline_articles(ticker, limit)
            logger.info(f"      ✅ DATA FETCHER: Generated {len(articles)} offline sample articles")
            return articles

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
            logger.info(f"      📡 DATA FETCHER: Fetching news for {ticker}")
            logger.info(f"         → API: Alpha Vantage NEWS_SENTIMENT")
            logger.info(f"         → Limit: {params['limit']} articles")

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

            # Log article statistics
            if articles:
                sentiments = [a["sentiment"]["label"] for a in articles]
                pos_count = sentiments.count("Positive")
                neg_count = sentiments.count("Negative")
                neu_count = sentiments.count("Neutral")

                logger.info(f"      ✅ DATA FETCHER: Successfully fetched {len(articles)} news articles")
                logger.info(f"         → Sentiment distribution: {pos_count} positive, {neu_count} neutral, {neg_count} negative")
                logger.info(f"         → Latest article: \"{articles[0]['title'][:50]}...\"")
                logger.info(f"         → Sources: {len(set(a['source'] for a in articles))} unique sources")

            # Rate limiting
            time.sleep(self.rate_limit_delay)

            return articles

        except requests.exceptions.RequestException as e:
            logger.error(f"      ❌ DATA FETCHER: Failed to fetch news for {ticker}: {e}")
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

    def _offline_articles(self, ticker: str, limit: int) -> List[Dict]:
        """Return deterministic offline articles when API access is unavailable."""
        templates = [
            {
                "title": "{ticker} extends rally as demand stays resilient",
                "content": (
                    "{ticker} shares advanced in extended trading after analysts pointed "
                    "to resilient demand across core product lines."
                ),
                "source": "FreshStart Daily",
                "sentiment": {"label": "Positive", "score": 0.32},
            },
            {
                "title": "Regulators scrutinize {ticker} ahead of policy update",
                "content": (
                    "Regulators signaled fresh scrutiny for {ticker}, though management "
                    "believes existing compliance investments limit downside risk."
                ),
                "source": "MarketWatch",
                "sentiment": {"label": "Neutral", "score": 0.04},
            },
            {
                "title": "{ticker} suppliers flag mixed signals heading into earnings",
                "content": (
                    "Key suppliers reported softer component orders tied to {ticker}, "
                    "suggesting investors should brace for modest volatility."
                ),
                "source": "GlobalWire",
                "sentiment": {"label": "Negative", "score": -0.21},
            },
        ]

        if limit <= 0:
            return []

        articles: List[Dict] = []
        now = datetime.utcnow()
        max_items = min(limit, 50)
        for idx in range(max_items):
            template = templates[idx % len(templates)]
            articles.append(
                {
                    "title": template["title"].format(ticker=ticker.upper()),
                    "content": template["content"].format(ticker=ticker.upper()),
                    "source": template["source"],
                    "timestamp": (now.isoformat()),
                    "url": "",
                    "sentiment": template["sentiment"],
                    "ticker_sentiment": [
                        {
                            "ticker": ticker.upper(),
                            "relevance_score": "0.75",
                            "ticker_sentiment_label": template["sentiment"]["label"],
                            "ticker_sentiment_score": str(template["sentiment"]["score"]),
                        }
                    ],
                }
            )

        return articles
