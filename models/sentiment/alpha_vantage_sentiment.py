import os
import requests
from typing import Dict, Any, List, Optional
from .base_sentiment import BaseSentimentModel, SentimentResult


class AlphaVantageSentiment(BaseSentimentModel):
    """Alpha Vantage sentiment API wrapper.

    Note: This is an API service, not an ML model.
    Wraps Alpha Vantage NEWS_SENTIMENT API to provide consistent interface.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Alpha Vantage sentiment API wrapper.

        Args:
            api_key: Alpha Vantage API key (defaults to ALPHA_VANTAGE_API_KEY env var)

        Raises:
            ValueError: If API key not provided
        """
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Alpha Vantage API key required")

        self.base_url = "https://www.alphavantage.co/query"

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment using Alpha Vantage API.

        Note: Alpha Vantage doesn't support arbitrary text analysis.
        This method raises NotImplementedError.
        Use analyze_ticker_sentiment() instead for ticker-specific sentiment.

        Args:
            text: Text to analyze (not supported)

        Raises:
            NotImplementedError: Alpha Vantage API doesn't support arbitrary text
        """
        raise NotImplementedError(
            "Alpha Vantage API doesn't support arbitrary text analysis. "
            "Use analyze_ticker_sentiment() for ticker-specific sentiment from news."
        )

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Batch analysis not supported by Alpha Vantage API.

        Raises:
            NotImplementedError: Alpha Vantage doesn't support batch text analysis
        """
        raise NotImplementedError(
            "Alpha Vantage API doesn't support batch text analysis. "
            "Use analyze_ticker_sentiment() for ticker-specific sentiment."
        )

    def analyze_ticker_sentiment(self, ticker: str) -> SentimentResult:
        """Get overall sentiment for a ticker from Alpha Vantage news.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")

        Returns:
            SentimentResult with aggregated sentiment

        Raises:
            RuntimeError: If API call fails
        """
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": self.api_key,
            "limit": 50
        }

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

            # Aggregate sentiment from news feed
            sentiments = []
            for article in data["feed"]:
                ticker_sentiments = article.get("ticker_sentiment", [])
                for ts in ticker_sentiments:
                    if ts.get("ticker") == ticker:
                        score = float(ts.get("ticker_sentiment_score", 0.0))
                        sentiments.append(score)

            if not sentiments:
                # No ticker-specific sentiment found, use overall article sentiment
                for article in data["feed"]:
                    score = float(article.get("overall_sentiment_score", 0.0))
                    sentiments.append(score)

            if not sentiments:
                raise RuntimeError(f"No sentiment data found for {ticker}")

            # Calculate average sentiment score
            avg_score = sum(sentiments) / len(sentiments)

            # Convert score to label
            # Alpha Vantage: -1 (bearish) to 1 (bullish)
            if avg_score > 0.15:
                label = "positive"
                confidence = min(1.0, (avg_score + 1.0) / 2.0)  # Normalize to 0-1
            elif avg_score < -0.15:
                label = "negative"
                confidence = min(1.0, (1.0 - avg_score) / 2.0)  # Normalize to 0-1
            else:
                label = "neutral"
                confidence = 1.0 - abs(avg_score)  # Neutral confidence

            # Create probability distribution
            pos_prob = max(0.0, min(1.0, (avg_score + 1.0) / 2.0))
            neg_prob = max(0.0, min(1.0, (1.0 - avg_score) / 2.0))
            neu_prob = 1.0 - pos_prob - neg_prob
            neu_prob = max(0.0, neu_prob)  # Ensure non-negative

            # Normalize to sum to 1.0
            total = pos_prob + neg_prob + neu_prob
            if total > 0:
                pos_prob /= total
                neg_prob /= total
                neu_prob /= total

            return SentimentResult(
                label=label,
                confidence=confidence,
                probabilities={
                    "positive": float(pos_prob),
                    "negative": float(neg_prob),
                    "neutral": float(neu_prob)
                },
                metadata={
                    "model": "AlphaVantage API",
                    "ticker": ticker,
                    "average_score": avg_score,
                    "num_articles": len(sentiments),
                    "source": "NEWS_SENTIMENT API"
                }
            )

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Alpha Vantage API call failed: {e}") from e

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "name": "AlphaVantage",
            "version": "NEWS_SENTIMENT API",
            "type": "api",
            "capabilities": ["analyze_ticker_sentiment"],
            "fine_tuned": False,
            "note": "API service, not ML model"
        }
