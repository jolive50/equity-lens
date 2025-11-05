"""Sentiment Agent for news analysis.

JOSH's Component - Sentiment Agent
Analyzes news sentiment using TAE's sentiment models.
"""
import logging
from typing import Dict, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SentimentAgentResult:
    """Result from SentimentAgent."""
    current: str  # "positive", "neutral", "negative"
    score: float  # 0.0-1.0
    trend: str  # "improving", "stable", "declining"
    headlines: List[str]  # Top headlines


class SentimentAgent:
    """Agent that analyzes news sentiment."""

    def __init__(self, sentiment_model=None):
        """Initialize SentimentAgent.

        Args:
            sentiment_model: Optional BaseSentimentModel instance
        """
        self.sentiment_model = sentiment_model
        logger.info("SentimentAgent initialized")

    def run(
        self,
        ticker: str,
        news_data: List[Dict[str, Any]]
    ) -> SentimentAgentResult:
        """Analyze sentiment for stock news.

        Args:
            ticker: Stock symbol
            news_data: List of news article dicts

        Returns:
            SentimentAgentResult with sentiment analysis
        """
        if not news_data:
            logger.warning(f"No news data for {ticker}, returning neutral sentiment")
            return SentimentAgentResult(
                current="neutral",
                score=0.5,
                trend="stable",
                headlines=[]
            )

        # Simple sentiment analysis (placeholder for TAE's models)
        sentiment_scores = []
        headlines = []

        for article in news_data[:10]:  # Analyze top 10 articles
            title = article.get("title", "")
            headlines.append(title)

            # Simple keyword-based sentiment (will be replaced by TAE's FinBERT)
            title_lower = title.lower()
            if any(word in title_lower for word in ["beat", "surge", "growth", "profit"]):
                sentiment_scores.append(0.7)
            elif any(word in title_lower for word in ["miss", "decline", "loss", "weak"]):
                sentiment_scores.append(0.3)
            else:
                sentiment_scores.append(0.5)

        # Calculate average sentiment
        avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5

        # Determine sentiment category
        if avg_score > 0.6:
            current = "positive"
        elif avg_score < 0.4:
            current = "negative"
        else:
            current = "neutral"

        # Determine trend (placeholder logic)
        trend = "stable"

        return SentimentAgentResult(
            current=current,
            score=avg_score,
            trend=trend,
            headlines=headlines[:3]  # Top 3 headlines
        )
