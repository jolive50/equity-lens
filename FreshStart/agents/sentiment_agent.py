"""Sentiment agent for news analysis."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Any
from langchain_core.runnables import Runnable


logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Structured output from sentiment agent.

    Attributes:
        current: Overall sentiment (positive/neutral/negative)
        score: Sentiment strength (0.0-1.0)
        trend: Sentiment trend direction
        headlines: Top 3 relevant news headlines
    """
    current: Literal["positive", "neutral", "negative"]
    score: float
    trend: Literal["improving", "stable", "declining"]
    headlines: List[str]


class SentimentAgent:
    """Analyzes news sentiment using FinBERT.

    Uses FinBERT from Tae's models/sentiment/ for financial sentiment analysis.
    Integrates with ChromaDB VectorStore for semantic search.
    """

    def __init__(
        self,
        llm: Optional[Runnable] = None,
        *,
        use_finbert: bool = True,
        vector_store: Optional[Any] = None
    ) -> None:
        """Initialize sentiment agent.

        Args:
            llm: (Unused) Retained for compatibility
            use_finbert: Whether to use FinBERT (must be True)
            vector_store: Optional VectorStore for semantic search

        Raises:
            RuntimeError: If FinBERT dependencies missing or initialization fails
        """
        if llm is not None:
            logger.warning("SentimentAgent ignores provided LLM")

        if not use_finbert:
            logger.warning("SentimentAgent requires FinBERT")

        try:
            from models.sentiment.finbert import create_sentiment_analyzer
        except ImportError as exc:
            raise RuntimeError(
                "FinBERT dependencies missing. Install tensorflow/transformers."
            ) from exc

        self.sentiment_analyzer = create_sentiment_analyzer(
            vector_store=vector_store,
            enable_similarity_search=True
        )
        if self.sentiment_analyzer is None:
            raise RuntimeError("create_sentiment_analyzer returned None")

        self.vector_store = vector_store

    def run(
        self,
        *,
        ticker: str,
        news_data: List[Dict]
    ) -> SentimentResult:
        """Analyze sentiment for a stock's news.

        Args:
            ticker: Stock symbol
            news_data: List of news article dictionaries

        Returns:
            SentimentResult with current sentiment, score, trend, headlines

        Raises:
            ValueError: If news_data empty
            RuntimeError: If FinBERT analysis fails
        """
        if not news_data:
            raise ValueError("news_data must contain at least one article")

        try:
            finbert_result = self.sentiment_analyzer.process_news_articles(
                news_data,
                ticker=ticker
            )
        except Exception as exc:
            raise RuntimeError(f"FinBERT analysis failed for {ticker}: {exc}") from exc

        required_keys = {"current", "score", "trend", "headlines"}
        if not isinstance(finbert_result, dict) or not required_keys.issubset(finbert_result.keys()):
            raise RuntimeError("FinBERT returned unexpected response structure")

        allowed_sentiments = {"positive", "neutral", "negative"}
        allowed_trends = {"improving", "stable", "declining"}
        current = str(finbert_result["current"]).lower()
        trend = str(finbert_result["trend"]).lower()
        current = current if current in allowed_sentiments else "neutral"
        trend = trend if trend in allowed_trends else "stable"
        score = float(finbert_result["score"])
        headlines = list(finbert_result.get("headlines", []))[:3]

        from typing import cast
        current_literal = cast(Literal["positive", "neutral", "negative"], current)
        trend_literal = cast(Literal["improving", "stable", "declining"], trend)

        return SentimentResult(
            current=current_literal,
            score=score,
            trend=trend_literal,
            headlines=headlines
        )
