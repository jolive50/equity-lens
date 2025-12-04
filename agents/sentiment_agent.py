"""Sentiment Agent for news analysis.

JOSH's Component - Sentiment Agent
Analyzes news sentiment using TAE's sentiment models.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from tools.sentiment_tools import analyze_news_articles, prioritize_news_items, summarize_sentiment

logger = logging.getLogger(__name__)


@dataclass
class SentimentAgentResult:
    """Result from SentimentAgent."""

    current: str  # "positive", "neutral", "negative"
    score: float  # 0.0-1.0
    trend: str  # "improving", "stable", "declining"
    headlines: List[str]  # Top headlines


class SentimentAgent:
    """Agent that analyzes news sentiment using Tae's real ML models."""

    def __init__(self, sentiment_model: Any = None):
        """Initialize SentimentAgent.

        Args:
            sentiment_model: Optional BaseSentimentModel instance
        """
        self.sentiment_model = sentiment_model
        if self.sentiment_model:
            try:
                model_info = self.sentiment_model.get_model_info()
                logger.info("SentimentAgent initialized with %s model", model_info.get("name", "unknown"))
                if hasattr(self.sentiment_model, "load"):
                    logger.info("Loading sentiment model weights...")
                    load_start = __import__("time").time()
                    self.sentiment_model.load()
                    logger.info("Sentiment model loaded in %.2fs", __import__("time").time() - load_start)
            except Exception:
                logger.info("SentimentAgent initialized with provided sentiment model")
        else:
            logger.warning("SentimentAgent initialized without a sentiment model (model should be supplied externally)")

    def run(
        self,
        ticker: str,
        news_data: List[Dict[str, Any]],
    ) -> SentimentAgentResult:
        """Analyze sentiment for stock news using real ML models."""
        import time

        agent_start = time.time()

        logger.info("      SentimentAgent Execution")
        logger.info("         Ticker: %s", ticker)
        if self.sentiment_model:
            try:
                model_info = self.sentiment_model.get_model_info()
                logger.info("         Model: %s v%s", model_info.get("name"), model_info.get("version", "unknown"))
            except Exception:
                logger.info("         Model: available (details unavailable)")
        logger.info("         News articles: %d", len(news_data))

        if not news_data:
            logger.warning("         No news data for %s, returning neutral sentiment", ticker)
            return SentimentAgentResult(current="neutral", score=0.5, trend="stable", headlines=[])

        if self.sentiment_model is None:
            logger.error("         No sentiment model available, returning neutral")
            headlines = [article.get("title", "") for article in news_data[:3]]
            return SentimentAgentResult(
                current="neutral",
                score=0.5,
                trend="stable",
                headlines=headlines,
            )

        sorted_news = prioritize_news_items(news_data)
        sentiment_results, headlines = analyze_news_articles(
            self.sentiment_model, sorted_news, max_articles=10, log=logger
        )

        if not sentiment_results:
            logger.warning("         No articles successfully analyzed for %s", ticker)
            return SentimentAgentResult(
                current="neutral",
                score=0.5,
                trend="stable",
                headlines=headlines[:3],
            )

        summary = summarize_sentiment(sentiment_results)
        total_time = time.time() - agent_start

        pos_count = sum(1 for r in sentiment_results if r["label"] == "positive")
        neg_count = sum(1 for r in sentiment_results if r["label"] == "negative")
        neu_count = sum(1 for r in sentiment_results if r["label"] == "neutral")

        logger.info("      Sentiment analysis complete")
        logger.info("         Sentiment: %s", summary["current"].upper())
        logger.info("         Score: %.2f (0=negative, 0.5=neutral, 1=positive)", summary["avg_score"])
        logger.info("         Trend: %s", summary["trend"])
        logger.info("         Sentiment distribution: pos=%d neg=%d neu=%d", pos_count, neg_count, neu_count)
        logger.info(
            "         Avg score: %.3f | Trend basis: first_half=%s second_half=%s",
            summary["avg_score"],
            f"{summary['first_half_avg']:.3f}" if summary["first_half_avg"] is not None else "n/a",
            f"{summary['second_half_avg']:.3f}" if summary["second_half_avg"] is not None else "n/a",
        )
        logger.info("         Total time: %.3fs", total_time)

        return SentimentAgentResult(
            current=summary["current"],
            score=summary["avg_score"],
            trend=summary["trend"],
            headlines=headlines[:3],
        )
