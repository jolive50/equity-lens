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
    """Agent that analyzes news sentiment using Tae's real ML models."""

    def __init__(self, sentiment_model=None):
        """Initialize SentimentAgent.

        Args:
            sentiment_model: Optional BaseSentimentModel instance
        """
        if sentiment_model is None:
            # Use Tae's FinBERT model as default (best for financial sentiment)
            try:
                from models.sentiment.finbert_model import FinBERTModel
                self.sentiment_model = FinBERTModel()
                logger.info("SentimentAgent initialized with FinBERT model")
            except Exception as e:
                logger.warning(f"Failed to load FinBERT, trying VADER fallback: {e}")
                try:
                    from models.sentiment.vader_model import VADERModel
                    self.sentiment_model = VADERModel()
                    logger.info("SentimentAgent initialized with VADER model")
                except Exception as e2:
                    logger.error(f"Failed to load any sentiment model: {e2}")
                    self.sentiment_model = None
        else:
            self.sentiment_model = sentiment_model
            logger.info(f"SentimentAgent initialized with {sentiment_model.get_model_info()['name']}")

    def run(
        self,
        ticker: str,
        news_data: List[Dict[str, Any]]
    ) -> SentimentAgentResult:
        """Analyze sentiment for stock news using real ML models.

        Args:
            ticker: Stock symbol
            news_data: List of news article dicts

        Returns:
            SentimentAgentResult with sentiment analysis
        """
        import time
        agent_start = time.time()

        logger.info("      ╔══════════════════════════════════════════════════════╗")
        logger.info("      ║  💬 SentimentAgent Execution                       ║")
        logger.info("      ╚══════════════════════════════════════════════════════╝")
        logger.info(f"         Ticker: {ticker}")
        if self.sentiment_model:
            model_info = self.sentiment_model.get_model_info()
            logger.info(f"         Model: {model_info['name']} v{model_info['version']}")
        logger.info(f"         News articles: {len(news_data)}")

        if not news_data:
            logger.warning(f"         ⚠️  No news data for {ticker}, returning neutral sentiment")
            return SentimentAgentResult(
                current="neutral",
                score=0.5,
                trend="stable",
                headlines=[]
            )

        if self.sentiment_model is None:
            logger.error("         ❌ No sentiment model available, returning neutral")
            return SentimentAgentResult(
                current="neutral",
                score=0.5,
                trend="stable",
                headlines=[article.get("title", "") for article in news_data[:3]]
            )

        # Use REAL sentiment analysis with Tae's models
        logger.info(f"         → Analyzing top 10 articles...")
        sentiment_results = []
        headlines = []

        for idx, article in enumerate(news_data[:10], 1):  # Analyze top 10 articles
            title = article.get("title", "")
            content = article.get("content", "") or article.get("summary", "")

            if not title and not content:
                continue

            headlines.append(title)

            # Analyze using REAL ML model (FinBERT or VADER)
            try:
                # Combine title and content for better analysis
                text_to_analyze = f"{title}. {content[:200]}" if content else title

                result = self.sentiment_model.analyze(text_to_analyze)

                # Convert sentiment label to score
                if result.label == "positive":
                    score = 0.5 + (result.confidence * 0.5)  # 0.5-1.0
                elif result.label == "negative":
                    score = 0.5 - (result.confidence * 0.5)  # 0.0-0.5
                else:  # neutral
                    score = 0.5

                sentiment_results.append({
                    "score": score,
                    "label": result.label,
                    "confidence": result.confidence,
                    "probabilities": result.probabilities
                })

                logger.debug(f"         Article {idx}: {result.label} ({result.confidence:.2f}) - {title[:40]}...")

            except Exception as e:
                logger.warning(f"         ⚠️  Failed to analyze article {idx}: {e}")
                continue

        if not sentiment_results:
            logger.warning(f"         ⚠️  No articles successfully analyzed for {ticker}")
            return SentimentAgentResult(
                current="neutral",
                score=0.5,
                trend="stable",
                headlines=headlines[:3]
            )

        # Calculate average sentiment score
        avg_score = sum(r["score"] for r in sentiment_results) / len(sentiment_results)

        # Determine sentiment category
        if avg_score > 0.6:
            current = "positive"
        elif avg_score < 0.4:
            current = "negative"
        else:
            current = "neutral"

        # Calculate trend (compare first half vs second half)
        if len(sentiment_results) >= 4:
            mid = len(sentiment_results) // 2
            first_half_avg = sum(r["score"] for r in sentiment_results[:mid]) / mid
            second_half_avg = sum(r["score"] for r in sentiment_results[mid:]) / (len(sentiment_results) - mid)

            if second_half_avg - first_half_avg > 0.1:
                trend = "improving"
            elif first_half_avg - second_half_avg > 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        total_time = time.time() - agent_start

        logger.info("      ┌──────────────────────────────────────────────────┐")
        logger.info("      │  💬 Sentiment Result                             │")
        logger.info("      └──────────────────────────────────────────────────┘")
        logger.info(f"         Articles analyzed: {len(sentiment_results)}/{len(news_data[:10])}")
        logger.info(f"         Sentiment: {current.upper()}")
        logger.info(f"         Score: {avg_score:.2f} (0=negative, 0.5=neutral, 1=positive)")
        logger.info(f"         Trend: {trend}")
        logger.info(f"         Sentiment distribution:")
        pos_count = sum(1 for r in sentiment_results if r["label"] == "positive")
        neg_count = sum(1 for r in sentiment_results if r["label"] == "negative")
        neu_count = sum(1 for r in sentiment_results if r["label"] == "neutral")
        logger.info(f"            Positive: {pos_count}/{len(sentiment_results)}")
        logger.info(f"            Negative: {neg_count}/{len(sentiment_results)}")
        logger.info(f"            Neutral:  {neu_count}/{len(sentiment_results)}")
        logger.info(f"         Total time: {total_time:.3f}s")

        return SentimentAgentResult(
            current=current,
            score=avg_score,
            trend=trend,
            headlines=headlines[:3]  # Top 3 headlines
        )
