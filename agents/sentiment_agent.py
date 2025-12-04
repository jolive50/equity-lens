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
        self.sentiment_model = sentiment_model
        if self.sentiment_model:
            try:
                model_info = self.sentiment_model.get_model_info()
                logger.info(f"SentimentAgent initialized with {model_info.get('name', 'unknown')} model")
                # Eagerly load weights if available to surface issues early
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

        # Use REAL sentiment analysis with Tae's new API
        logger.info(f"         → Analyzing top 10 articles...")
        sentiment_results = []
        headlines = []

        # Prioritize non-AlphaVantage articles so we don't get stuck on AV anchors
        news_data_sorted = sorted(
            news_data,
            key=lambda a: 1 if (a.get("provider") or "").lower() == "alpha_vantage" else 0
        )

        for idx, article in enumerate(news_data_sorted[:10], 1):  # Analyze top 10 articles
            title = article.get("title", "")
            body = article.get("body", "") or article.get("content", "") or article.get("summary", "")

            if not title:
                continue

            headlines.append(title)

            # Analyze using new API - predict_one() with news item
            try:
                # Prepare item for new API
                news_item = {
                    "id": article.get("id", f"article_{idx}"),
                    "title": title,
                    "body": body,
                    "word_count": article.get("word_count", len(body.split()) if body else 0),
                    "provider": article.get("provider", "unknown"),
                    "source": article.get("source", "unknown"),
                    "url": article.get("url", ""),
                    "time_published": article.get("time_published", ""),
                    "meta": article.get("meta", {}),
                    "av_meta": (article.get("av_meta") or (article.get("meta") or {}).get("av") or {}),
                }

                # Use new API
                result = self.sentiment_model.predict_one(news_item)

                # Convert model output to 3-class label and normalized score
                label_map = {
                    -2: "negative",
                    -1: "negative",
                    0: "neutral",
                    1: "positive",
                    2: "positive",
                }
                label_3class = label_map.get(int(result.label), "neutral")
                # result.score is -1..1 (pos-neg); map to 0..1
                score = max(0.0, min(1.0, (float(result.score) + 1.0) / 2.0))

                sentiment_results.append({
                    "score": score,
                    "label": label_3class,
                    "confidence": result.confidence,
                    "probabilities": result.probs
                })

                logger.debug(f"         Article {idx}: {label_3class} ({result.confidence:.2f}) - {title[:40]}...")

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
        logger.info(f"         Article scores (0-1): {[round(r['score'],3) for r in sentiment_results]}")
        logger.info(f"         Avg score: {avg_score:.3f} | Trend basis: first_half={first_half_avg if len(sentiment_results)>=4 else 'n/a'} second_half={second_half_avg if len(sentiment_results)>=4 else 'n/a'}")
        logger.info(f"         Total time: {total_time:.3f}s")

        return SentimentAgentResult(
            current=current,
            score=avg_score,
            trend=trend,
            headlines=headlines[:3]  # Top 3 headlines
        )
