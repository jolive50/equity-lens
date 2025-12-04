"""Utilities for running sentiment models outside of agent classes."""

from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def prioritize_news_items(news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Order news items so non-AlphaVantage articles are handled first."""
    return sorted(
        news_data,
        key=lambda article: 1 if (article.get("provider") or "").lower() == "alpha_vantage" else 0,
    )


def analyze_news_articles(
    sentiment_model: Any,
    news_data: List[Dict[str, Any]],
    max_articles: int = 10,
    log: Optional[logging.Logger] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Run the sentiment model across a batch of news articles."""
    log = log or logger
    sentiment_results: List[Dict[str, Any]] = []
    headlines: List[str] = []

    for idx, article in enumerate(news_data[:max_articles], 1):
        title = article.get("title", "")
        body = article.get("body", "") or article.get("content", "") or article.get("summary", "")

        if not title:
            continue

        headlines.append(title)

        try:
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

            result = sentiment_model.predict_one(news_item)

            label_map = {-2: "negative", -1: "negative", 0: "neutral", 1: "positive", 2: "positive"}
            label_3class = label_map.get(int(result.label), "neutral")
            score = max(0.0, min(1.0, (float(result.score) + 1.0) / 2.0))

            sentiment_results.append(
                {
                    "score": score,
                    "label": label_3class,
                    "confidence": result.confidence,
                    "probabilities": result.probs,
                }
            )

            log.debug("Article %s: %s (%.2f) - %s", idx, label_3class, result.confidence, title[:40])
        except Exception as exc:
            log.warning("Failed to analyze article %s: %s", idx, exc)
            continue

    return sentiment_results, headlines


def summarize_sentiment(sentiment_results: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Aggregate per-article sentiment into overall label, score, and trend."""
    avg_score = sum(r["score"] for r in sentiment_results) / len(sentiment_results)

    if avg_score > 0.6:
        current = "positive"
    elif avg_score < 0.4:
        current = "negative"
    else:
        current = "neutral"

    first_half_avg: Optional[float] = None
    second_half_avg: Optional[float] = None
    trend = "stable"

    if len(sentiment_results) >= 4:
        mid = len(sentiment_results) // 2
        first_half_avg = sum(r["score"] for r in sentiment_results[:mid]) / mid
        second_half_avg = sum(r["score"] for r in sentiment_results[mid:]) / (len(sentiment_results) - mid)

        if second_half_avg - first_half_avg > 0.1:
            trend = "improving"
        elif first_half_avg - second_half_avg > 0.1:
            trend = "declining"

    return {
        "current": current,
        "avg_score": avg_score,
        "trend": trend,
        "first_half_avg": first_half_avg,
        "second_half_avg": second_half_avg,
    }

