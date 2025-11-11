"""Cache performance metrics tracking.

Tracks cache hit/miss rates and performance metrics.
"""
import logging
from typing import Dict
from datetime import datetime

logger = logging.getLogger("freshstart.metrics")


class CacheMetrics:
    """Track cache performance metrics."""

    def __init__(self):
        self.metrics = {
            'price_hits': 0,
            'price_misses': 0,
            'news_hits': 0,
            'news_misses': 0,
            'started_at': datetime.now().isoformat()
        }

    def record_price_hit(self):
        """Record a price cache hit."""
        self.metrics['price_hits'] += 1
        logger.debug("Price cache hit recorded")

    def record_price_miss(self):
        """Record a price cache miss."""
        self.metrics['price_misses'] += 1
        logger.debug("Price cache miss recorded")

    def record_news_hit(self):
        """Record a news cache hit."""
        self.metrics['news_hits'] += 1
        logger.debug("News cache hit recorded")

    def record_news_miss(self):
        """Record a news cache miss."""
        self.metrics['news_misses'] += 1
        logger.debug("News cache miss recorded")

    def get_metrics(self) -> Dict[str, any]:
        """Get current metrics with calculated rates."""
        total_price = self.metrics['price_hits'] + self.metrics['price_misses']
        total_news = self.metrics['news_hits'] + self.metrics['news_misses']

        return {
            **self.metrics,
            'price_hit_rate': (self.metrics['price_hits'] / total_price * 100) if total_price > 0 else 0,
            'news_hit_rate': (self.metrics['news_hits'] / total_news * 100) if total_news > 0 else 0,
            'total_requests': total_price + total_news
        }

    def log_summary(self):
        """Log cache performance summary."""
        metrics = self.get_metrics()
        logger.info("=" * 60)
        logger.info("Cache Performance Summary")
        logger.info("=" * 60)
        logger.info(f"Price Cache - Hits: {metrics['price_hits']}, Misses: {metrics['price_misses']}, Hit Rate: {metrics['price_hit_rate']:.1f}%")
        logger.info(f"News Cache - Hits: {metrics['news_hits']}, Misses: {metrics['news_misses']}, Hit Rate: {metrics['news_hit_rate']:.1f}%")
        logger.info(f"Total Requests: {metrics['total_requests']}")
        logger.info(f"Started At: {metrics['started_at']}")
        logger.info("=" * 60)


# Global metrics instance
_metrics = CacheMetrics()


def get_cache_metrics() -> CacheMetrics:
    """Get the global cache metrics instance."""
    return _metrics
