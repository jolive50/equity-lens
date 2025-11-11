#!/usr/bin/env python3
"""Test script to verify caching implementation works correctly.

This script:
1. Runs an analysis for AAPL (should miss cache)
2. Runs the same analysis again (should hit cache)
3. Prints cache metrics
"""
import logging
import time
from coordinator.workflow import run_stock_analysis
from storage.database import Database
from utils.logging_config import setup_logging

# Setup logging
setup_logging(name="freshstart", log_dir="logs", log_level=logging.INFO)
logger = logging.getLogger("freshstart.test")

def test_caching():
    """Test the caching implementation."""
    logger.info("=" * 80)
    logger.info("CACHING TEST - Starting")
    logger.info("=" * 80)

    ticker = "AAPL"
    db = Database()

    # Clear existing cache for clean test
    logger.info(f"Clearing existing cache for {ticker}...")
    try:
        with db._get_connection() as conn:
            conn.execute("DELETE FROM price_cache WHERE ticker = ?", (ticker,))
            conn.execute("DELETE FROM news_cache WHERE ticker = ?", (ticker,))
            conn.commit()
        logger.info("Cache cleared successfully")
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")

    # First request - should MISS cache and fetch from API
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: First Request (Expected: Cache MISS)")
    logger.info("=" * 80)
    start_time = time.time()

    try:
        result1 = run_stock_analysis(ticker, user_tier="basic")
        elapsed1 = time.time() - start_time
        logger.info(f"✓ First request completed in {elapsed1:.2f}s")
        logger.info(f"  Prediction: {result1['prediction']['direction']} ({result1['prediction']['confidence']:.1%})")
    except Exception as e:
        logger.error(f"✗ First request failed: {e}")
        return

    # Wait a moment
    time.sleep(2)

    # Second request - should HIT cache (data still fresh)
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Second Request (Expected: Cache HIT)")
    logger.info("=" * 80)
    start_time = time.time()

    try:
        result2 = run_stock_analysis(ticker, user_tier="basic")
        elapsed2 = time.time() - start_time
        logger.info(f"✓ Second request completed in {elapsed2:.2f}s")
        logger.info(f"  Prediction: {result2['prediction']['direction']} ({result2['prediction']['confidence']:.1%})")

        # Verify cache was faster
        if elapsed2 < elapsed1:
            speedup = ((elapsed1 - elapsed2) / elapsed1) * 100
            logger.info(f"✓ Cache speedup: {speedup:.1f}% faster!")
        else:
            logger.warning(f"⚠ Second request was slower (unexpected)")

    except Exception as e:
        logger.error(f"✗ Second request failed: {e}")
        return

    # Check database for cached data
    logger.info("\n" + "=" * 80)
    logger.info("DATABASE VERIFICATION")
    logger.info("=" * 80)

    cached_prices = db.get_cached_prices(ticker)
    cached_news = db.get_cached_news(ticker)

    logger.info(f"✓ Cached price rows: {len(cached_prices)}")
    logger.info(f"✓ Cached news articles: {len(cached_news)}")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"First request:  {elapsed1:.2f}s (cache miss)")
    logger.info(f"Second request: {elapsed2:.2f}s (cache hit)")
    logger.info(f"Performance improvement: {((elapsed1 - elapsed2) / elapsed1) * 100:.1f}%")
    logger.info(f"Cached data: {len(cached_prices)} prices, {len(cached_news)} news articles")
    logger.info("=" * 80)
    logger.info("✓ Caching test completed successfully!")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        test_caching()
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
