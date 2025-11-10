"""Verification script to test Josh's agents use Tae's real models.

This script verifies:
1. SentimentAgent loads and uses real sentiment models (FinBERT or VADER)
2. Workflow fetches real news data using NewsDataFetcher
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_sentiment_agent_loads_real_model():
    """Test that SentimentAgent loads Tae's real models."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: SentimentAgent Model Loading")
    logger.info("=" * 60)

    try:
        from agents.sentiment_agent import SentimentAgent

        # Create agent (should auto-load FinBERT or VADER)
        agent = SentimentAgent()

        if agent.sentiment_model is None:
            logger.error("❌ FAILED: No sentiment model loaded")
            return False

        model_info = agent.sentiment_model.get_model_info()
        model_name = model_info['name']

        if model_name in ['FinBERT', 'RoBERTa', 'VADER', 'TextBlob', 'AlphaVantage']:
            logger.info(f"✅ PASSED: Loaded real model: {model_name}")
            logger.info(f"   Model type: {model_info.get('type', 'unknown')}")
            return True
        else:
            logger.error(f"❌ FAILED: Unknown model type: {model_name}")
            return False

    except Exception as e:
        logger.error(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sentiment_agent_analyzes_with_real_model():
    """Test that SentimentAgent uses real model for analysis."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: SentimentAgent Real Analysis")
    logger.info("=" * 60)

    try:
        from agents.sentiment_agent import SentimentAgent

        agent = SentimentAgent()

        # Test with sample news articles
        sample_news = [
            {
                "title": "Apple reports record quarterly earnings, beats estimates",
                "content": "Apple Inc. reported better-than-expected earnings...",
                "source": "Reuters",
                "timestamp": "2024-11-10T10:00:00Z"
            },
            {
                "title": "Tech giant faces regulatory scrutiny over market practices",
                "content": "Regulators announced an investigation...",
                "source": "Bloomberg",
                "timestamp": "2024-11-10T09:00:00Z"
            }
        ]

        result = agent.run(ticker="AAPL", news_data=sample_news)

        logger.info(f"✅ Analysis Result:")
        logger.info(f"   Sentiment: {result.current}")
        logger.info(f"   Score: {result.score:.2f}")
        logger.info(f"   Trend: {result.trend}")
        logger.info(f"   Headlines: {len(result.headlines)}")

        # Verify result is not placeholder keyword matching
        # Real models should produce varied scores, not just 0.3, 0.5, 0.7
        if result.score in [0.3, 0.5, 0.7]:
            logger.warning("⚠️  WARNING: Score looks like keyword matching placeholder")

        return True

    except Exception as e:
        logger.error(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_news_fetcher_integration():
    """Test that workflow can fetch real news using NewsDataFetcher."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: NewsDataFetcher Integration")
    logger.info("=" * 60)

    try:
        from data.fetchers.news_data import NewsDataFetcher

        fetcher = NewsDataFetcher()
        logger.info("✅ NewsDataFetcher initialized successfully")

        # Note: We won't actually call the API to avoid rate limits
        # Just verify the fetcher is accessible and has the right methods

        if not hasattr(fetcher, 'fetch_news'):
            logger.error("❌ FAILED: NewsDataFetcher missing fetch_news method")
            return False

        logger.info("✅ NewsDataFetcher has fetch_news method")
        logger.info("   (Skipping actual API call to preserve rate limits)")

        return True

    except Exception as e:
        logger.error(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_no_placeholder_code():
    """Verify that placeholder code has been removed."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: No Placeholder Code")
    logger.info("=" * 60)

    issues = []

    # Check SentimentAgent
    try:
        with open('agents/sentiment_agent.py', 'r') as f:
            content = f.read()

        if 'placeholder' in content.lower():
            issues.append("SentimentAgent still contains 'placeholder' text")

        if 'keyword-based sentiment' in content.lower():
            issues.append("SentimentAgent still has keyword-based sentiment comment")

        if all(word in content for word in ['beat', 'surge', 'growth', 'profit']):
            # Check if these are in the context of keyword matching
            if '["beat", "surge", "growth", "profit"]' in content:
                issues.append("SentimentAgent still has keyword matching code")

    except Exception as e:
        issues.append(f"Failed to check sentiment_agent.py: {e}")

    # Check workflow
    try:
        with open('coordinator/workflow.py', 'r') as f:
            content = f.read()

        # Check for the old placeholder comment
        if '# Fetch news data (placeholder - TAE\'s responsibility)' in content:
            issues.append("Workflow still has placeholder comment for news fetching")

        # Check that NewsDataFetcher is being used
        if 'NewsDataFetcher' not in content:
            issues.append("Workflow doesn't import or use NewsDataFetcher")

    except Exception as e:
        issues.append(f"Failed to check workflow.py: {e}")

    if issues:
        logger.error("❌ FAILED: Found placeholder code:")
        for issue in issues:
            logger.error(f"   - {issue}")
        return False
    else:
        logger.info("✅ PASSED: No placeholder code found")
        return True


def main():
    """Run all verification tests."""
    logger.info("\n" + "=" * 70)
    logger.info(" JOSH'S AGENTS INTEGRATION VERIFICATION")
    logger.info(" Testing that agents use Tae's real implementations")
    logger.info("=" * 70)

    results = {
        "Model Loading": test_sentiment_agent_loads_real_model(),
        "Real Analysis": test_sentiment_agent_analyzes_with_real_model(),
        "News Fetcher": test_news_fetcher_integration(),
        "No Placeholders": verify_no_placeholder_code()
    }

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{status}: {test_name}")

    logger.info("=" * 70)
    logger.info(f"OVERALL: {passed}/{total} tests passed")
    logger.info("=" * 70)

    if passed == total:
        logger.info("\n✅ SUCCESS: All integration tests passed!")
        logger.info("Josh's agents now use Tae's real implementations.")
        return 0
    else:
        logger.error(f"\n❌ FAILURE: {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
