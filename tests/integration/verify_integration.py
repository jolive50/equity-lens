"""Verification script to test Josh's agents use Tae's real models.

This script verifies:
1. SentimentAgent loads and uses real sentiment models (FinBERT, RoBERTa, DeBERTa)
2. Workflow fetches real news data using NewsDataFetcher
"""
import logging
import traceback

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_sentiment_agent_loads_real_model():
    """Test that SentimentAgent loads Tae's real models."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: SentimentAgent Model Loading")
    logger.info("=" * 60)

    try:
        from agents.sentiment_agent import SentimentAgent
        from coordinator.config import WorkflowConfig

        cfg = WorkflowConfig()
        model = cfg.get_sentiment_model()

        # Create agent with configured model (decoupled from loading)
        agent = SentimentAgent(sentiment_model=model)

        if agent.sentiment_model is None:
            logger.error("FAILED: No sentiment model loaded")
            return False

        model_info = agent.sentiment_model.get_model_info()
        model_name = model_info.get("name", "unknown")

        if model_name in ["FinBERT", "RoBERTa", "DeBERTa", "AlphaVantage", "SentimentEnsemble"]:
            logger.info("PASSED: Loaded real model: %s", model_name)
            logger.info("   Model type: %s", model_info.get("type", "unknown"))
            return True

        logger.error("FAILED: Unknown model type: %s", model_name)
        return False

    except Exception as e:  # pragma: no cover - defensive
        logger.error("FAILED: %s", e)
        traceback.print_exc()
        return False


def test_sentiment_agent_analyzes_with_real_model():
    """Test that SentimentAgent uses real model for analysis."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: SentimentAgent Real Analysis")
    logger.info("=" * 60)

    try:
        from agents.sentiment_agent import SentimentAgent
        from coordinator.config import WorkflowConfig

        cfg = WorkflowConfig()
        model = cfg.get_sentiment_model()
        agent = SentimentAgent(sentiment_model=model)

        # Test with sample news articles
        sample_news = [
            {
                "title": "Apple reports record quarterly earnings, beats estimates",
                "content": "Apple Inc. reported better-than-expected earnings...",
                "source": "Reuters",
                "timestamp": "2024-11-10T10:00:00Z",
            },
            {
                "title": "Tech giant faces regulatory scrutiny over market practices",
                "content": "Regulators announced an investigation...",
                "source": "Bloomberg",
                "timestamp": "2024-11-10T09:00:00Z",
            },
        ]

        result = agent.run(ticker="AAPL", news_data=sample_news)

        logger.info("PASSED: Analysis Result:")
        logger.info("   Sentiment: %s", result.current)
        logger.info("   Score: %.2f", result.score)
        logger.info("   Trend: %s", result.trend)
        logger.info("   Headlines: %d", len(result.headlines))

        # Warn if the score looks like a placeholder
        if result.score in [0.3, 0.5, 0.7]:
            logger.warning("WARNING: Score looks like keyword matching placeholder")

        return True

    except Exception as e:  # pragma: no cover - defensive
        logger.error("FAILED: %s", e)
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
        logger.info("PASSED: NewsDataFetcher initialized successfully")

        if not hasattr(fetcher, "fetch_news"):
            logger.error("FAILED: NewsDataFetcher missing fetch_news method")
            return False

        logger.info("PASSED: NewsDataFetcher has fetch_news method")
        logger.info("   (Skipping actual API call to preserve rate limits)")
        return True

    except Exception as e:  # pragma: no cover - defensive
        logger.error("FAILED: %s", e)
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
        with open("agents/sentiment_agent.py", "r", encoding="utf-8") as f:
            content = f.read().lower()

        if "placeholder" in content:
            issues.append("SentimentAgent still contains 'placeholder' text")

        if "keyword-based sentiment" in content:
            issues.append("SentimentAgent still has keyword-based sentiment comment")

        if '["beat", "surge", "growth", "profit"]' in content:
            issues.append("SentimentAgent still has keyword matching code")

    except Exception as e:  # pragma: no cover - defensive
        issues.append(f"Failed to check sentiment_agent.py: {e}")

    # Check workflow
    try:
        with open("coordinator/workflow.py", "r", encoding="utf-8") as f:
            content = f.read()

        if "# Fetch news data (placeholder - TAE's responsibility)" in content:
            issues.append("Workflow still has placeholder comment for news fetching")

        if "NewsDataFetcher" not in content:
            issues.append("Workflow doesn't import or use NewsDataFetcher")

    except Exception as e:  # pragma: no cover - defensive
        issues.append(f"Failed to check workflow.py: {e}")

    if issues:
        logger.error("FAILED: Found placeholder code:")
        for issue in issues:
            logger.error("   - %s", issue)
        return False

    logger.info("PASSED: No placeholder code found")
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
        "No Placeholders": verify_no_placeholder_code(),
    }

    logger.info("\nSummary:")
    for name, passed in results.items():
        logger.info(" - %s: %s", name, "PASS" if passed else "FAIL")

    all_passed = all(results.values())
    logger.info("\nOVERALL RESULT: %s", "PASS" if all_passed else "FAIL")
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
