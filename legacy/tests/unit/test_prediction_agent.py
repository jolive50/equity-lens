"""Test script for PredictionAgent with SHAP + GPT explanations.

WHAT: Tests PredictionAgent with both basic and GPT-powered narratives
WHY: Verify SHAP integration and GPT explanations work correctly
HOW: Create sample market data, run predictions with/without GPT
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# WHAT: Load environment variables from secrets.env
# WHY: Need OPENAI_API_KEY for GPT tests
from dotenv import load_dotenv
load_dotenv(project_root / 'secrets.env')

import logging
from langchain_openai import ChatOpenAI
from pipelines.realtime.agents import PredictionAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_market_data(days: int = 60):
    """Create sample market data for testing.

    WHAT: Generates realistic-looking OHLCV data
    WHY: Need at least 30 days for LSTM sequence
    HOW: Random walk with volume
    """
    import random
    import numpy as np

    market_data = []
    base_price = 150.0

    for i in range(days):
        # Simulate realistic price movement
        change_pct = np.random.normal(0.001, 0.02)  # 0.1% mean, 2% std
        base_price = base_price * (1 + change_pct)

        # Create daily candle
        daily_range = base_price * 0.02  # 2% daily range
        open_price = base_price + random.uniform(-daily_range/2, daily_range/2)
        close_price = base_price
        high_price = max(open_price, close_price) + random.uniform(0, daily_range/2)
        low_price = min(open_price, close_price) - random.uniform(0, daily_range/2)
        volume = int(np.random.normal(1000000, 200000))

        market_data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': max(volume, 100000)
        })

    return market_data


def test_basic_narrative():
    """Test PredictionAgent with basic narrative (no GPT)."""
    logger.info("=" * 60)
    logger.info("TEST 1: Basic Narrative (No GPT)")
    logger.info("=" * 60)

    # WHAT: Initialize agent without GPT
    agent = PredictionAgent(use_gpt_explanations=False)

    # WHAT: Create test data
    market_data = create_sample_market_data(days=60)
    fundamentals = {'pe_ratio': 28.5, 'revenue_growth': 0.15}

    # WHAT: Run prediction
    logger.info("\nRunning prediction for AAPL...")
    result = agent.run(
        ticker="AAPL",
        market_data=market_data,
        fundamentals=fundamentals
    )

    # WHAT: Display results
    logger.info("\n--- RESULTS ---")
    logger.info(f"Direction: {result.direction}")
    logger.info(f"Confidence: {result.confidence:.1%}")
    logger.info(f"\nNarrative:\n{result.narrative}")
    logger.info(f"\nTop Features:")
    for feature, importance in list(result.feature_importance.items())[:5]:
        logger.info(f"  {feature}: {importance:.3f}")

    return result


def test_gpt_narrative():
    """Test PredictionAgent with GPT-powered explanations."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: GPT-Powered Narrative with SHAP")
    logger.info("=" * 60)

    # WHAT: Check if OpenAI API key available
    import os
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set, skipping GPT test")
        logger.info("To test GPT explanations, set your API key:")
        logger.info("  export OPENAI_API_KEY='your-key-here'")
        return None

    # WHAT: Initialize agent with GPT
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    agent = PredictionAgent(
        llm=llm,
        use_gpt_explanations=True
    )

    # WHAT: Create test data
    market_data = create_sample_market_data(days=60)
    fundamentals = {'pe_ratio': 28.5, 'revenue_growth': 0.15}

    # WHAT: Run prediction
    logger.info("\nRunning prediction for AAPL with GPT explanation...")
    result = agent.run(
        ticker="AAPL",
        market_data=market_data,
        fundamentals=fundamentals
    )

    # WHAT: Display results
    logger.info("\n--- RESULTS ---")
    logger.info(f"Direction: {result.direction}")
    logger.info(f"Confidence: {result.confidence:.1%}")
    logger.info(f"\nGPT-Generated Explanation:\n{result.narrative}")
    logger.info(f"\nTop Features (from SHAP):")
    for feature, importance in list(result.feature_importance.items())[:5]:
        logger.info(f"  {feature}: {importance:.3f}")

    return result


def test_shap_availability():
    """Check if SHAP background data exists."""
    logger.info("\n" + "=" * 60)
    logger.info("CHECKING SHAP AVAILABILITY")
    logger.info("=" * 60)

    model_dir = Path("pipelines/realtime/models/saved_models")
    shap_background = model_dir / "shap_background.pkl"

    if shap_background.exists():
        logger.info(f"✓ SHAP background data found: {shap_background}")
        logger.info("  SHAP feature importance will be available")
    else:
        logger.warning(f"✗ SHAP background data NOT found: {shap_background}")
        logger.warning("  Feature importance will use placeholder values")
        logger.warning("\nTo enable SHAP, run:")
        logger.warning("  python pipelines/realtime/models/train_lstm_forecaster.py")

    return shap_background.exists()


if __name__ == "__main__":
    try:
        # WHAT: Check SHAP availability first
        shap_available = test_shap_availability()

        # WHAT: Run basic test
        basic_result = test_basic_narrative()

        # WHAT: Run GPT test if API key available
        gpt_result = test_gpt_narrative()

        # WHAT: Summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✓ Basic narrative: Working")
        if gpt_result:
            logger.info(f"✓ GPT narrative: Working")
        else:
            logger.info(f"⊘ GPT narrative: Skipped (no API key)")

        if shap_available:
            logger.info(f"✓ SHAP explainer: Available")
        else:
            logger.info(f"✗ SHAP explainer: Not available (retrain model)")

        logger.info("\n✓ All tests completed successfully!")

    except Exception as e:
        logger.error(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
