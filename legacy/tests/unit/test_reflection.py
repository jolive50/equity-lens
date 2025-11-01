"""Unit tests for ReflectionAgent quality assurance functionality.

This test suite validates the ReflectionAgent's ability to:
- Detect invalid predictions (confidence out of range, bad direction)
- Identify sentiment-prediction misalignments
- Flag stale market data
- Catch missing fundamental metrics
- Calculate appropriate confidence adjustments
- Generate actionable recommendations

WHAT: Comprehensive test coverage for all ReflectionAgent validation checks
WHY: Ensures quality control logic works correctly before production use
HOW: Pytest-based unit tests with fixtures for common test data
DATA: Mock agent outputs → validation results → assertions on expected behavior
"""
from datetime import datetime, timedelta

import pytest

from pipelines.realtime.reflection import ReflectionAgent


# ===== FIXTURES =====
# Reusable test data for common scenarios


@pytest.fixture
def valid_prediction():
    """
    Valid prediction output for testing.

    WHAT: Creates a realistic PredictionAgent output dictionary
    WHY: Many tests need a baseline valid prediction
    HOW: Returns dict with valid direction, confidence, narrative
    DATA: Standard prediction format expected by ReflectionAgent
    """
    return {
        "direction": "up",
        "confidence": 0.85,
        "narrative": "Gradient boosting model indicates upward movement with 85% confidence.",
    }


@pytest.fixture
def valid_sentiment():
    """
    Valid sentiment output for testing.

    WHAT: Creates a realistic SentimentAgent output dictionary
    WHY: Many tests need a baseline valid sentiment
    HOW: Returns dict with valid current, score, trend, headlines
    DATA: Standard sentiment format expected by ReflectionAgent
    """
    return {
        "current": "positive",
        "score": 0.75,
        "trend": "improving",
        "headlines": [
            "Company beats earnings expectations",
            "Strong revenue growth reported",
            "Analyst upgrades stock rating",
        ],
    }


@pytest.fixture
def valid_filing():
    """
    Valid filing/smart money output for testing.

    WHAT: Creates a realistic SmartMoneyAgent output dictionary
    WHY: Tests need a baseline valid filing data
    HOW: Returns dict with institutions, insiders, congress summaries
    DATA: Standard filing format expected by ReflectionAgent
    """
    return {
        "institutions": {"summary": "Institutional ownership increased 3%"},
        "insiders": {"summary": "No significant insider transactions"},
        "congress": {"summary": "No congressional trades reported"},
    }


@pytest.fixture
def valid_market_data():
    """
    Valid market data (fresh) for testing.

    WHAT: Creates realistic historical price data with recent timestamp
    WHY: Tests need baseline fresh market data
    HOW: Returns list of dicts with today's date
    DATA: List of OHLCV records with fresh timestamps
    """
    # WHAT: Use today's date to ensure data is fresh
    # WHY: Fresh data should pass validation
    # HOW: Format as ISO string (YYYY-MM-DD)
    # DATA: Today's date in ISO format
    today = datetime.now().date().isoformat()

    return [
        {"date": today, "open": 148.0, "high": 151.0, "low": 147.5, "close": 150.0, "volume": 1000000},
        {"date": today, "open": 150.0, "high": 152.0, "low": 149.0, "close": 151.5, "volume": 1100000},
    ]


@pytest.fixture
def stale_market_data():
    """
    Stale market data (>3 days old) for testing.

    WHAT: Creates historical price data with old timestamp
    WHY: Tests need baseline stale data to validate freshness check
    HOW: Returns list of dicts with date from 5 days ago
    DATA: List of OHLCV records with stale timestamps
    """
    # WHAT: Use date from 5 days ago to ensure data is stale
    # WHY: Data >3 days old should fail freshness validation
    # HOW: Subtract 5 days from today, format as ISO string
    # DATA: Old date in ISO format
    old_date = (datetime.now() - timedelta(days=5)).date().isoformat()

    return [
        {"date": old_date, "open": 148.0, "high": 151.0, "low": 147.5, "close": 150.0, "volume": 1000000},
    ]


@pytest.fixture
def valid_fundamentals():
    """
    Valid fundamentals with all required metrics.

    WHAT: Creates complete financial metrics dictionary
    WHY: Tests need baseline complete fundamentals
    HOW: Returns dict with all required keys (pe_ratio, revenue_growth, ebitda_margin)
    DATA: Standard fundamentals format expected by ReflectionAgent
    """
    return {
        "pe_ratio": 25.0,
        "revenue_growth": 0.08,  # 8% growth
        "ebitda_margin": 0.22,  # 22% margin
        "debt_to_ebitda": 1.5,
        "roe": 0.18,  # 18% return on equity
    }


@pytest.fixture
def incomplete_fundamentals():
    """
    Incomplete fundamentals missing required metrics.

    WHAT: Creates financial metrics dictionary missing critical keys
    WHY: Tests need baseline incomplete fundamentals to validate completeness check
    HOW: Returns dict missing pe_ratio (required field)
    DATA: Incomplete fundamentals format for testing error handling
    """
    return {
        "revenue_growth": 0.08,
        "ebitda_margin": 0.22,
        # Missing pe_ratio (required field)
    }


@pytest.fixture
def reflection_agent():
    """
    ReflectionAgent instance with default configuration.

    WHAT: Creates standard ReflectionAgent for testing
    WHY: Most tests need a basic agent instance
    HOW: Instantiate with default parameters
    DATA: ReflectionAgent object ready for testing
    """
    return ReflectionAgent()


# ===== PREDICTION VALIDITY TESTS =====


def test_valid_prediction_passes(
    reflection_agent, valid_prediction, valid_sentiment, valid_filing, valid_market_data, valid_fundamentals
):
    """
    Test that a completely valid set of inputs passes validation.

    WHAT: Verify ReflectionAgent passes all checks with good data
    WHY: Baseline test to ensure agent doesn't false-positive on valid data
    HOW: Call run() with all valid inputs, assert validation_passed=True
    DATA: Valid inputs → validation_passed=True, no issues, no adjustment
    """
    # WHAT: Run reflection validation with all valid inputs
    # WHY: Should pass all checks and return no issues
    # HOW: Call agent.run() with valid fixtures
    # DATA: All valid agent outputs → validation result
    result = reflection_agent.run(
        prediction=valid_prediction,
        sentiment=valid_sentiment,
        filing=valid_filing,
        market_data=valid_market_data,
        fundamentals=valid_fundamentals,
    )

    # WHAT: Assert validation passed
    # WHY: Valid inputs should not trigger any warnings
    # HOW: Check validation_passed flag is True
    # DATA: result["validation_passed"] should be True
    assert result["validation_passed"] is True

    # WHAT: Assert no confidence adjustment
    # WHY: No issues means no penalty to confidence
    # HOW: Check confidence_adjustment is 0.0
    # DATA: result["confidence_adjustment"] should be 0.0
    assert result["confidence_adjustment"] == 0.0

    # WHAT: Assert no issues found
    # WHY: Valid inputs should produce empty issues list
    # HOW: Check issues list is empty
    # DATA: result["issues"] should be []
    assert len(result["issues"]) == 0

    # WHAT: Assert all checks performed
    # WHY: Verify all validation checks ran (not disabled)
    # HOW: Check checks_performed dict
    # DATA: All checks should be True in checks_performed
    assert result["checks_performed"]["prediction_validity"] is True
    assert result["checks_performed"]["sentiment_alignment"] is True
    assert result["checks_performed"]["data_freshness"] is True
    assert result["checks_performed"]["fundamentals_completeness"] is True

    # WHAT: Assert all checks passed
    # WHY: Valid inputs should pass every individual check
    # HOW: Check checks_passed dict
    # DATA: All checks should be True in checks_passed
    assert result["checks_passed"]["prediction_validity"] is True
    assert result["checks_passed"]["sentiment_alignment"] is True
    assert result["checks_passed"]["data_freshness"] is True
    assert result["checks_passed"]["fundamentals_completeness"] is True


def test_invalid_confidence_fails(
    reflection_agent, valid_sentiment, valid_filing, valid_market_data, valid_fundamentals
):
    """
    Test that confidence outside [0, 1] range fails validation.

    WHAT: Verify agent detects confidence values outside valid probability range
    WHY: Confidence must be a probability (0-100% expressed as 0.0-1.0)
    HOW: Test with confidence=1.5 (>1.0), assert validation fails
    DATA: Invalid confidence → validation_passed=False, confidence issue flagged
    """
    # WHAT: Create prediction with invalid confidence (>1.0)
    # WHY: Confidence must be in [0, 1] range
    # HOW: Set confidence to 1.5 (150%, impossible probability)
    # DATA: Prediction dict with out-of-range confidence
    invalid_prediction = {
        "direction": "up",
        "confidence": 1.5,  # Invalid: >1.0
        "narrative": "Test prediction",
    }

    # WHAT: Run reflection validation
    # WHY: Should detect invalid confidence
    # HOW: Call agent.run() with invalid prediction
    # DATA: Invalid prediction → validation result with error
    result = reflection_agent.run(
        prediction=invalid_prediction,
        sentiment=valid_sentiment,
        filing=valid_filing,
        market_data=valid_market_data,
        fundamentals=valid_fundamentals,
    )

    # WHAT: Assert validation failed
    # WHY: Invalid confidence should trigger failure
    # HOW: Check validation_passed is False
    # DATA: result["validation_passed"] should be False
    assert result["validation_passed"] is False

    # WHAT: Assert confidence adjustment is negative
    # WHY: Invalid prediction should incur penalty (-0.2)
    # HOW: Check confidence_adjustment < 0
    # DATA: result["confidence_adjustment"] should be -0.2
    assert result["confidence_adjustment"] < 0
    assert result["confidence_adjustment"] == -0.2

    # WHAT: Assert issue was flagged
    # WHY: Should have error message about invalid confidence
    # HOW: Check issues list is not empty
    # DATA: result["issues"] should contain confidence error
    assert len(result["issues"]) > 0
    assert any("confidence" in issue.lower() for issue in result["issues"])

    # WHAT: Assert prediction_validity check failed
    # WHY: This specific check should detect the error
    # HOW: Check checks_passed dict
    # DATA: prediction_validity should be False
    assert result["checks_passed"]["prediction_validity"] is False


def test_invalid_direction_fails(
    reflection_agent, valid_sentiment, valid_filing, valid_market_data, valid_fundamentals
):
    """
    Test that invalid direction value fails validation.

    WHAT: Verify agent detects direction not in [up, down, neutral]
    WHY: Direction must match expected enum values from forecaster
    HOW: Test with direction="sideways" (invalid), assert validation fails
    DATA: Invalid direction → validation_passed=False, direction issue flagged
    """
    # WHAT: Create prediction with invalid direction
    # WHY: Direction must be one of [up, down, neutral]
    # HOW: Set direction to "sideways" (not valid)
    # DATA: Prediction dict with invalid direction
    invalid_prediction = {
        "direction": "sideways",  # Invalid: not in allowed values
        "confidence": 0.85,
        "narrative": "Test prediction",
    }

    # WHAT: Run reflection validation
    # WHY: Should detect invalid direction
    # HOW: Call agent.run() with invalid prediction
    # DATA: Invalid prediction → validation result with error
    result = reflection_agent.run(
        prediction=invalid_prediction,
        sentiment=valid_sentiment,
        filing=valid_filing,
        market_data=valid_market_data,
        fundamentals=valid_fundamentals,
    )

    # WHAT: Assert validation failed
    # WHY: Invalid direction should trigger failure
    assert result["validation_passed"] is False

    # WHAT: Assert confidence penalty applied
    # WHY: Invalid prediction should incur -0.2 penalty
    assert result["confidence_adjustment"] == -0.2

    # WHAT: Assert issue mentions direction
    # WHY: Error message should explain what's wrong
    assert len(result["issues"]) > 0
    assert any("direction" in issue.lower() for issue in result["issues"])


# ===== SENTIMENT-PREDICTION ALIGNMENT TESTS =====


def test_misaligned_bullish_prediction_bearish_sentiment_fails(
    reflection_agent, valid_filing, valid_market_data, valid_fundamentals
):
    """
    Test that "up" prediction with "negative" sentiment triggers warning.

    WHAT: Verify agent detects misalignment between bullish forecast and bearish news
    WHY: Predicting "up" when sentiment is "negative" suggests data issue or model error
    HOW: Test with direction="up" + current="negative", assert alignment issue flagged
    DATA: Misaligned prediction+sentiment → alignment check fails, -0.1 penalty
    """
    # WHAT: Create bullish prediction
    # WHY: Testing misalignment with bearish sentiment
    # HOW: Set direction="up", confidence=0.85
    # DATA: Prediction dict with bullish direction
    bullish_prediction = {
        "direction": "up",
        "confidence": 0.85,
        "narrative": "Test prediction",
    }

    # WHAT: Create bearish sentiment
    # WHY: Conflicts with bullish prediction
    # HOW: Set current="negative"
    # DATA: Sentiment dict with bearish sentiment
    bearish_sentiment = {
        "current": "negative",
        "score": 0.3,
        "trend": "declining",
        "headlines": ["Company misses earnings", "Revenue declines"],
    }

    # WHAT: Run reflection validation
    # WHY: Should detect prediction-sentiment misalignment
    # HOW: Call agent.run() with conflicting inputs
    # DATA: Misaligned inputs → validation result with warning
    result = reflection_agent.run(
        prediction=bullish_prediction,
        sentiment=bearish_sentiment,
        filing=valid_filing,
        market_data=valid_market_data,
        fundamentals=valid_fundamentals,
    )

    # WHAT: Assert validation failed
    # WHY: Misalignment should trigger failure
    assert result["validation_passed"] is False

    # WHAT: Assert alignment penalty applied
    # WHY: Misalignment should incur -0.1 penalty
    assert result["confidence_adjustment"] == -0.1

    # WHAT: Assert issue mentions conflict
    # WHY: Error message should explain misalignment
    assert len(result["issues"]) > 0
    assert any("conflict" in issue.lower() for issue in result["issues"])

    # WHAT: Assert alignment check failed
    # WHY: This specific check detected the issue
    assert result["checks_passed"]["sentiment_alignment"] is False


def test_misaligned_bearish_prediction_bullish_sentiment_fails(
    reflection_agent, valid_filing, valid_market_data, valid_fundamentals
):
    """
    Test that "down" prediction with "positive" sentiment triggers warning.

    WHAT: Verify agent detects misalignment between bearish forecast and bullish news
    WHY: Predicting "down" when sentiment is "positive" suggests data issue
    HOW: Test with direction="down" + current="positive", assert alignment issue flagged
    DATA: Misaligned prediction+sentiment → alignment check fails, -0.1 penalty
    """
    # WHAT: Create bearish prediction
    bearish_prediction = {
        "direction": "down",
        "confidence": 0.80,
        "narrative": "Test prediction",
    }

    # WHAT: Create bullish sentiment
    bullish_sentiment = {
        "current": "positive",
        "score": 0.75,
        "trend": "improving",
        "headlines": ["Strong earnings beat", "Revenue growth accelerates"],
    }

    # WHAT: Run reflection validation
    # WHY: Should detect misalignment
    result = reflection_agent.run(
        prediction=bearish_prediction,
        sentiment=bullish_sentiment,
        filing=valid_filing,
        market_data=valid_market_data,
        fundamentals=valid_fundamentals,
    )

    # WHAT: Assert validation failed due to misalignment
    assert result["validation_passed"] is False
    assert result["confidence_adjustment"] == -0.1
    assert any("conflict" in issue.lower() for issue in result["issues"])
    assert result["checks_passed"]["sentiment_alignment"] is False


def test_aligned_prediction_sentiment_passes(
    reflection_agent, valid_filing, valid_market_data, valid_fundamentals
):
    """
    Test that aligned prediction and sentiment passes validation.

    WHAT: Verify agent passes alignment check when prediction matches sentiment
    WHY: "up" prediction with "positive" sentiment is consistent
    HOW: Test with direction="up" + current="positive", assert no alignment issue
    DATA: Aligned prediction+sentiment → alignment check passes
    """
    # WHAT: Create bullish prediction
    bullish_prediction = {
        "direction": "up",
        "confidence": 0.85,
        "narrative": "Test prediction",
    }

    # WHAT: Create matching bullish sentiment
    bullish_sentiment = {
        "current": "positive",
        "score": 0.75,
        "trend": "improving",
        "headlines": ["Strong performance reported"],
    }

    # WHAT: Run reflection validation
    # WHY: Aligned inputs should pass
    result = reflection_agent.run(
        prediction=bullish_prediction,
        sentiment=bullish_sentiment,
        filing=valid_filing,
        market_data=valid_market_data,
        fundamentals=valid_fundamentals,
    )

    # WHAT: Assert validation passed
    # WHY: Alignment is good, no issues
    assert result["validation_passed"] is True
    assert result["confidence_adjustment"] == 0.0
    assert result["checks_passed"]["sentiment_alignment"] is True


# ===== DATA FRESHNESS TESTS =====


def test_stale_market_data_fails(
    reflection_agent, valid_prediction, valid_sentiment, valid_filing, valid_fundamentals
):
    """
    Test that market data >3 days old fails freshness check.

    WHAT: Verify agent detects stale market data (>max_data_age_days)
    WHY: Stale data leads to unreliable forecasts
    HOW: Test with data from 5 days ago, assert freshness issue flagged
    DATA: Old market_data → freshness check fails, -0.15 penalty
    """
    # WHAT: Create stale market data (5 days old)
    # WHY: Should trigger freshness warning
    # HOW: Use date from 5 days ago
    # DATA: Market data with old timestamp
    old_date = (datetime.now() - timedelta(days=5)).date().isoformat()
    stale_data = [
        {"date": old_date, "close": 150.0},
    ]

    # WHAT: Run reflection validation
    # WHY: Should detect stale data
    result = reflection_agent.run(
        prediction=valid_prediction,
        sentiment=valid_sentiment,
        filing=valid_filing,
        market_data=stale_data,
        fundamentals=valid_fundamentals,
    )

    # WHAT: Assert validation failed
    # WHY: Stale data should trigger failure
    assert result["validation_passed"] is False

    # WHAT: Assert freshness penalty applied
    # WHY: Stale data should incur -0.15 penalty
    assert result["confidence_adjustment"] == -0.15

    # WHAT: Assert issue mentions staleness
    # WHY: Error message should explain data age problem
    assert len(result["issues"]) > 0
    assert any("stale" in issue.lower() or "days old" in issue.lower() for issue in result["issues"])

    # WHAT: Assert freshness check failed
    # WHY: This specific check detected the issue
    assert result["checks_passed"]["data_freshness"] is False


def test_fresh_market_data_passes(
    reflection_agent, valid_prediction, valid_sentiment, valid_filing, valid_fundamentals
):
    """
    Test that recent market data passes freshness check.

    WHAT: Verify agent passes freshness check with today's data
    WHY: Fresh data (<= max_data_age_days) is reliable
    HOW: Test with today's date, assert no freshness issue
    DATA: Fresh market_data → freshness check passes
    """
    # WHAT: Create fresh market data (today)
    # WHY: Should pass freshness check
    fresh_data = [
        {"date": datetime.now().date().isoformat(), "close": 150.0},
    ]

    # WHAT: Run reflection validation
    # WHY: Fresh data should pass
    result = reflection_agent.run(
        prediction=valid_prediction,
        sentiment=valid_sentiment,
        filing=valid_filing,
        market_data=fresh_data,
        fundamentals=valid_fundamentals,
    )

    # WHAT: Assert freshness check passed
    # WHY: Fresh data should not trigger warnings
    assert result["checks_passed"]["data_freshness"] is True


def test_missing_market_data_fails(
    reflection_agent, valid_prediction, valid_sentiment, valid_filing, valid_fundamentals
):
    """
    Test that missing market data fails freshness check.

    WHAT: Verify agent detects missing market data (None or empty)
    WHY: Missing data prevents reliable forecasting
    HOW: Test with market_data=None, assert freshness issue flagged
    DATA: Missing market_data → freshness check fails
    """
    # WHAT: Test with None market data
    # WHY: Should trigger missing data error
    result = reflection_agent.run(
        prediction=valid_prediction,
        sentiment=valid_sentiment,
        filing=valid_filing,
        market_data=None,
        fundamentals=valid_fundamentals,
    )

    # WHAT: Assert validation failed
    # WHY: Missing data should trigger failure
    assert result["validation_passed"] is False
    assert result["checks_passed"]["data_freshness"] is False


# ===== FUNDAMENTALS COMPLETENESS TESTS =====


def test_missing_fundamentals_fails(
    reflection_agent, valid_prediction, valid_sentiment, valid_filing, valid_market_data
):
    """
    Test that missing required fundamental metrics fail completeness check.

    WHAT: Verify agent detects missing critical fundamentals (pe_ratio, etc.)
    WHY: Missing metrics degrade forecast quality
    HOW: Test with fundamentals missing pe_ratio, assert completeness issue flagged
    DATA: Incomplete fundamentals → completeness check fails, -0.05 penalty
    """
    # WHAT: Create fundamentals missing required key
    # WHY: Should trigger completeness warning
    # HOW: Omit pe_ratio (required field)
    # DATA: Fundamentals dict missing critical metric
    incomplete = {
        "revenue_growth": 0.08,
        "ebitda_margin": 0.22,
        # Missing pe_ratio
    }

    # WHAT: Run reflection validation
    # WHY: Should detect missing metric
    result = reflection_agent.run(
        prediction=valid_prediction,
        sentiment=valid_sentiment,
        filing=valid_filing,
        market_data=valid_market_data,
        fundamentals=incomplete,
    )

    # WHAT: Assert validation failed
    # WHY: Missing fundamental should trigger failure
    assert result["validation_passed"] is False

    # WHAT: Assert completeness penalty applied
    # WHY: Missing data should incur -0.05 penalty
    assert result["confidence_adjustment"] == -0.05

    # WHAT: Assert issue mentions missing metrics
    # WHY: Error message should list missing keys
    assert len(result["issues"]) > 0
    assert any("missing" in issue.lower() for issue in result["issues"])

    # WHAT: Assert completeness check failed
    # WHY: This specific check detected the issue
    assert result["checks_passed"]["fundamentals_completeness"] is False


def test_complete_fundamentals_passes(
    reflection_agent, valid_prediction, valid_sentiment, valid_filing, valid_market_data, valid_fundamentals
):
    """
    Test that complete fundamentals pass completeness check.

    WHAT: Verify agent passes completeness check with all required metrics
    WHY: Complete fundamentals enable high-quality forecasts
    HOW: Test with all required keys present, assert no completeness issue
    DATA: Complete fundamentals → completeness check passes
    """
    # WHAT: Run reflection validation with complete fundamentals
    # WHY: Should pass completeness check
    result = reflection_agent.run(
        prediction=valid_prediction,
        sentiment=valid_sentiment,
        filing=valid_filing,
        market_data=valid_market_data,
        fundamentals=valid_fundamentals,
    )

    # WHAT: Assert completeness check passed
    # WHY: All required metrics present
    assert result["checks_passed"]["fundamentals_completeness"] is True


# ===== CONFIDENCE ADJUSTMENT TESTS =====


def test_multiple_issues_accumulate_penalties(reflection_agent, valid_filing):
    """
    Test that multiple issues result in cumulative confidence adjustment.

    WHAT: Verify penalties stack when multiple validation checks fail
    WHY: Multiple issues should result in larger confidence downgrade
    HOW: Test with invalid prediction + stale data, assert cumulative penalty
    DATA: Multiple issues → cumulative confidence_adjustment (-0.2 + -0.15 = -0.35, clamped to -0.25)
    """
    # WHAT: Create invalid prediction (bad confidence)
    # WHY: Triggers -0.2 penalty
    invalid_prediction = {
        "direction": "up",
        "confidence": 1.5,  # Invalid
        "narrative": "Test",
    }

    # WHAT: Create stale market data
    # WHY: Triggers -0.15 penalty
    old_date = (datetime.now() - timedelta(days=5)).date().isoformat()
    stale_data = [{"date": old_date, "close": 150.0}]

    # WHAT: Create valid sentiment (no penalty)
    valid_sentiment = {"current": "positive", "score": 0.7, "trend": "stable"}

    # WHAT: Create incomplete fundamentals
    # WHY: Triggers -0.05 penalty
    incomplete_fundamentals = {"revenue_growth": 0.08}  # Missing pe_ratio, ebitda_margin

    # WHAT: Run reflection validation
    # WHY: Should accumulate all penalties
    result = reflection_agent.run(
        prediction=invalid_prediction,
        sentiment=valid_sentiment,
        filing=valid_filing,
        market_data=stale_data,
        fundamentals=incomplete_fundamentals,
    )

    # WHAT: Assert validation failed
    # WHY: Multiple issues present
    assert result["validation_passed"] is False

    # WHAT: Assert confidence adjustment is clamped to -0.25
    # WHY: Raw penalty would be -0.2 + -0.15 + -0.05 = -0.4, but clamped at -0.25
    # HOW: Check confidence_adjustment is -0.25 (maximum penalty)
    # DATA: Multiple penalties → clamped adjustment
    assert result["confidence_adjustment"] == -0.25

    # WHAT: Assert multiple issues flagged
    # WHY: Should have errors for prediction, freshness, completeness
    assert len(result["issues"]) >= 3


# ===== CONFIGURATION TESTS =====


def test_disable_alignment_check():
    """
    Test that alignment check can be disabled.

    WHAT: Verify enable_alignment_check=False skips alignment validation
    WHY: Allows selective disabling of checks for testing or special cases
    HOW: Create agent with enable_alignment_check=False, test misaligned inputs
    DATA: Misaligned inputs + disabled check → alignment check not performed
    """
    # WHAT: Create agent with alignment check disabled
    # WHY: Test configuration flexibility
    agent = ReflectionAgent(enable_alignment_check=False)

    # WHAT: Create misaligned prediction and sentiment
    # WHY: Would normally trigger alignment issue
    prediction = {"direction": "up", "confidence": 0.85, "narrative": "Test"}
    sentiment = {"current": "negative", "score": 0.3, "trend": "declining"}

    # WHAT: Run validation
    # WHY: Should NOT check alignment
    result = agent.run(
        prediction=prediction,
        sentiment=sentiment,
        filing={},
        market_data=[{"date": datetime.now().date().isoformat(), "close": 150.0}],
        fundamentals={"pe_ratio": 25, "revenue_growth": 0.08, "ebitda_margin": 0.22},
    )

    # WHAT: Assert alignment check not performed
    # WHY: Check was disabled
    assert "sentiment_alignment" not in result["checks_performed"]


def test_custom_max_data_age():
    """
    Test that custom max_data_age_days threshold works correctly.

    WHAT: Verify agent respects custom data age threshold
    WHY: Different use cases may need different freshness requirements
    HOW: Create agent with max_data_age_days=1, test 2-day-old data
    DATA: 2-day-old data + max_age=1 → freshness check fails
    """
    # WHAT: Create agent with strict freshness requirement (1 day)
    # WHY: Test custom threshold configuration
    agent = ReflectionAgent(max_data_age_days=1)

    # WHAT: Create market data from 2 days ago
    # WHY: Should fail with max_age=1 (but would pass with default max_age=3)
    two_days_ago = (datetime.now() - timedelta(days=2)).date().isoformat()
    market_data = [{"date": two_days_ago, "close": 150.0}]

    # WHAT: Run validation
    # WHY: Should detect stale data with custom threshold
    result = agent.run(
        prediction={"direction": "up", "confidence": 0.85, "narrative": "Test"},
        sentiment={"current": "positive", "score": 0.7, "trend": "stable"},
        filing={},
        market_data=market_data,
        fundamentals={"pe_ratio": 25, "revenue_growth": 0.08, "ebitda_margin": 0.22},
    )

    # WHAT: Assert freshness check failed
    # WHY: 2-day-old data exceeds 1-day threshold
    assert result["checks_passed"]["data_freshness"] is False
    assert result["validation_passed"] is False


# ===== RECOMMENDATIONS TESTS =====


def test_recommendations_generated_for_issues(reflection_agent, valid_filing, valid_fundamentals):
    """
    Test that actionable recommendations are generated for each issue type.

    WHAT: Verify agent provides helpful recommendations for fixing issues
    WHY: Users need guidance on how to resolve quality problems
    HOW: Trigger multiple issues, assert recommendations present
    DATA: Multiple issues → recommendations list with actionable fixes
    """
    # WHAT: Create inputs that trigger multiple issues
    # WHY: Want to test recommendation generation for different issue types
    invalid_prediction = {"direction": "up", "confidence": 1.5, "narrative": "Test"}
    old_date = (datetime.now() - timedelta(days=5)).date().isoformat()
    stale_data = [{"date": old_date, "close": 150.0}]
    valid_sentiment = {"current": "positive", "score": 0.7, "trend": "stable"}

    # WHAT: Run validation
    # WHY: Should generate recommendations for each issue
    result = reflection_agent.run(
        prediction=invalid_prediction,
        sentiment=valid_sentiment,
        filing=valid_filing,
        market_data=stale_data,
        fundamentals=valid_fundamentals,
    )

    # WHAT: Assert recommendations were generated
    # WHY: Should have actionable fixes for prediction and freshness issues
    assert len(result["recommendations"]) > 0

    # WHAT: Assert recommendations mention specific fixes
    # WHY: Should guide user to solutions
    recommendations_text = " ".join(result["recommendations"]).lower()
    # Should mention refreshing data for stale data issue
    assert "refresh" in recommendations_text or "data" in recommendations_text
