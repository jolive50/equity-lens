"""Unit tests for tools/utils.py utility functions.

This test suite validates:
- Date/time operations (trading days, parsing, formatting)
- Financial calculations (returns, volatility, Sharpe ratio)
- Data validation (ticker format, date ranges)
- Output formatting (currency, percentages, large numbers)

WHAT: Comprehensive test coverage for all utility functions
WHY: Ensures utility functions work correctly and handle edge cases
HOW: Pytest-based unit tests with fixtures for common test data
DATA: Sample inputs → utility functions → assertions on outputs
"""
from datetime import datetime

import numpy as np
import pytest

from pipelines.realtime.tools.utils import (
    # Date/time
    get_trading_days,
    is_trading_day,
    parse_date,
    format_date,
    # Financial calculations
    calculate_returns,
    calculate_volatility,
    calculate_sharpe_ratio,
    # Validation
    validate_ticker,
    validate_date_range,
    # Formatting
    format_currency,
    format_percentage,
    format_large_number,
)


# ===== DATE/TIME TESTS =====


def test_get_trading_days():
    """Test get_trading_days returns list of business days."""
    # WHAT: Get trading days for a week
    # WHY: Should return 5 days (Mon-Fri)
    days = get_trading_days("2025-10-27", "2025-10-31")  # Mon-Fri

    # WHAT: Assert we get 5 trading days
    assert len(days) == 5
    assert days[0] == "2025-10-27"  # Monday
    assert days[-1] == "2025-10-31"  # Friday


def test_get_trading_days_excludes_weekends():
    """Test get_trading_days excludes Saturday and Sunday."""
    # WHAT: Get trading days spanning a weekend
    # WHY: Should skip Sat/Sun
    days = get_trading_days("2025-10-31", "2025-11-03")  # Fri-Mon

    # WHAT: Assert Saturday and Sunday excluded
    assert "2025-11-01" not in days  # Saturday
    assert "2025-11-02" not in days  # Sunday
    assert "2025-10-31" in days      # Friday
    assert "2025-11-03" in days      # Monday


def test_is_trading_day_weekday():
    """Test is_trading_day returns True for weekdays."""
    # WHAT: Check Monday through Friday
    # WHY: All weekdays are trading days
    assert is_trading_day("2025-10-27") is True  # Monday
    assert is_trading_day("2025-10-28") is True  # Tuesday
    assert is_trading_day("2025-10-29") is True  # Wednesday
    assert is_trading_day("2025-10-30") is True  # Thursday
    assert is_trading_day("2025-10-31") is True  # Friday


def test_is_trading_day_weekend():
    """Test is_trading_day returns False for weekends."""
    # WHAT: Check Saturday and Sunday
    # WHY: Weekends are not trading days
    assert is_trading_day("2025-11-01") is False  # Saturday
    assert is_trading_day("2025-11-02") is False  # Sunday


def test_parse_date_iso_format():
    """Test parse_date handles ISO format (YYYY-MM-DD)."""
    # WHAT: Parse standard ISO date
    # WHY: Most common format
    dt = parse_date("2025-10-28")

    # WHAT: Assert correct datetime returned
    assert dt.year == 2025
    assert dt.month == 10
    assert dt.day == 28


def test_parse_date_us_format():
    """Test parse_date handles US format (MM/DD/YYYY)."""
    # WHAT: Parse US date format
    # WHY: Some data sources use this format
    dt = parse_date("10/28/2025", default_format='%m/%d/%Y')

    # WHAT: Assert correct parsing
    assert dt.year == 2025
    assert dt.month == 10
    assert dt.day == 28


def test_parse_date_invalid_raises_error():
    """Test parse_date raises ValueError for invalid dates."""
    # WHAT: Try to parse completely invalid date
    # WHY: Should raise descriptive error
    with pytest.raises(ValueError, match="Unable to parse"):
        parse_date("not-a-date")


def test_format_date():
    """Test format_date converts datetime to string."""
    # WHAT: Format datetime to ISO string
    dt = datetime(2025, 10, 28)
    formatted = format_date(dt)

    # WHAT: Assert correct format
    assert formatted == "2025-10-28"


def test_format_date_custom_format():
    """Test format_date with custom format string."""
    # WHAT: Format with US-style format
    dt = datetime(2025, 10, 28)
    formatted = format_date(dt, format_str='%m/%d/%Y')

    # WHAT: Assert custom format applied
    assert formatted == "10/28/2025"


# ===== FINANCIAL CALCULATION TESTS =====


def test_calculate_returns_simple():
    """Test calculate_returns with simple returns."""
    # WHAT: Calculate returns from price series
    # WHY: Basic return calculation test
    prices = np.array([100, 105, 110, 108])
    returns = calculate_returns(prices, method='simple')

    # WHAT: Assert correct returns calculated
    # (105-100)/100 = 0.05, (110-105)/105 ≈ 0.0476, (108-110)/110 ≈ -0.0182
    assert len(returns) == 3
    assert np.isclose(returns[0], 0.05)
    assert np.isclose(returns[1], 0.0476, atol=0.001)
    assert returns[2] < 0  # Negative return


def test_calculate_returns_log():
    """Test calculate_returns with log returns."""
    # WHAT: Calculate log returns
    # WHY: Verify log return formula
    prices = np.array([100, 105])
    returns = calculate_returns(prices, method='log')

    # WHAT: Assert log return calculated
    # log(105/100) = log(1.05) ≈ 0.04879
    assert len(returns) == 1
    assert np.isclose(returns[0], 0.04879, atol=0.0001)


def test_calculate_returns_too_few_prices():
    """Test calculate_returns raises error with insufficient data."""
    # WHAT: Try to calculate returns with only 1 price
    # WHY: Need at least 2 prices for a return
    prices = np.array([100])

    with pytest.raises(ValueError, match="at least 2 prices"):
        calculate_returns(prices)


def test_calculate_volatility():
    """Test calculate_volatility calculates standard deviation."""
    # WHAT: Calculate volatility from returns
    returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
    vol = calculate_volatility(returns, annualize=False)

    # WHAT: Assert volatility calculated
    # Should be std dev of returns
    expected_vol = np.std(returns, ddof=1)
    assert np.isclose(vol, expected_vol)


def test_calculate_volatility_annualized():
    """Test calculate_volatility with annualization."""
    # WHAT: Calculate annualized volatility
    returns = np.array([0.01, -0.02, 0.015, -0.005, 0.02])
    vol_daily = calculate_volatility(returns, annualize=False)
    vol_annual = calculate_volatility(returns, annualize=True, trading_days=252)

    # WHAT: Assert annualized vol is daily * sqrt(252)
    assert np.isclose(vol_annual, vol_daily * np.sqrt(252))


def test_calculate_sharpe_ratio():
    """Test calculate_sharpe_ratio basic calculation."""
    # WHAT: Calculate Sharpe ratio
    # WHY: Risk-adjusted return metric
    returns = np.array([0.001, 0.002, -0.001, 0.0015, 0.003])  # Daily returns
    sharpe = calculate_sharpe_ratio(
        returns,
        risk_free_rate=0.02,
        annualize=True,
        trading_days=252
    )

    # WHAT: Assert Sharpe ratio is calculated
    # WHY: Should be positive with positive mean return
    assert isinstance(sharpe, float)
    # Mean return > risk-free rate, so Sharpe should be positive
    assert sharpe > 0


def test_calculate_sharpe_ratio_zero_volatility():
    """Test calculate_sharpe_ratio handles zero volatility."""
    # WHAT: Calculate Sharpe with constant returns (zero volatility)
    # WHY: Should handle division by zero gracefully
    returns = np.array([0.01, 0.01, 0.01])  # Constant returns
    sharpe = calculate_sharpe_ratio(returns, annualize=False)

    # WHAT: Assert Sharpe is 0 when vol is 0
    # WHY: Cannot have infinite Sharpe ratio
    assert sharpe == 0.0


# ===== VALIDATION TESTS =====


def test_validate_ticker_valid():
    """Test validate_ticker accepts valid tickers."""
    # WHAT: Validate common ticker formats
    # WHY: Should accept standard ticker symbols
    assert validate_ticker("AAPL") == (True, None)
    assert validate_ticker("MSFT") == (True, None)
    assert validate_ticker("GOOGL") == (True, None)
    assert validate_ticker("BRK") == (True, None)  # 3 chars
    assert validate_ticker("A") == (True, None)     # 1 char


def test_validate_ticker_empty():
    """Test validate_ticker rejects empty string."""
    # WHAT: Try to validate empty ticker
    # WHY: Empty string is not valid
    valid, error = validate_ticker("")

    assert valid is False
    assert "cannot be empty" in error


def test_validate_ticker_too_long():
    """Test validate_ticker rejects tickers > 5 chars."""
    # WHAT: Try to validate overly long ticker
    # WHY: Max ticker length is 5 characters
    valid, error = validate_ticker("TOOLONG")

    assert valid is False
    assert "1-5 characters" in error


def test_validate_ticker_invalid_characters():
    """Test validate_ticker rejects non-uppercase letters."""
    # WHAT: Try tickers with invalid characters
    # WHY: Tickers must be uppercase letters only
    assert validate_ticker("aapl")[0] is False  # Lowercase
    assert validate_ticker("AA123")[0] is False  # Numbers
    assert validate_ticker("AA-PL")[0] is False  # Hyphens


def test_validate_date_range_valid():
    """Test validate_date_range accepts valid ranges."""
    # WHAT: Validate proper date range
    # WHY: Start before end is valid
    valid, error = validate_date_range("2025-10-01", "2025-10-31")

    assert valid is True
    assert error is None


def test_validate_date_range_reversed():
    """Test validate_date_range rejects reversed dates."""
    # WHAT: Try start date after end date
    # WHY: Invalid chronological order
    valid, error = validate_date_range("2025-10-31", "2025-10-01")

    assert valid is False
    assert "before end date" in error


def test_validate_date_range_max_days():
    """Test validate_date_range enforces max_days limit."""
    # WHAT: Validate range exceeding max_days
    # WHY: Some APIs have date range limits
    valid, error = validate_date_range(
        "2025-01-01",
        "2025-12-31",
        max_days=30
    )

    assert valid is False
    assert "exceeds maximum" in error


def test_validate_date_range_invalid_format():
    """Test validate_date_range catches invalid date formats."""
    # WHAT: Try invalid date string
    # WHY: Should catch parse errors
    valid, error = validate_date_range("invalid", "2025-10-31")

    assert valid is False
    assert "Invalid date format" in error


# ===== FORMATTING TESTS =====


def test_format_currency_usd():
    """Test format_currency with USD."""
    # WHAT: Format US dollar amount
    # WHY: Most common currency format
    formatted = format_currency(1234.56, currency="USD")

    # WHAT: Assert proper formatting with $ and commas
    assert formatted == "$1,234.56"


def test_format_currency_large_amount():
    """Test format_currency with large amounts."""
    # WHAT: Format million-dollar amount
    # WHY: Should have thousands separators
    formatted = format_currency(1234567.89, currency="USD")

    assert formatted == "$1,234,567.89"


def test_format_currency_other_currencies():
    """Test format_currency with various currencies."""
    # WHAT: Format different currency codes
    # WHY: Support multiple currencies
    assert format_currency(100, currency="EUR") == "€100.00"
    assert format_currency(100, currency="GBP") == "£100.00"
    assert format_currency(100, currency="JPY") == "¥100.00"


def test_format_percentage_positive():
    """Test format_percentage with positive values."""
    # WHAT: Format positive percentage
    # WHY: Should include + sign by default
    formatted = format_percentage(0.1550, decimals=2)

    assert formatted == "+15.50%"


def test_format_percentage_negative():
    """Test format_percentage with negative values."""
    # WHAT: Format negative percentage
    # WHY: Should include - sign
    formatted = format_percentage(-0.0525, decimals=2)

    assert formatted == "-5.25%"


def test_format_percentage_no_sign():
    """Test format_percentage without sign for positive values."""
    # WHAT: Format without + sign
    # WHY: Sometimes don't want explicit positive sign
    formatted = format_percentage(0.15, decimals=2, include_sign=False)

    assert formatted == "15.00%"


def test_format_large_number_thousands():
    """Test format_large_number with thousands."""
    # WHAT: Format number in thousands
    # WHY: Should add K suffix
    formatted = format_large_number(1500, decimals=1)

    assert formatted == "1.5K"


def test_format_large_number_millions():
    """Test format_large_number with millions."""
    # WHAT: Format number in millions
    # WHY: Should add M suffix
    formatted = format_large_number(2500000, decimals=1)

    assert formatted == "2.5M"


def test_format_large_number_billions():
    """Test format_large_number with billions."""
    # WHAT: Format number in billions
    # WHY: Should add B suffix
    formatted = format_large_number(3750000000, decimals=2)

    assert formatted == "3.75B"


def test_format_large_number_trillions():
    """Test format_large_number with trillions."""
    # WHAT: Format number in trillions
    # WHY: Should add T suffix
    formatted = format_large_number(1250000000000, decimals=2)

    assert formatted == "1.25T"


def test_format_large_number_small():
    """Test format_large_number with small numbers."""
    # WHAT: Format number less than 1000
    # WHY: Should not add suffix
    formatted = format_large_number(123.45, decimals=2)

    assert formatted == "123.45"


def test_format_large_number_negative():
    """Test format_large_number with negative numbers."""
    # WHAT: Format negative number
    # WHY: Should preserve negative sign
    formatted = format_large_number(-2500000, decimals=1)

    assert formatted == "-2.5M"
