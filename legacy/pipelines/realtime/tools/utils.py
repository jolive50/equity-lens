"""Common utility functions for StockSense.

This module provides reusable helper functions for:
- Date/time operations (trading days, date parsing, formatting)
- Financial calculations (returns, volatility, Sharpe ratio)
- Data validation (ticker format, date ranges)
- Output formatting (currency, percentages, large numbers)

WHAT: Centralized utility functions used across agents and pipelines
WHY: Avoid code duplication, ensure consistent behavior, easier testing
HOW: Pure functions with clear inputs/outputs, comprehensive error handling
DATA: Various inputs → validated/calculated/formatted outputs
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ===== DATE/TIME UTILITIES =====


def get_trading_days(
    start_date: str,
    end_date: str,
    exchange: str = "NYSE"
) -> List[str]:
    """
    Get list of trading days between start and end dates.

    Args:
        start_date: Start date in ISO format (YYYY-MM-DD)
        end_date: End date in ISO format (YYYY-MM-DD)
        exchange: Exchange calendar to use (default: "NYSE")

    Returns:
        List of trading days in ISO format

    WHAT: Generate list of valid trading days excluding weekends/holidays
    WHY: Need to filter out non-trading days for data analysis
    HOW: Use pandas date_range with BDay (business day) frequency
    DATA: (start_date, end_date) → list of trading days
    """
    # WHAT: Convert string dates to pandas datetime
    # WHY: pandas date_range requires datetime objects
    # HOW: Parse ISO format strings
    # DATA: ISO strings → pandas Timestamp objects
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # WHAT: Generate business days (Monday-Friday)
    # WHY: Stock markets don't trade on weekends
    # HOW: Use pandas BDay frequency (excludes Sat/Sun)
    # DATA: datetime range → list of business days
    trading_days = pd.date_range(start=start, end=end, freq='B')

    # WHAT: Convert to ISO format strings
    # WHY: Consistent string format for API compatibility
    # HOW: Use strftime to format dates
    # DATA: pandas DatetimeIndex → list of ISO strings
    return [day.strftime('%Y-%m-%d') for day in trading_days]


def is_trading_day(date: str, exchange: str = "NYSE") -> bool:
    """
    Check if a given date is a trading day.

    Args:
        date: Date in ISO format (YYYY-MM-DD)
        exchange: Exchange calendar to use (default: "NYSE")

    Returns:
        bool: True if trading day, False otherwise

    WHAT: Determine if a specific date is a trading day
    WHY: Validate dates before fetching market data
    HOW: Check if weekday and not a holiday
    DATA: date string → bool (is trading day)
    """
    # WHAT: Parse date string to datetime
    # WHY: Need datetime object to check weekday
    # HOW: Use datetime.fromisoformat
    # DATA: ISO string → datetime object
    dt = datetime.fromisoformat(date)

    # WHAT: Check if weekend
    # WHY: Markets closed Saturday (5) and Sunday (6)
    # HOW: Use weekday() method (0=Mon, 6=Sun)
    # DATA: datetime → weekday int → bool check
    if dt.weekday() >= 5:  # Saturday or Sunday
        return False

    # TODO: Add holiday calendar check (NYSE, NASDAQ holidays)
    # For now, just exclude weekends
    return True


def parse_date(date_str: str, default_format: str = '%Y-%m-%d') -> datetime:
    """
    Parse date string to datetime object with flexible format support.

    Args:
        date_str: Date string in various formats
        default_format: Default format to try first (default: ISO format)

    Returns:
        datetime object

    Raises:
        ValueError: If date string cannot be parsed

    WHAT: Convert date strings in various formats to datetime
    WHY: APIs return dates in different formats, need standardization
    HOW: Try multiple parsing strategies (ISO, common formats)
    DATA: date string → datetime object
    """
    # WHAT: Try default format first (most common case)
    # WHY: Faster if we know expected format
    # HOW: Use strptime with provided format
    # DATA: date string + format → datetime or exception
    try:
        return datetime.strptime(date_str, default_format)
    except ValueError:
        pass

    # WHAT: Try common alternative formats
    # WHY: Different data sources use different formats
    # HOW: Iterate through known formats
    # DATA: List of formats to try
    common_formats = [
        '%Y-%m-%d',           # ISO format (YYYY-MM-DD)
        '%m/%d/%Y',           # US format (MM/DD/YYYY)
        '%d/%m/%Y',           # European format (DD/MM/YYYY)
        '%Y%m%d',             # Compact format (YYYYMMDD)
        '%b %d, %Y',          # Month name format (Jan 01, 2025)
        '%B %d, %Y',          # Full month name (January 01, 2025)
    ]

    # WHAT: Try each format until one works
    # WHY: Flexible parsing for various input formats
    # HOW: Loop through formats, return on first success
    # DATA: format attempts → datetime or final exception
    for fmt in common_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # WHAT: If all formats fail, raise descriptive error
    # WHY: Need to know what string failed and why
    # HOW: Raise ValueError with input string
    # DATA: Original string in error message
    raise ValueError(f"Unable to parse date string: {date_str}")


def format_date(dt: datetime, format_str: str = '%Y-%m-%d') -> str:
    """
    Format datetime object to string.

    Args:
        dt: Datetime object to format
        format_str: Output format (default: ISO format)

    Returns:
        Formatted date string

    WHAT: Convert datetime to formatted string
    WHY: Need consistent date formatting for output
    HOW: Use strftime with format string
    DATA: datetime → formatted string
    """
    # WHAT: Format datetime using strftime
    # WHY: Standard way to convert datetime to string
    # HOW: Apply format string template
    # DATA: datetime + format → string representation
    return dt.strftime(format_str)


# ===== FINANCIAL CALCULATIONS =====


def calculate_returns(
    prices: np.ndarray,
    method: str = 'simple'
) -> np.ndarray:
    """
    Calculate returns from price series.

    Args:
        prices: Array of prices (chronological order)
        method: 'simple' or 'log' returns (default: 'simple')

    Returns:
        Array of returns (length = len(prices) - 1)

    WHAT: Compute returns from price time series
    WHY: Returns are fundamental for volatility, Sharpe ratio calculations
    HOW: Calculate price differences, normalize by previous price
    DATA: price series → return series
    """
    # WHAT: Validate input array
    # WHY: Need at least 2 prices to calculate a return
    # HOW: Check array length
    # DATA: Array length check
    if len(prices) < 2:
        raise ValueError("Need at least 2 prices to calculate returns")

    # WHAT: Calculate returns based on method
    # WHY: Simple returns for reporting, log returns for modeling
    # HOW: Use different formulas for each method
    # DATA: prices → returns via formula
    if method == 'simple':
        # WHAT: Simple returns: (P_t - P_{t-1}) / P_{t-1}
        # WHY: Intuitive percentage change
        # HOW: Subtract adjacent prices, divide by previous
        # DATA: (current - previous) / previous
        returns = np.diff(prices) / prices[:-1]
    elif method == 'log':
        # WHAT: Log returns: log(P_t / P_{t-1})
        # WHY: Better statistical properties for modeling
        # HOW: Take log of price ratios
        # DATA: log(current / previous)
        returns = np.log(prices[1:] / prices[:-1])
    else:
        raise ValueError(f"Invalid method: {method}. Use 'simple' or 'log'")

    return returns


def calculate_volatility(
    returns: np.ndarray,
    annualize: bool = True,
    trading_days: int = 252
) -> float:
    """
    Calculate volatility (standard deviation) of returns.

    Args:
        returns: Array of returns
        annualize: Whether to annualize the volatility (default: True)
        trading_days: Trading days per year for annualization (default: 252)

    Returns:
        Volatility (annualized if annualize=True)

    WHAT: Compute standard deviation of returns as volatility measure
    WHY: Volatility is key risk metric for portfolio management
    HOW: Calculate std dev, optionally scale to annual basis
    DATA: returns → volatility (annualized or not)
    """
    # WHAT: Calculate standard deviation of returns
    # WHY: Standard measure of dispersion/risk
    # HOW: Use numpy std with ddof=1 (sample std dev)
    # DATA: returns array → single volatility number
    vol = np.std(returns, ddof=1)

    # WHAT: Annualize volatility if requested
    # WHY: Convention to report annual volatility
    # HOW: Multiply by square root of trading days per year
    # DATA: daily vol → annual vol via sqrt scaling
    if annualize:
        vol = vol * np.sqrt(trading_days)

    return float(vol)


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.02,
    annualize: bool = True,
    trading_days: int = 252
) -> float:
    """
    Calculate Sharpe ratio (risk-adjusted return).

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate (default: 0.02 = 2%)
        annualize: Whether to annualize the ratio (default: True)
        trading_days: Trading days per year for annualization (default: 252)

    Returns:
        Sharpe ratio

    WHAT: Compute risk-adjusted return metric
    WHY: Compare strategies with different risk profiles
    HOW: (mean return - risk-free rate) / volatility
    DATA: returns + risk-free rate → Sharpe ratio
    """
    # WHAT: Calculate mean return
    # WHY: Need average return for numerator
    # HOW: Use numpy mean
    # DATA: returns array → single mean value
    mean_return = np.mean(returns)

    # WHAT: Calculate volatility
    # WHY: Need risk measure for denominator
    # HOW: Call calculate_volatility helper
    # DATA: returns → volatility
    vol = calculate_volatility(returns, annualize=False)

    # WHAT: Convert annual risk-free rate to daily
    # WHY: Need same time scale as returns
    # HOW: Divide annual rate by trading days
    # DATA: annual rate → daily rate
    daily_rf = risk_free_rate / trading_days if annualize else risk_free_rate

    # WHAT: Calculate Sharpe ratio
    # WHY: Standard risk-adjusted return formula
    # HOW: (excess return) / volatility
    # DATA: (mean - rf) / vol → Sharpe ratio
    sharpe = (mean_return - daily_rf) / vol if vol > 0 else 0.0

    # WHAT: Annualize Sharpe ratio if requested
    # WHY: Convention to report annual Sharpe
    # HOW: Multiply by square root of trading days
    # DATA: daily Sharpe → annual Sharpe
    if annualize:
        sharpe = sharpe * np.sqrt(trading_days)

    return float(sharpe)


# ===== DATA VALIDATION =====


def validate_ticker(ticker: str) -> Tuple[bool, Optional[str]]:
    """
    Validate ticker symbol format.

    Args:
        ticker: Stock ticker symbol to validate

    Returns:
        Tuple of (is_valid, error_message)

    WHAT: Check if ticker format is valid
    WHY: Catch invalid tickers before API calls
    HOW: Regex pattern matching + length checks
    DATA: ticker string → (bool, error message)
    """
    # WHAT: Check if ticker is empty
    # WHY: Empty string not valid ticker
    # HOW: String truthiness check
    # DATA: Empty check → bool
    if not ticker:
        return False, "Ticker cannot be empty"

    # WHAT: Check ticker length
    # WHY: Valid tickers are typically 1-5 characters
    # HOW: String length check
    # DATA: Length constraint validation
    if len(ticker) < 1 or len(ticker) > 5:
        return False, "Ticker must be 1-5 characters"

    # WHAT: Check ticker format using regex
    # WHY: Valid tickers are uppercase letters only
    # HOW: Regex pattern ^[A-Z]+$
    # DATA: Pattern match → bool
    pattern = r'^[A-Z]+$'
    if not re.match(pattern, ticker):
        return False, "Ticker must contain only uppercase letters (A-Z)"

    # WHAT: Ticker passed all checks
    # DATA: Return success tuple
    return True, None


def validate_date_range(
    start_date: str,
    end_date: str,
    max_days: Optional[int] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate date range for data requests.

    Args:
        start_date: Start date in ISO format
        end_date: End date in ISO format
        max_days: Optional maximum allowed days in range

    Returns:
        Tuple of (is_valid, error_message)

    WHAT: Validate date range is sensible
    WHY: Catch invalid date ranges before data fetch
    HOW: Parse dates, check ordering and span
    DATA: (start, end, max) → (bool, error message)
    """
    # WHAT: Parse date strings
    # WHY: Need datetime objects to compare
    # HOW: Use parse_date helper
    # DATA: Date strings → datetime objects or error
    try:
        start = parse_date(start_date)
        end = parse_date(end_date)
    except ValueError as e:
        return False, f"Invalid date format: {e}"

    # WHAT: Check start is before end
    # WHY: Time series data requires chronological order
    # HOW: Datetime comparison
    # DATA: datetime comparison → bool
    if start >= end:
        return False, "Start date must be before end date"

    # WHAT: Check date range span if max_days specified
    # WHY: Some APIs have limits on date range
    # HOW: Calculate days between dates
    # DATA: datetime subtraction → timedelta → days
    if max_days is not None:
        days_span = (end - start).days
        if days_span > max_days:
            return False, f"Date range exceeds maximum of {max_days} days"

    # WHAT: Date range passed all checks
    # DATA: Return success tuple
    return True, None


# ===== OUTPUT FORMATTING =====


def format_currency(
    amount: float,
    currency: str = "USD",
    decimals: int = 2
) -> str:
    """
    Format number as currency string.

    Args:
        amount: Numerical amount to format
        currency: Currency code (default: "USD")
        decimals: Number of decimal places (default: 2)

    Returns:
        Formatted currency string

    WHAT: Convert number to human-readable currency format
    WHY: User-facing output needs proper formatting
    HOW: Add currency symbol, thousands separators, decimal places
    DATA: float → formatted string (e.g., "$1,234.56")
    """
    # WHAT: Define currency symbols
    # WHY: Need symbol for formatting
    # HOW: Dictionary lookup
    # DATA: currency code → symbol
    symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
    }

    symbol = symbols.get(currency, currency)

    # WHAT: Format with thousands separator and decimals
    # WHY: Standard currency formatting conventions
    # HOW: Use Python f-string formatting
    # DATA: float → formatted string with commas
    formatted = f"{symbol}{amount:,.{decimals}f}"

    return formatted


def format_percentage(
    value: float,
    decimals: int = 2,
    include_sign: bool = True
) -> str:
    """
    Format number as percentage string.

    Args:
        value: Value to format (0.15 = 15%)
        decimals: Number of decimal places (default: 2)
        include_sign: Include + for positive values (default: True)

    Returns:
        Formatted percentage string

    WHAT: Convert decimal to percentage format
    WHY: User-facing percentages need consistent formatting
    HOW: Multiply by 100, add % symbol, optionally add sign
    DATA: float → percentage string (e.g., "+15.50%")
    """
    # WHAT: Convert to percentage (multiply by 100)
    # WHY: Input is decimal (0.15), output is percent (15%)
    # HOW: Multiply by 100
    # DATA: decimal value → percentage value
    percentage = value * 100

    # WHAT: Format with sign if requested
    # WHY: Makes positive/negative changes clear
    # HOW: Use + in format string if include_sign
    # DATA: Apply formatting with optional sign
    if include_sign and percentage >= 0:
        formatted = f"+{percentage:.{decimals}f}%"
    else:
        formatted = f"{percentage:.{decimals}f}%"

    return formatted


def format_large_number(
    number: float,
    decimals: int = 2
) -> str:
    """
    Format large numbers with K/M/B suffixes.

    Args:
        number: Number to format
        decimals: Number of decimal places (default: 2)

    Returns:
        Formatted string with suffix

    WHAT: Convert large numbers to human-readable format
    WHY: Market caps, volumes easier to read as "1.5B" than "1500000000"
    HOW: Determine magnitude, divide, add suffix
    DATA: large number → compact string (e.g., "1.5B")
    """
    # WHAT: Define magnitude thresholds and suffixes
    # WHY: Standard abbreviations for thousands, millions, billions
    # HOW: List of (threshold, suffix) tuples in descending order
    # DATA: Magnitude thresholds
    magnitudes = [
        (1_000_000_000_000, 'T'),  # Trillion
        (1_000_000_000, 'B'),       # Billion
        (1_000_000, 'M'),           # Million
        (1_000, 'K'),               # Thousand
    ]

    # WHAT: Get absolute value for magnitude calculation
    # WHY: Negative numbers have same magnitude as positive
    # HOW: Use abs() function
    # DATA: Preserve sign, work with absolute value
    abs_number = abs(number)
    sign = '-' if number < 0 else ''

    # WHAT: Find appropriate magnitude and suffix
    # WHY: Use largest applicable suffix
    # HOW: Iterate through magnitudes, use first match
    # DATA: number → (divisor, suffix) pair
    for threshold, suffix in magnitudes:
        if abs_number >= threshold:
            # WHAT: Divide by threshold, format with suffix
            # WHY: Express number in appropriate units
            # HOW: Division + string formatting
            # DATA: (number / threshold) + suffix
            scaled = abs_number / threshold
            return f"{sign}{scaled:.{decimals}f}{suffix}"

    # WHAT: Number is small, format normally
    # WHY: No suffix needed for numbers < 1000
    # HOW: Standard float formatting
    # DATA: Float → string with decimals
    return f"{sign}{abs_number:.{decimals}f}"
