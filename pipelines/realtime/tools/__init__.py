"""Tools layer for StockSense utility functions and helpers.

This package provides reusable utility functions for:
- Date/time manipulation (trading days, date parsing, timezone handling)
- Financial calculations (returns, volatility, technical indicators)
- Data validation (ticker format, data quality checks)
- Formatting (currency, percentages, large numbers)

What this layer does:
- Centralizes common utility functions used across agents and pipelines
- Provides consistent implementations for financial calculations
- Handles edge cases and error conditions uniformly
- Reduces code duplication across the codebase

Why we need this:
- Single source of truth for utility logic (DRY principle)
- Easier testing (test once, use everywhere)
- Consistent behavior across all agents and workflows
- Cleaner code in agents (delegate utilities to this layer)
"""
from .utils import (
    # Date/time utilities
    get_trading_days,
    is_trading_day,
    parse_date,
    format_date,

    # Financial calculations
    calculate_returns,
    calculate_volatility,
    calculate_sharpe_ratio,

    # Data validation
    validate_ticker,
    validate_date_range,

    # Formatting
    format_currency,
    format_percentage,
    format_large_number,
)

__all__ = [
    # Date/time
    "get_trading_days",
    "is_trading_day",
    "parse_date",
    "format_date",

    # Financial calculations
    "calculate_returns",
    "calculate_volatility",
    "calculate_sharpe_ratio",

    # Validation
    "validate_ticker",
    "validate_date_range",

    # Formatting
    "format_currency",
    "format_percentage",
    "format_large_number",
]
