"""Tests for the API key helper functions."""
import os

from pipelines.realtime.api_keys import get_api_key, get_available_api_keys, missing_api_keys


def test_get_api_key_returns_value(monkeypatch):
    """get_api_key should return the value when the environment variable exists."""
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "demo-key")
    # What: Fetch the configured key through the helper.
    # Why: Ensures runtime code only reads keys via the central registry.
    # How: Call get_api_key and assert the returned string matches expectation.
    # Data: The environment variable is set to a test placeholder.
    assert get_api_key("alpha_vantage") == "demo-key"


def test_get_api_key_required_flag(monkeypatch):
    """get_api_key should raise when required=True and the key is missing."""
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    try:
        get_api_key("finnhub", required=True)
    except ValueError as exc:
        assert "FINNHUB_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected ValueError when key is missing")


def test_get_available_api_keys_filters_missing(monkeypatch):
    """get_available_api_keys should only include services with configured keys."""
    monkeypatch.setenv("NEWSAPI_KEY", "news-key")
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    available = get_available_api_keys("newsapi", "tiingo")
    assert available == {"newsapi": "news-key"}


def test_missing_api_keys_reports(monkeypatch):
    """missing_api_keys should return env names for services that are absent."""
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    missing = missing_api_keys(["alpha_vantage", "tiingo"])
    assert missing["alpha_vantage"] == "ALPHA_VANTAGE_API_KEY"
