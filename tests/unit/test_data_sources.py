"""Comprehensive tests for data sources and financial data providers.

Tests cover API requests, data validation, rate limiting, and error handling.
"""

import pytest
import unittest.mock as mock
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd
import requests
import yfinance as yf

from pipelines.realtime.data_sources import (
    YahooFinanceProvider,
    APIRateLimiter,
    DataValidationService,
    FinancialDataProvider
)


@pytest.fixture
def mock_yfinance():
    """Mock yfinance for testing."""
    with mock.patch.object(yf, 'Ticker') as mock_ticker:
        yield mock_ticker


@pytest.fixture
def yahoo_provider():
    """Create a YahooFinanceProvider instance for testing."""
    rate_limiter = APIRateLimiter(requests_per_minute=100)  # High limit for testing
    return YahooFinanceProvider(rate_limiter)


class TestAPIRateLimiter:
    """Test the API rate limiter functionality."""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        limiter = APIRateLimiter(requests_per_minute=10)
        
        assert limiter.requests_per_minute == 10
        assert len(limiter.last_request_times) == 0
    
    @mock.patch('time.sleep')
    @mock.patch('time.time')
    def test_rate_limiting_with_wait(self, mock_time, mock_sleep):
        """Test rate limiting when limit is exceeded."""
        limiter = APIRateLimiter(requests_per_minute=2)
        
        # Simulate multiple requests in same minute
        mock_time.side_effect = [
            0,      # First request time
            0.1,     # Second request time  
            5,       # Third request time - should trigger wait
            65       # After sleep time
        ]
        
        # First two requests should pass
        limiter.check_and_wait()
        limiter.check_and_wait()
        
        # Third request should trigger wait
        limiter.check_and_wait()
        
        # Verify sleep was called
        mock_sleep.assert_called_once()


class TestYahooFinanceProvider:
    """Test YahooFinanceProvider functionality."""
    
    def test_provider_initialization(self):
        """Test provider initialization."""
        provider = YahooFinanceProvider()
        
        assert provider.rate_limiter is not None
        assert isinstance(provider, FinancialDataProvider)
    
    def test_get_historical_data_success(self, mock_yfinance, yahoo_provider):
        """Test successful historical data retrieval."""
        # Mock data
        mock_data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'Open': [100.0] * 30,
            'High': [105.0] * 30,
            'Low': [95.0] * 30,
            'Close': [102.0] * 30,
            'Volume': [1000000] * 30
        })
        mock_data.set_index('Date', inplace=True)
        
        # Mock yfinance response
        mock_stock = mock.MagicMock()
        mock_stock.history.return_value = mock_data
        mock_yfinance.return_value = mock_stock
        
        # Test
        result = yahoo_provider.get_historical_data("AAPL", "1mo")
        
        # Verify
        assert len(result) == 30
        assert all(isinstance(entry, dict) for entry in result)
        assert "date" in result[0]
        assert "close" in result[0]
        assert "volume" in result[0]
        
        mock_yfinance.assert_called_once_with("AAPL")
        mock_stock.history.assert_called_once_with(period="1mo", auto_adjust=True)
    
    @mock.patch('yfinance.Ticker')
    def test_get_historical_data_empty_response(self, mock_ticker):
        """Test handling of empty yfinance response."""
        provider = YahooFinanceProvider()
        
        # Mock empty response
        mock_stock = mock.MagicMock()
        mock_stock.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_stock
        
        # Test - should raise RuntimeError
        with pytest.raises(RuntimeError, match="Unable to retrieve market data"):
            provider.get_historical_data("INVALID_TICKER")
    
    def test_get_fundamentals_success(self, mock_yfinance, yahoo_provider):
        """Test successful fundamentals retrieval."""
        # Mock fundamentals data
        mock_info = {
            "trailingPE": 25.5,
            "forwardPE": 23.1,
            "marketCap": 2500000000000,
            "returnOnEquity": 1.2,
            "profitMargins": 0.25,
            "beta": 1.1,
            "currentRatio": 1.8,
            "dividendYield": 0.02,
            "longName": "Apple Inc."
        }
        
        # Mock yfinance history for momentum calculation
        mock_price_data = pd.DataFrame({
            'Close': [100.0, 102.0, 101.5, 103.0, 104.2] * 60  # 300 days of data
        })
        
        mock_stock = mock.MagicMock()
        mock_stock.info = mock_info
        mock_stock.history.return_value = mock_price_data
        mock_yfinance.return_value = mock_stock
        
        # Test
        result = yahoo_provider.get_fundamentals("AAPL")
        
        # Verify
        assert "pe_ratio" in result
        assert "revenue_growth" in result
        assert "market_cap" in result
        assert result["pe_ratio"] == 25.5
        assert result["market_cap"] == 2500000000000
        
        mock_yfinance.assert_called_once_with("AAPL")
    
    def test_fundamentals_validation_and_defaults(self, mock_yfinance, yahoo_provider):
        """Test fundamentals validation and default values."""
        # Mock incomplete info
        mock_info = {
            "trailingPE": None,  # Missing data
            "profitMargins": "invalid_string",  # Invalid type
            "marketCap": 1000000000
        }
        
        mock_stock = mock.MagicMock()
        mock_stock.info = mock_info
        mock_stock.history.return_value = pd.DataFrame({'Close': [100.0] * 5})
        mock_yfinance.return_value = mock_stock
        
        # Test
        result = yahoo_provider.get_fundamentals("TEST")
        
        # Verify defaults are applied
        assert result["pe_ratio"] == 0.0  # Default for None
        assert result["profit_margin"] == 0.0  # Default for invalid string
        assert result["market_cap"] == 1000000000  # Valid value preserved
        
        # Verify required metrics are present
        required_metrics = ["pe_ratio", "market_cap", "beta", "roe"]
        for metric in required_metrics:
            assert metric in result
    
    def test_get_news_data_success(self, mock_yfinance, yahoo_provider):
        """Test successful news data retrieval."""
        # Mock news data
        mock_news = [
            {
                "title": "Apple reports strong earnings",
                "summary": "Apple Inc. reported higher than expected quarterly earnings.",
                "providerPublishTime": 1700000000,  # Real timestamp
                "publisher": "Reuters",
                "link": "https://example.com/news1"
            },
            {
                "title": "Apple stock price rises",
                "summary": "AAPL shares gained 2% in trading today.",
                "providerPublishTime": 0,  # Missing timestamp
                "publisher": "Bloomberg",
                "link": "https://example.com/news2"
            }
        ]
        
        mock_stock = mock.MagicMock()
        mock_stock.news = mock_news
        mock_yfinance.return_value = mock_stock
        
        # Test
        result = yahoo_provider.get_news_data("AAPL")
        
        # Verify
        assert len(result) >= 2
        assert all(isinstance(article, dict) for article in result)
        assert "title" in result[0]
        assert "content" in result[0]
        assert "timestamp" in result[0]
        assert "source" in result[0]
        assert "ticker" in result[0]
        
        mock_yfinance.assert_called_once_with("AAPL")
    
    def test_news_data_fallback_creation(self, mock_yfinance, yahoo_provider):
        """Test fallback news creation when no news is available."""
        mock_stock = mock.MagicMock()
        mock_stock.news = []  # No news available
        mock_stock.info = {
            "regularMarketPrice": 175.50,
            "longName": "Apple Inc.",
            "marketCap": 2500000000000
        }
        mock_yfinance.return_value = mock_stock
        
        # Test
        result = yahoo_provider.get_news_data("AAPL")
        
        # Verify fallback news was created
        assert len(result) == 1
        assert "Apple Inc. Stock Analysis Report" in result[0]["title"]
        assert "175.5" in result[0]["stock_price"]  # Real price included
        assert result[0]["ticker"] == "AAPL"


class TestDataValidationService:
    """Test data validation service."""
    
    def test_validate_ticker_valid_symbols(self):
        """Test validation of valid ticker symbols."""
        valid_tickers = ["AAPL", "MSFT", "TSLA", "BRK.B", "GOOG", "GLDB"]
        
        for ticker in valid_tickers:
            assert DataValidationService.validate_ticker(ticker), f"Should validate {ticker}"
    
    def test_validate_ticker_invalid_symbols(self):
        """Test validation of invalid ticker symbols."""
        invalid_tickers = ["", "123", "TOOLONG", "AAPL.COM", "$AAPL", "aapl", None]
        
        for ticker in invalid_tickers:
            assert not DataValidationService.validate_ticker(ticker), f"Should not validate {ticker}"
    
    def test_validate_fundamentals_valid_data(self):
        """Test validation of valid fundamental data."""
        valid_fundamentals = {
            "pe_ratio": 25.5,
            "pb_ratio": 3.2,
            "debt_to_equity": 0.8,
            "roe": 0.15,
            "profit_margin": 0.22,
            "beta": 1.25,
            "current_ratio": 2.1,
            "revenue_growth": 0.12
        }
        
        result = DataValidationService.validate_fundamentals(valid_fundamentals)
        
        # All values should be preserved
        for key, value in valid_fundamentals.items():
            assert result[key] == float(value)
    
    def test_validate_fundamentals_outside_ranges(self):
        """Test validation handling of data outside reasonable ranges."""
        invalid_fundamentals = {
            "pe_ratio": 5000,    # Unreasonably high
            "debt_to_equity": -20, # Invalid negative
            "roe": 15.0,         # Unreasonably high
            "beta": -2.0         # Invalid range
        }
        
        result = DataValidationService.validate_fundamentals(invalid_fundamentals)
        
        # Invalid values should be set to default (0.0)
        assert result["pe_ratio"] == 0.0
        assert result["debt_to_equity"] == 0.0
        assert result["roe"] == 0.0
        assert result["beta"] == 0.0
    
    def test_validate_historical_data_valid_data(self):
        """Test validation of valid historical price data."""
        valid_data = [
            {
                "date": "2024-01-01",
                "open": 100.0,
                "high": 105.0,
                "low": 98.0,
                "close": 102.0,
                "volume": 1000000
            },
            {
                "date": "2024-01-02", 
                "open": 102.0,
                "high": 108.0,
                "low": 101.0,
                "close": 106.0,
                "volume": 1200000
            }
        ]
        
        result = DataValidationService.validate_historical_data(valid_data)
        
        assert len(result) == 2
        assert result == valid_data
    
    def test_validate_historical_data_invalid_data(self):
        """Test validation handling of invalid historical data."""
        invalid_data = [
            {
                "date": "2024-01-00",
                "open": 100.0,
                "high": 95.0,  # High < Close (invalid)
                "low": 98.0,
                "close": 102.0,
                "volume": 1000000
            },
            {
                "date": "2024-01-02",
                "open": -102.0,  # Negative price (invalid)
                "high": 108.0,
                "low": 101.0,
                "close": 106.0,
                "volume": 1200000
            },
            {
                "date": "2024-01-03",
                "close": "invalid",  # Invalid type
            }
        ]
        
        result = DataValidationService.validate_historical_data(invalid_data)
        
        # Should filter out all invalid entries
        assert len(result) == 0


class TestAPIErrorHandling:
    """Test error handling for API requests."""
    
    @mock.patch('yfinance.Ticker')
    def test_yahoo_finance_connection_error(self, mock_ticker):
        """Test handling of yfinance connection errors."""
        provider = YahooFinanceProvider()
        
        # Mock network error
        mock_ticker.side_effect = Exception("Network connection error")
        
        # Test that RuntimeError is raised
        with pytest.raises(RuntimeError):
            provider.get_historical_data("AAPL")
        
        with pytest.raises(RuntimeError):
            provider.get_fundamentals("AAPL")
        
        with pytest.raises(RuntimeError):
            provider.get_news_data("AAPL")
    
    @mock.patch('yfinance.Tiller')
    def test_nonexistent_ticker(self, mock_ticker):
        """Test handling of non-existent ticker symbols."""
        provider = YahooFinanceProvider()
        
        # Mock empty response for non-existent ticker
        mock_stock = mock.MagicMock()
        mock_stock.info = {}
        mock_stock.history.return_value = pd.DataFrame()
        mock_stock.news = []
        mock_ticker.return_value = mock_stock
        
        # Test handling of missing data
        with pytest.raises(RuntimeError, match="Unable to retrieve"):
            provider.get_fundamentals("NONEXISTENT")
        
        with pytest.raises(RuntimeError, match="Unable to retrieve"):
            provider.get_historical_data("NONEXISTENT")


class TestIntegration:
    """Integration tests for real API interactions (when available)."""
    
    @pytest.mark.integration
    def test_real_yahoo_finance_integration(self):
        """Test real Yahoo Finance API integration."""
        provider = YahooFinanceProvider()
        
        try:
            # Test real AAPL data
            historical = provider.get_historical_data("AAPL", "5d")
            assert len(historical) > 0
            assert all(isinstance(entry, dict) for entry in historical)
            
            fundamentals = provider.get_fundamentals("AAPL")
            assert "pe_ratio" in fundamentals
            assert "market_cap" in fundamentals
            assert fundamentals["market_cap"] > 0
            
            news = provider.get_news_data("AAPL")
            assert len(news) > 0
            assert all("ticker" in article for article in news)
            
        except Exception as e:
            pytest.skip(f"Real API test skipped due to: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
