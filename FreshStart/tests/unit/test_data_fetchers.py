import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.fetchers.price_data import get_historical_data, get_fundamentals


@pytest.fixture
def mock_yfinance_data():
    """Mock yfinance historical data."""
    dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
    return pd.DataFrame({
        'Open': [100 + i*0.5 for i in range(60)],
        'High': [105 + i*0.5 for i in range(60)],
        'Low': [95 + i*0.5 for i in range(60)],
        'Close': [102 + i*0.5 for i in range(60)],
        'Volume': [1000000 + i*10000 for i in range(60)]
    }, index=dates)


@pytest.fixture
def mock_fundamentals():
    """Mock fundamental data."""
    return {
        'trailingPE': 25.5,
        'forwardPE': 22.3,
        'marketCap': 2500000000000,
        'revenueGrowth': 0.12,
        'profitMargins': 0.25,
        'debtToEquity': 1.5,
        'returnOnEquity': 0.35,
        'beta': 1.2
    }


class TestPriceDataFetcher:
    """Test suite for price data fetcher."""

    @patch('data.fetchers.price_data.yf.Ticker')
    def test_get_historical_data_success(self, mock_ticker, mock_yfinance_data, mock_ticker):
        """Test successful historical data fetch."""
        mock_stock = MagicMock()
        mock_stock.history.return_value = mock_yfinance_data
        mock_ticker.return_value = mock_stock

        result = get_historical_data(mock_ticker, period="3mo")

        assert len(result) == 60
        assert 'date' in result[0]
        assert 'close' in result[0]
        assert 'open' in result[0]
        assert 'volume' in result[0]

    @patch('data.fetchers.price_data.yf.Ticker')
    def test_get_historical_data_invalid_ticker(self, mock_ticker):
        """Test handling of invalid ticker."""
        mock_stock = MagicMock()
        mock_stock.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_stock

        with pytest.raises(ValueError, match="No historical data found"):
            get_historical_data("INVALID", period="3mo")

    @patch('data.fetchers.price_data.yf.Ticker')
    def test_get_historical_data_api_failure(self, mock_ticker):
        """Test handling of API failures."""
        mock_ticker.side_effect = Exception("API connection failed")

        with pytest.raises(RuntimeError, match="Unable to fetch market data"):
            get_historical_data("AAPL", period="3mo")

    @patch('data.fetchers.price_data.yf.Ticker')
    def test_get_fundamentals_success(self, mock_ticker, mock_fundamentals):
        """Test successful fundamentals fetch."""
        mock_stock = MagicMock()
        mock_stock.info = mock_fundamentals
        mock_ticker.return_value = mock_stock

        result = get_fundamentals("AAPL")

        assert 'pe_ratio' in result
        assert 'market_cap' in result
        assert 'beta' in result
        assert result['pe_ratio'] == 25.5

    @patch('data.fetchers.price_data.yf.Ticker')
    def test_get_fundamentals_missing_data(self, mock_ticker):
        """Test handling of missing fundamental data."""
        mock_stock = MagicMock()
        mock_stock.info = {}
        mock_ticker.return_value = mock_stock

        result = get_fundamentals("AAPL")
        assert result['pe_ratio'] == 0
        assert result['beta'] == 1.0

    @patch('data.fetchers.price_data.yf.Ticker')
    def test_get_historical_data_different_periods(self, mock_ticker, mock_yfinance_data):
        """Test fetching data with different time periods."""
        mock_stock = MagicMock()
        mock_stock.history.return_value = mock_yfinance_data
        mock_ticker.return_value = mock_stock

        for period in ['1mo', '3mo', '6mo', '1y']:
            result = get_historical_data("AAPL", period=period)
            assert len(result) > 0
            mock_stock.history.assert_called_with(period=period, auto_adjust=True)


class TestNewsDataFetcher:
    """Test suite for news data fetcher (Tae's component)."""

    def test_news_fetcher_placeholder(self):
        """Placeholder test for news fetcher - will be implemented when Tae completes news_data.py."""
        pytest.skip("News data fetcher not yet implemented by Tae")


class TestDataFetcherIntegration:
    """Integration tests for data fetchers with database caching."""

    @patch('data.fetchers.price_data.yf.Ticker')
    def test_price_fetch_with_cache(self, mock_ticker, mock_yfinance_data, test_db):
        """Test fetching prices and caching to database."""
        mock_stock = MagicMock()
        mock_stock.history.return_value = mock_yfinance_data
        mock_ticker.return_value = mock_stock

        ticker = "AAPL"
        price_data = get_historical_data(ticker, period="3mo")

        df = pd.DataFrame(price_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High',
                          'low': 'Low', 'volume': 'Volume'}, inplace=True)

        inserted = test_db.cache_prices(ticker, df)
        assert inserted == len(price_data)

        cached = test_db.get_cached_prices(ticker)
        assert len(cached) == len(price_data)
