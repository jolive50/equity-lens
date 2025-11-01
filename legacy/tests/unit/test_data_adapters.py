"""Unit tests for StockSense data adapters."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from pipelines.realtime.data_adapters import (
    DataAdapterFactory,
    DataService,
    AlphaVantageAdapter,
    TiingoAdapter,
    FinnhubAdapter,
    NewsAPIAdapter,
    MarketData,
    NewsItem,
    FundamentalData,
)


class TestDataAdapterFactory:
    """Test cases for DataAdapterFactory."""
    
    def test_create_alpha_vantage_adapter(self):
        """Test creating Alpha Vantage adapter."""
        adapter = DataAdapterFactory.create_adapter("alpha_vantage", "test-key")
        assert isinstance(adapter, AlphaVantageAdapter)
        assert adapter.api_key == "test-key"
    
    def test_create_tiingo_adapter(self):
        """Test creating Tiingo adapter."""
        adapter = DataAdapterFactory.create_adapter("tiingo", "test-key")
        assert isinstance(adapter, TiingoAdapter)
        assert adapter.api_key == "test-key"
    
    def test_create_finnhub_adapter(self):
        """Test creating Finnhub adapter."""
        adapter = DataAdapterFactory.create_adapter("finnhub", "test-key")
        assert isinstance(adapter, FinnhubAdapter)
        assert adapter.api_key == "test-key"
    
    def test_create_newsapi_adapter(self):
        """Test creating NewsAPI adapter."""
        adapter = DataAdapterFactory.create_adapter("newsapi", "test-key")
        assert isinstance(adapter, NewsAPIAdapter)
        assert adapter.api_key == "test-key"
    
    def test_create_unknown_adapter(self):
        """Test creating unknown adapter type."""
        with pytest.raises(ValueError, match="Unknown adapter type"):
            DataAdapterFactory.create_adapter("unknown", "test-key")


class TestAlphaVantageAdapter:
    """Test cases for Alpha Vantage adapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = AlphaVantageAdapter("test-key")
    
    @patch('requests.get')
    def test_fetch_market_data_success(self, mock_get):
        """Test successful market data fetch."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "Global Quote": {
                "05. price": "150.00",
                "06. volume": "1000000",
                "02. open": "148.00",
                "03. high": "152.00",
                "04. low": "147.00",
                "07. latest trading day": "2024-01-24"
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.adapter.fetch_market_data("AAPL")
        
        assert isinstance(result, MarketData)
        assert result.symbol == "AAPL"
        assert result.price == 150.00
        assert result.volume == 1000000
        assert result.source == "alpha_vantage"
    
    @patch('requests.get')
    def test_fetch_market_data_no_data(self, mock_get):
        """Test market data fetch with no data."""
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.adapter.fetch_market_data("AAPL")
        
        assert result is None
    
    @patch('requests.get')
    def test_fetch_historical_data_success(self, mock_get):
        """Test successful historical data fetch."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "Time Series (Daily)": {
                "2024-01-24": {
                    "4. close": "150.00",
                    "5. volume": "1000000",
                    "1. open": "148.00",
                    "2. high": "152.00",
                    "3. low": "147.00"
                },
                "2024-01-23": {
                    "4. close": "149.00",
                    "5. volume": "950000",
                    "1. open": "147.00",
                    "2. high": "150.00",
                    "3. low": "146.00"
                }
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.adapter.fetch_historical_data("AAPL", 2)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(item, MarketData) for item in result)
        assert result[0].symbol == "AAPL"
        assert result[0].close == 150.00
    
    @patch('requests.get')
    def test_fetch_news_success(self, mock_get):
        """Test successful news fetch."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "feed": [
                {
                    "title": "Test News",
                    "summary": "Test summary",
                    "overall_sentiment_score": 0.8,
                    "source": "Test Source",
                    "url": "https://test.com",
                    "time_published": "2024-01-24T10:00:00Z"
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.adapter.fetch_news("AAPL", 1)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], NewsItem)
        assert result[0].title == "Test News"
        assert result[0].sentiment_score == 0.8
        assert result[0].ticker == "AAPL"


class TestTiingoAdapter:
    """Test cases for Tiingo adapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = TiingoAdapter("test-key")
    
    @patch('requests.get')
    def test_fetch_market_data_success(self, mock_get):
        """Test successful market data fetch."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "close": 150.00,
                "volume": 1000000,
                "open": 148.00,
                "high": 152.00,
                "low": 147.00,
                "date": "2024-01-24"
            }
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.adapter.fetch_market_data("AAPL")
        
        assert isinstance(result, MarketData)
        assert result.symbol == "AAPL"
        assert result.price == 150.00
        assert result.source == "tiingo"
    
    @patch('requests.get')
    def test_fetch_news_success(self, mock_get):
        """Test successful news fetch."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "title": "Test News",
                "description": "Test description",
                "source": "Test Source",
                "url": "https://test.com",
                "publishedDate": "2024-01-24T10:00:00Z"
            }
        ]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.adapter.fetch_news("AAPL", 1)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], NewsItem)
        assert result[0].title == "Test News"
        assert result[0].ticker == "AAPL"


class TestFinnhubAdapter:
    """Test cases for Finnhub adapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = FinnhubAdapter("test-key")
    
    @patch('requests.get')
    def test_fetch_market_data_success(self, mock_get):
        """Test successful market data fetch."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "c": 150.00,
            "v": 1000000,
            "o": 148.00,
            "h": 152.00,
            "l": 147.00
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.adapter.fetch_market_data("AAPL")
        
        assert isinstance(result, MarketData)
        assert result.symbol == "AAPL"
        assert result.price == 150.00
        assert result.source == "finnhub"
    
    @patch('requests.get')
    def test_fetch_historical_data_success(self, mock_get):
        """Test successful historical data fetch."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "c": [150.00, 149.00],
            "v": [1000000, 950000],
            "o": [148.00, 147.00],
            "h": [152.00, 150.00],
            "l": [147.00, 146.00],
            "t": [1706092800, 1706006400]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.adapter.fetch_historical_data("AAPL", 2)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(item, MarketData) for item in result)
        assert result[0].symbol == "AAPL"
        assert result[0].close == 150.00


class TestNewsAPIAdapter:
    """Test cases for NewsAPI adapter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = NewsAPIAdapter("test-key")
    
    def test_fetch_market_data_not_supported(self):
        """Test that market data is not supported."""
        result = self.adapter.fetch_market_data("AAPL")
        assert result is None
    
    def test_fetch_historical_data_not_supported(self):
        """Test that historical data is not supported."""
        result = self.adapter.fetch_historical_data("AAPL")
        assert result == []
    
    @patch('requests.get')
    def test_fetch_news_success(self, mock_get):
        """Test successful news fetch."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "articles": [
                {
                    "title": "Test News",
                    "description": "Test description",
                    "source": {"name": "Test Source"},
                    "url": "https://test.com",
                    "publishedAt": "2024-01-24T10:00:00Z"
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.adapter.fetch_news("AAPL", 1)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], NewsItem)
        assert result[0].title == "Test News"
        assert result[0].ticker == "AAPL"


class TestDataService:
    """Test cases for DataService."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_adapter = Mock()
        self.mock_adapter.fetch_market_data.return_value = MarketData(
            symbol="AAPL",
            price=150.0,
            volume=1000000,
            open_price=148.0,
            high=152.0,
            low=147.0,
            close=150.0,
            timestamp="2024-01-24",
            source="test"
        )
        self.mock_adapter.fetch_news.return_value = [
            NewsItem(
                title="Test News",
                content="Test content",
                sentiment_score=0.8,
                source="Test Source",
                url="https://test.com",
                timestamp="2024-01-24T10:00:00Z",
                ticker="AAPL"
            )
        ]
        
        self.service = DataService({"test": self.mock_adapter})
    
    def test_get_market_data_success(self):
        """Test successful market data retrieval."""
        result = self.service.get_market_data("AAPL", "test")
        
        assert isinstance(result, MarketData)
        assert result.symbol == "AAPL"
        self.mock_adapter.fetch_market_data.assert_called_once_with("AAPL")
    
    def test_get_market_data_unknown_source(self):
        """Test market data retrieval with unknown source."""
        result = self.service.get_market_data("AAPL", "unknown")
        assert result is None
    
    def test_get_news_success(self):
        """Test successful news retrieval."""
        result = self.service.get_news("AAPL", "test")
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], NewsItem)
        self.mock_adapter.fetch_news.assert_called_once_with("AAPL")
    
    def test_get_all_news_success(self):
        """Test successful news retrieval from all sources."""
        result = self.service.get_all_news("AAPL")
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], NewsItem)
        self.mock_adapter.fetch_news.assert_called_once_with("AAPL")
    
    def test_get_all_news_with_error(self):
        """Test news retrieval with adapter error."""
        self.mock_adapter.fetch_news.side_effect = Exception("API Error")
        
        result = self.service.get_all_news("AAPL")
        
        assert isinstance(result, list)
        assert len(result) == 0  # Should return empty list on error


class TestDataStructures:
    """Test cases for data structures."""
    
    def test_market_data_creation(self):
        """Test MarketData creation."""
        data = MarketData(
            symbol="AAPL",
            price=150.0,
            volume=1000000,
            open_price=148.0,
            high=152.0,
            low=147.0,
            close=150.0,
            timestamp="2024-01-24",
            source="test"
        )
        
        assert data.symbol == "AAPL"
        assert data.price == 150.0
        assert data.volume == 1000000
        assert data.source == "test"
    
    def test_news_item_creation(self):
        """Test NewsItem creation."""
        item = NewsItem(
            title="Test News",
            content="Test content",
            sentiment_score=0.8,
            source="Test Source",
            url="https://test.com",
            timestamp="2024-01-24T10:00:00Z",
            ticker="AAPL"
        )
        
        assert item.title == "Test News"
        assert item.content == "Test content"
        assert item.sentiment_score == 0.8
        assert item.ticker == "AAPL"
    
    def test_fundamental_data_creation(self):
        """Test FundamentalData creation."""
        data = FundamentalData(
            symbol="AAPL",
            revenue_growth=0.08,
            ebitda_margin=0.28,
            debt_to_ebitda=2.1,
            pe_ratio=25.5,
            fcf_yield=0.04,
            timestamp="2024-01-24",
            source="test"
        )
        
        assert data.symbol == "AAPL"
        assert data.revenue_growth == 0.08
        assert data.pe_ratio == 25.5
        assert data.source == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
