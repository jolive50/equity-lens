"""Data adapters following OOP principles for StockSense.

This module provides swappable data adapters for different financial data sources,
following the Strategy pattern and SOLID principles.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Protocol

import requests


@dataclass
class MarketData:
    """Standardized market data structure."""
    symbol: str
    price: float
    volume: int
    open_price: float
    high: float
    low: float
    close: float
    timestamp: str
    source: str


@dataclass
class NewsItem:
    """Standardized news item structure."""
    title: str
    content: str
    sentiment_score: Optional[float]
    source: str
    url: str
    timestamp: str
    ticker: str


@dataclass
class FundamentalData:
    """Standardized fundamental data structure."""
    symbol: str
    revenue_growth: Optional[float]
    ebitda_margin: Optional[float]
    debt_to_ebitda: Optional[float]
    pe_ratio: Optional[float]
    fcf_yield: Optional[float]
    timestamp: str
    source: str


class DataAdapter(ABC):
    """Abstract base class for all data adapters following the Strategy pattern."""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def fetch_market_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch current market data for a symbol."""
        pass
    
    @abstractmethod
    def fetch_historical_data(self, symbol: str, days: int = 30) -> List[MarketData]:
        """Fetch historical market data for a symbol."""
        pass
    
    @abstractmethod
    def fetch_news(self, symbol: str, limit: int = 100) -> List[NewsItem]:
        """Fetch news articles for a symbol."""
        pass
    
    @abstractmethod
    def fetch_fundamentals(self, symbol: str) -> Optional[FundamentalData]:
        """Fetch fundamental data for a symbol."""
        pass
    
    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Common request method with error handling."""
        try:
            params['apikey'] = self.api_key
            response = requests.get(f"{self.base_url}{endpoint}", params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Request failed for {endpoint}: {e}")
            return {}


class AlphaVantageAdapter(DataAdapter):
    """Alpha Vantage API adapter implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://www.alphavantage.co/query")
    
    def fetch_market_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch current market data from Alpha Vantage."""
        data = self._make_request("", {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol
        })
        
        if "Global Quote" not in data:
            return None
        
        quote = data["Global Quote"]
        return MarketData(
            symbol=symbol,
            price=float(quote.get("05. price", 0)),
            volume=int(quote.get("06. volume", 0)),
            open_price=float(quote.get("02. open", 0)),
            high=float(quote.get("03. high", 0)),
            low=float(quote.get("04. low", 0)),
            close=float(quote.get("05. price", 0)),
            timestamp=quote.get("07. latest trading day", ""),
            source="alpha_vantage"
        )
    
    def fetch_historical_data(self, symbol: str, days: int = 30) -> List[MarketData]:
        """Fetch historical data from Alpha Vantage."""
        data = self._make_request("", {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "compact"
        })
        
        if "Time Series (Daily)" not in data:
            return []
        
        time_series = data["Time Series (Daily)"]
        market_data = []
        
        for date_str, values in list(time_series.items())[:days]:
            market_data.append(MarketData(
                symbol=symbol,
                price=float(values.get("4. close", 0)),
                volume=int(values.get("5. volume", 0)),
                open_price=float(values.get("1. open", 0)),
                high=float(values.get("2. high", 0)),
                low=float(values.get("3. low", 0)),
                close=float(values.get("4. close", 0)),
                timestamp=date_str,
                source="alpha_vantage"
            ))
        
        return market_data
    
    def fetch_news(self, symbol: str, limit: int = 100) -> List[NewsItem]:
        """Fetch news from Alpha Vantage."""
        data = self._make_request("", {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "limit": min(limit, 1000)
        })
        
        if "feed" not in data:
            return []
        
        news_items = []
        for item in data["feed"][:limit]:
            news_items.append(NewsItem(
                title=item.get("title", ""),
                content=item.get("summary", ""),
                sentiment_score=self._extract_sentiment_score(item),
                source=item.get("source", ""),
                url=item.get("url", ""),
                timestamp=item.get("time_published", ""),
                ticker=symbol
            ))
        
        return news_items
    
    def fetch_fundamentals(self, symbol: str) -> Optional[FundamentalData]:
        """Fetch fundamental data from Alpha Vantage."""
        # Alpha Vantage doesn't have comprehensive fundamentals in free tier
        # This would need to be implemented with premium data or other sources
        return None
    
    def _extract_sentiment_score(self, item: Dict) -> Optional[float]:
        """Extract sentiment score from Alpha Vantage news item."""
        if "overall_sentiment_score" in item:
            return float(item["overall_sentiment_score"])
        return None


class TiingoAdapter(DataAdapter):
    """Tiingo API adapter implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://api.tiingo.com")
    
    def fetch_market_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch current market data from Tiingo."""
        data = self._make_request(f"/tiingo/daily/{symbol}/prices", {})
        
        if not data or not isinstance(data, list) or len(data) == 0:
            return None
        
        latest = data[0]
        return MarketData(
            symbol=symbol,
            price=float(latest.get("close", 0)),
            volume=int(latest.get("volume", 0)),
            open_price=float(latest.get("open", 0)),
            high=float(latest.get("high", 0)),
            low=float(latest.get("low", 0)),
            close=float(latest.get("close", 0)),
            timestamp=latest.get("date", ""),
            source="tiingo"
        )
    
    def fetch_historical_data(self, symbol: str, days: int = 30) -> List[MarketData]:
        """Fetch historical data from Tiingo."""
        data = self._make_request(f"/tiingo/daily/{symbol}/prices", {
            "startDate": "2024-01-01",
            "endDate": datetime.now().strftime("%Y-%m-%d")
        })
        
        if not data or not isinstance(data, list):
            return []
        
        market_data = []
        for item in data[-days:]:  # Get last N days
            market_data.append(MarketData(
                symbol=symbol,
                price=float(item.get("close", 0)),
                volume=int(item.get("volume", 0)),
                open_price=float(item.get("open", 0)),
                high=float(item.get("high", 0)),
                low=float(item.get("low", 0)),
                close=float(item.get("close", 0)),
                timestamp=item.get("date", ""),
                source="tiingo"
            ))
        
        return market_data
    
    def fetch_news(self, symbol: str, limit: int = 100) -> List[NewsItem]:
        """Fetch news from Tiingo."""
        data = self._make_request(f"/tiingo/news", {
            "tickers": symbol,
            "limit": min(limit, 1000)
        })
        
        if not data or not isinstance(data, list):
            return []
        
        news_items = []
        for item in data[:limit]:
            news_items.append(NewsItem(
                title=item.get("title", ""),
                content=item.get("description", ""),
                sentiment_score=None,  # Tiingo doesn't provide sentiment scores
                source=item.get("source", ""),
                url=item.get("url", ""),
                timestamp=item.get("publishedDate", ""),
                ticker=symbol
            ))
        
        return news_items
    
    def fetch_fundamentals(self, symbol: str) -> Optional[FundamentalData]:
        """Fetch fundamental data from Tiingo."""
        # Tiingo fundamentals would require premium subscription
        return None


class FinnhubAdapter(DataAdapter):
    """Finnhub API adapter implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://finnhub.io/api/v1")
    
    def fetch_market_data(self, symbol: str) -> Optional[MarketData]:
        """Fetch current market data from Finnhub."""
        data = self._make_request("/quote", {"symbol": symbol})
        
        if not data or "c" not in data:
            return None
        
        return MarketData(
            symbol=symbol,
            price=float(data.get("c", 0)),
            volume=int(data.get("v", 0)),
            open_price=float(data.get("o", 0)),
            high=float(data.get("h", 0)),
            low=float(data.get("l", 0)),
            close=float(data.get("c", 0)),
            timestamp=datetime.now().isoformat(),
            source="finnhub"
        )
    
    def fetch_historical_data(self, symbol: str, days: int = 30) -> List[MarketData]:
        """Fetch historical data from Finnhub."""
        end_time = int(datetime.now().timestamp())
        start_time = end_time - (days * 24 * 60 * 60)
        
        data = self._make_request("/stock/candle", {
            "symbol": symbol,
            "resolution": "D",
            "from": start_time,
            "to": end_time
        })
        
        if not data or "c" not in data:
            return []
        
        market_data = []
        closes = data.get("c", [])
        volumes = data.get("v", [])
        opens = data.get("o", [])
        highs = data.get("h", [])
        lows = data.get("l", [])
        timestamps = data.get("t", [])
        
        for i in range(len(closes)):
            market_data.append(MarketData(
                symbol=symbol,
                price=float(closes[i]),
                volume=int(volumes[i]) if i < len(volumes) else 0,
                open_price=float(opens[i]) if i < len(opens) else 0,
                high=float(highs[i]) if i < len(highs) else 0,
                low=float(lows[i]) if i < len(lows) else 0,
                close=float(closes[i]),
                timestamp=datetime.fromtimestamp(timestamps[i]).isoformat() if i < len(timestamps) else "",
                source="finnhub"
            ))
        
        return market_data
    
    def fetch_news(self, symbol: str, limit: int = 100) -> List[NewsItem]:
        """Fetch news from Finnhub."""
        data = self._make_request("/company-news", {
            "symbol": symbol,
            "from": "2024-01-01",
            "to": datetime.now().strftime("%Y-%m-%d")
        })
        
        if not data or not isinstance(data, list):
            return []
        
        news_items = []
        for item in data[:limit]:
            news_items.append(NewsItem(
                title=item.get("headline", ""),
                content=item.get("summary", ""),
                sentiment_score=None,  # Finnhub doesn't provide sentiment scores
                source=item.get("source", ""),
                url=item.get("url", ""),
                timestamp=item.get("datetime", ""),
                ticker=symbol
            ))
        
        return news_items
    
    def fetch_fundamentals(self, symbol: str) -> Optional[FundamentalData]:
        """Fetch fundamental data from Finnhub."""
        # Finnhub fundamentals would require premium subscription
        return None


class NewsAPIAdapter(DataAdapter):
    """NewsAPI adapter implementation."""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, "https://newsapi.org/v2")
    
    def fetch_market_data(self, symbol: str) -> Optional[MarketData]:
        """NewsAPI doesn't provide market data."""
        return None
    
    def fetch_historical_data(self, symbol: str, days: int = 30) -> List[MarketData]:
        """NewsAPI doesn't provide historical market data."""
        return []
    
    def fetch_news(self, symbol: str, limit: int = 100) -> List[NewsItem]:
        """Fetch news from NewsAPI."""
        data = self._make_request("/everything", {
            "q": f"{symbol} OR {symbol} stock",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(limit, 100)
        })
        
        if "articles" not in data:
            return []
        
        news_items = []
        for item in data["articles"][:limit]:
            news_items.append(NewsItem(
                title=item.get("title", ""),
                content=item.get("description", ""),
                sentiment_score=None,  # NewsAPI doesn't provide sentiment scores
                source=item.get("source", {}).get("name", ""),
                url=item.get("url", ""),
                timestamp=item.get("publishedAt", ""),
                ticker=symbol
            ))
        
        return news_items
    
    def fetch_fundamentals(self, symbol: str) -> Optional[FundamentalData]:
        """NewsAPI doesn't provide fundamental data."""
        return None


class DataAdapterFactory:
    """Factory class for creating data adapters following the Factory pattern."""
    
    @staticmethod
    def create_adapter(adapter_type: str, api_key: str) -> DataAdapter:
        """Create a data adapter instance based on type."""
        adapters = {
            "alpha_vantage": AlphaVantageAdapter,
            "tiingo": TiingoAdapter,
            "finnhub": FinnhubAdapter,
            "newsapi": NewsAPIAdapter,
        }
        
        if adapter_type not in adapters:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
        
        return adapters[adapter_type](api_key)


class DataService:
    """Service class that orchestrates multiple data adapters following the Facade pattern."""
    
    def __init__(self, adapters: Dict[str, DataAdapter]):
        self.adapters = adapters
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_market_data(self, symbol: str, source: str = "alpha_vantage") -> Optional[MarketData]:
        """Get market data from specified source."""
        if source not in self.adapters:
            self.logger.error(f"Unknown source: {source}")
            return None
        
        return self.adapters[source].fetch_market_data(symbol)
    
    def get_news(self, symbol: str, source: str = "newsapi") -> List[NewsItem]:
        """Get news from specified source."""
        if source not in self.adapters:
            self.logger.error(f"Unknown source: {source}")
            return []
        
        return self.adapters[source].fetch_news(symbol)
    
    def get_fundamentals(self, symbol: str, source: str = "alpha_vantage") -> Optional[FundamentalData]:
        """Get fundamentals from specified source."""
        if source not in self.adapters:
            self.logger.error(f"Unknown source: {source}")
            return None
        
        return self.adapters[source].fetch_fundamentals(symbol)
    
    def get_all_news(self, symbol: str) -> List[NewsItem]:
        """Get news from all available news sources."""
        all_news = []
        for source, adapter in self.adapters.items():
            try:
                news = adapter.fetch_news(symbol)
                all_news.extend(news)
            except Exception as e:
                self.logger.error(f"Failed to fetch news from {source}: {e}")
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_news = []
        for item in all_news:
            if item.url not in seen_urls:
                seen_urls.add(item.url)
                unique_news.append(item)
        
        return unique_news
