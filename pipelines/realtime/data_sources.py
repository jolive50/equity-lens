"""Data source abstractions for financial data retrieval.

This module implements the Strategy pattern for different financial data sources,
following SOLID principles for clean, maintainable code.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Protocol
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class APIRateLimiter:
    """Rate limiter for API requests following Single Responsibility Principle."""
    
    def __init__(self, requests_per_minute: int = 12):
        self.requests_per_minute = requests_per_minute
        self.last_request_times: List[float] = []
    
    def check_and_wait(self) -> None:
        """Check rate limits and wait if necessary."""
        import time
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.last_request_times = [
            req_time for req_time in self.last_request_times 
            if current_time - req_time < 60
        ]
        
        # If we've hit the limit, wait
        if len(self.last_request_times) >= self.requests_per_minute:
            sleep_time = 60 - (current_time - min(self.last_request_times))
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
        
        self.last_request_times.append(current_time)


class FinancialDataProvider(ABC):
    """Abstract base class for financial data providers (Strategy pattern)."""
    
    @abstractmethod
    def get_historical_data(self, ticker: str, period: str = "3mo") -> List[Dict[str, Any]]:
        """Fetch historical price data for a ticker."""
        pass
    
    @abstractmethod
    def get_fundamentals(self, ticker: str) -> Dict[str, float]:
        """Fetch fundamental metrics for a ticker."""
        pass
    
    @abstractmethod
    def get_news_data(self, ticker: str, days: int = 7) -> List[Dict[str, str]]:
        """Fetch recent news for a ticker."""
        pass


class YahooFinanceProvider(FinancialDataProvider):
    """Yahoo Finance implementation of FinancialDataProvider."""
    
    def __init__(self, rate_limiter: Optional[APIRateLimiter] = None):
        self.rate_limiter = rate_limiter or APIRateLimiter(requests_per_minute=60)  # Yahoo is more generous
    
    def get_historical_data(self, ticker: str, period: str = "3mo") -> List[Dict[str, Any]]:
        """Fetch real historical price data for a ticker using yfinance."""
        import yfinance as yf
        
        self.rate_limiter.check_and_wait()
        
        try:
            logger.info(f"Fetching real historical data for {ticker} with period {period}")
            stock = yf.Ticker(ticker)
            
            # Fetch real data with extended period for better analysis
            hist = stock.history(period=period, auto_adjust=True)
            
            if hist.empty:
                logger.error(f"No historical data returned for {ticker}")
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Convert to list of dictionaries with real data
            data = []
            for date, row in hist.iterrows():
                data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "close": round(float(row["Close"]), 2),
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "volume": int(row["Volume"]),
                    "adj_close": round(float(row["Close"]), 2)
                })
            
            logger.info(f"Successfully fetched {len(data)} days of real historical data for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {ticker}: {e}")
            raise RuntimeError(f"Unable to retrieve market data for {ticker}: {str(e)}")
    
    def get_fundamentals(self, ticker: str) -> Dict[str, float]:
        """Fetch real fundamental metrics for a ticker from Yahoo Finance."""
        import yfinance as yf
        
        self.rate_limiter.check_and_wait()
        
        try:
            logger.info(f"Fetching real fundamental metrics for {ticker}")
            
            # Primary source: Yahoo Finance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info:
                raise ValueError(f"No fundamental data available for {ticker}")
            
            fundamentals = self._extract_fundamentals(info)
            fundamentals.update(self._calculate_momentum_metrics(stock))
            
            # Fill any missing critical metrics
            self._ensure_required_metrics(fundamentals, ticker)
            
            logger.info(f"Successfully fetched {len(fundamentals)} real fundamental metrics for {ticker}")
            return fundamentals
            
        except Exception as e:
            logger.error(f"Failed to fetch fundamentals for {ticker}: {e}")
            raise RuntimeError(f"Unable to retrieve fundamental data for {ticker}: {str(e)}")
    
    def get_news_data(self, ticker: str, days: int = 7) -> List[Dict[str, str]]:
        """Fetch real recent news for a ticker from Yahoo Finance."""
        import yfinance as yf
        import time
        
        try:
            logger.info(f"Fetching real news data for {ticker}")
            
            stock = yf.Ticker(ticker)
            processed_news = []
            
            # Yahoo Finance news
            try:
                yahoo_news = stock.news
                
                if not yahoo_news:
                    logger.warning(f"No Yahoo Finance news found for {ticker}")
                else:
                    for article in yahoo_news[:15]:  # Get more articles for better analysis
                        try:
                            publish_time = article.get("providerPublishTime", 0)
                            if publish_time == 0:
                                publish_time = int(time.time())  # Current time if no timestamp
                            
                            processed_news.append({
                                "title": article.get("title", f"News about {ticker}"),
                                "content": article.get("summary", article.get("title", "")),
                                "timestamp": datetime.fromtimestamp(publish_time).isoformat(),
                                "source": article.get("publisher", "Yahoo Finance"),
                                "link": article.get("link", ""),
                                "type": "financial_news",
                                "ticker": ticker
                            })
                        except Exception as article_error:
                            logger.warning(f"Error processing article for {ticker}: {article_error}")
                            continue
            except Exception as yahoo_error:
                logger.warning(f"Yahoo Finance news fetch failed for {ticker}: {yahoo_error}")
            
            # Sort by timestamp (most recent first)
            processed_news.sort(key=lambda x: x.get("timestamp", "1970-01-01"), reverse=True)
            
            if not processed_news:
                processed_news = self._create_fallback_news(stock, ticker)
            
            logger.info(f"Successfully retrieved {len(processed_news)} news articles for {ticker}")
            return processed_news
            
        except Exception as e:
            logger.error(f"Failed to fetch news for {ticker}: {e}")
            raise RuntimeError(f"Unable to retrieve news data for {ticker}: {str(e)}")
    
    def _extract_fundamentals(self, info: Dict) -> Dict[str, float]:
        """Extract fundamental metrics from Yahoo Finance info."""
        fundamentals = {}
        
        # Extract key metrics with validation
        metrics_mapping = {
            "revenue_growth": ("revenueGrowth", 0.0),
            "ebitda_margin": ("profitMargins", 0.0),
            "pe_ratio": ("trailingPE", 0.0),
            "forward_pe": ("forwardPE", 0.0),
            "debt_to_equity": ("debtToEquity", 0.0),
            "roe": ("returnOnEquity", 0.0),
            "profit_margin": ("profitMargins", 0.0),
            "market_cap": ("marketCap", 0),
            "enterprise_value": ("enterpriseValue", 0),
            "ebitda": ("ebitda", 0),
            "current_ratio": ("currentRatio", 0.0),
            "beta": ("beta", 1.0),
            "dividend_yield": ("dividendYield", 0.0),
        }
        
        for key, (yfinance_key, default_value) in metrics_mapping.items():
            try:
                value = info.get(yfinance_key)
                if value is not None:
                    if isinstance(value, (int, float)):
                        fundamentals[key] = float(value)
                    elif isinstance(value, str) and value.replace('-', '').replace('.', '').isdigit():
                        fundamentals[key] = float(value)
                    else:
                        fundamentals[key] = default_value
                else:
                    fundamentals[key] = default_value
            except (ValueError, TypeError):
                fundamentals[key] = default_value
        
        return fundamentals
    
    def _calculate_momentum_metrics(self, stock) -> Dict[str, float]:
        """Calculate momentum metrics from real price data."""
        import yfinance as yf
        
        momentum_metrics = {
            "price_momentum_20d": 0.0,
            "price_momentum_60d": 0.0,
            "price_momentum_12m": 0.0
        }
        
        try:
            price_data = stock.history(period="1y")
            if not price_data.empty and len(price_data) > 20:
                current_price = price_data["Close"].iloc[-1]
                
                # 20-day momentum
                if len(price_data) >= 21:
                    start_price = price_data["Close"].iloc[-21]
                    momentum_metrics["price_momentum_20d"] = round((current_price - start_price) / start_price, 3)
                
                # 60-day momentum
                if len(price_data) >= 61:
                    start_price = price_data["Close"].iloc[-61]
                    momentum_metrics["price_momentum_60d"] = round((current_price - start_price) / start_price, 3)
                
                # 12-month momentum
                if len(price_data.columns) > 0 and len(price_data) >= 253:
                    start_price = price_data["Close"].iloc[-253]
                    momentum_metrics["price_momentum_12m"] = round((current_price - start_price) / start_price, 3)
                
                logger.info(f"Calculated momentum metrics for {stock.ticker}")
            else:
                logger.warning("Insufficient price data for momentum calculation")
                
        except Exception as momentum_error:
            logger.warning(f"Could not calculate momentum: {momentum_error}")
        
        return momentum_metrics
    
    def _ensure_required_metrics(self, fundamentals: Dict[str, float], ticker: str) -> None:
        """Ensure all required metrics are present."""
        required_metrics = ["pe_ratio", "market_cap", "beta", "roe"]
        for metric in required_metrics:
            if metric not in fundamentals:
                logger.warning(f"Missing {metric} for {ticker}, using default")
                fundamentals[metric] = 0.0
    
    def _create_fallback_news(self, stock, ticker: str) -> List[Dict[str, str]]:
        """Create fallback news using real stock data."""
        try:
            info = stock.info
            stock_price = info.get('regularMarketPrice', 'N/A')
            company_name = info.get('longName', ticker)
            
            return [{
                "title": f"{company_name} Stock Analysis Report",
                "content": f"{company_name} stock analysis completed using real market data.",
                "timestamp": datetime.utcnow().isoformat(),
                "source": "Market Analysis Engine",
                "type": "report",
                "stock_price": str(stock_price),
                "ticker": ticker
            }]
        except Exception as e:
            logger.error(f"Could not create fallback news for {ticker}: {e}")
            return [{
                "title": f"{ticker} Stock Analysis",
                "content": f"{ticker} stock analysis report generated",
                "timestamp": datetime.utcnow().isoformat(),
                "source": "Market Analysis Engine",
                "type": "report",
                "ticker": ticker
            }]


class DataValidationService:
    """Service for validating financial data quality."""
    
    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """Validate that ticker symbol is properly formatted."""
        if not ticker or not isinstance(ticker, str):
            return False
        
        # Basic validation: 1-5 characters, uppercase letters
        return len(ticker.strip()) <= 5 and ticker.strip().isalpha()
    
    @staticmethod
    def validate_fundamentals(fundamentals: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean fundamental metrics."""
        validated = {}
        
        # Define reasonable ranges for key metrics
        validations = {
            "pe_ratio": (0, 1000),
            "pb_ratio": (0, 100),
            "debt_to_equity": (-10, 50),
            "roe": (-1, 10),
            "profit_margin": (-1, 1),
            "beta": (0, 5),
            "current_ratio": (0, 20),
            "revenue_growth": (-2, 5),
        }
        
        for key, value in fundamentals.items():
            if isinstance(value, (int, float)):
                # Check against valid ranges
                if key.lower() in validations:
                    min_val, max_val = validations[key.lower()]
                    if min_val <= value <= max_val:
                        validated[key] = float(value)
                    else:
                        logger.warning(f"{key} value {value} outside reasonable range [{min_val}, {max_val}]")
                        validated[key] = 0.0
                else:
                    validated[key] = float(value)
            else:
                logger.warning(f"Invalid type for {key}: {type(value)}")
                validated[key] = 0.0
        
        return validated
    
    @staticmethod
    def validate_historical_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate historical price data."""
        validated = []
        
        for entry in data:
            if not isinstance(entry, dict):
                continue
            
            required_keys = ["date", "close", "open", "high", "low", "volume"]
            if all(key in entry for key in required_keys):
                # Validate price relationships
                close = float(entry["close"])
                open_price = float(entry["open"])
                high = float(entry["high"])
                low = float(entry["low"])
                volume = int(entry["volume"])
                
                # Basic sanity checks
                if (low <= close <= high and 
                    low <= open_price <= high and 
                    volume >= 0 and
                    close > 0 and open_price > 0 and high > 0 and low > 0):
                    validated.append(entry)
                else:
                    logger.warning(f"Skipping invalid price data: {entry}")
        
        return validated
