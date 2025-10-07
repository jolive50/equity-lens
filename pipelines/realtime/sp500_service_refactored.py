"""Refactored S&B 500 data service following SOLID principles.

This service orchestrates data retrieval and analysis for multiple S&P 500 companies,
using dependency injection and interface segregation for maintainable, testable code.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .data_sources import (
    YahooFinanceProvider, 
    APIRateLimiter, 
    DataValidationService,
    FinancialDataProvider
)

from .api_keys import get_available_api_keys

logger = logging.getLogger(__name__)


class TickerListProvider:
    """Provides S&P 500 ticker lists (Single Responsibility Principle)."""
    
    DEFAULT_TICKERS = [
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "TSLA", "META", 
        "UNH", "JNJ", "V", "PG", "JPM", "MA", "HD", "DIS", "PYPL", "NFLX",
        "XOM", "BAC", "ABBV", "CVX", "LLY", "AVGO", "KO", "PFE", "PEP",
        "COST", "MRK", "CRM", "ABT", "ACN", "WMT", "CSCO"
    ]
    
    @staticmethod
    def get_sp500_tickers(count: int = 5) -> List[str]:
        """Get a list of popular S&P 500 tickers for analysis."""
        return TickerListProvider.DEFAULT_TICKERS[:count]
    
    @staticmethod
    def validate_ticker_list(tickers: List[str]) -> List[str]:
        """Validate and clean ticker list."""
        validated = []
        for ticker in tickers:
            if DataValidationService.validate_ticker(ticker):
                validated.append(ticker.upper())
            else:
                logger.warning(f"Invalid ticker symbol skipped: {ticker}")
        return validated


class FinancialDataValidator:
    """Validates financial data quality (Single Responsibility Principle)."""
    
    @staticmethod
    def process_ticker_data(ticker: str, historical_data: List[Dict], 
                           fundamentals: Dict[str, float], 
                           news_data: List[Dict]) -> Dict[str, Any]:
        """Process and validate data for a single ticker."""
        
        # Validate each data component
        validated_historical = DataValidationService.validate_historical_data(historical_data)
        validated_fundamentals = DataValidationService.validate_fundamentals(fundamentals)
        
        return {
            "ticker": ticker.upper(),
            "market_data": validated_historical,
            "fundamentals": validated_fundamentals, 
            "news_data": news_data,  # News doesn't need numerical validation
            "data_quality": {
                "historical_data_points": len(validated_historical),
                "fundamental_metrics_count": len(validated_fundamentals),
                "news_articles_count": len(news_data),
                "validation_timestamp": datetime.utcnow().isoformat()
            }
        }


class SP500DataService:
    """Main service for S&P 500 data retrieval (Open/Closed Principle).
    
    This service can be extended with new data providers without modifying
    its core functionality.
    """
    
    def __init__(self, 
                 data_provider: Optional[FinancialDataProvider] = None,
                 validator: Optional[Any] = None,
                 ticker_provider: Optional[TickerListProvider] = None):
        """Initialize with dependency injection.
        
        Args:
            data_provider: Financial data provider implementation
            validator: Data validator instance
            ticker_provider: Ticker list provider instance
        """
        self.data_provider = data_provider or YahooFinanceProvider(
            rate_limiter=APIRateLimiter(requests_per_minute=12)
        )
        self.validator = validator or FinancialDataValidator()
        self.ticker_provider = ticker_provider or TickerListProvider()
        
        logger.info("SP500DataService initialized with dependencies")
    
    def get_single_ticker_data(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing historical data, fundamentals, and news
            
        Raises:
            ValueError: If ticker is invalid
            RuntimeError: If data retrieval fails
        """
        if not DataValidationService.validate_ticker(ticker):
            raise ValueError(f"Invalid ticker symbol: {ticker}")
        
        ticker = ticker.upper()
        logger.info(f"Retrieving comprehensive data for {ticker}")
        
        try:
            # Retrieve data from provider
            historical_data = self.data_provider.get_historical_data(ticker)
            fundamentals = self.data_provider.get_fundamentals(ticker)
            news_data = self.data_provider.get_news_data(ticker)
            
            # Validate and process data
            processed_data = self.validator.process_ticker_data(
                ticker, historical_data, fundamentals, news_data
            )
            
            logger.info(f"Successfully retrieved data for {ticker}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Failed to retrieve data for {ticker}: {e}")
            raise RuntimeError(f"Data retrieval failed for {ticker}: {str(e)}")
    
    def get_multiple_tickers_data(self, tickers: List[str], 
                                 max_tickers: int = 8) -> Dict[str, Dict[str, Any]]:
        """Get data for multiple tickers with rate limiting.
        
        Args:
            tickers: List of ticker symbols
            max_tickers: Maximum number of tickers to process
            
        Returns:
            Dictionary mapping ticker symbols to their data
            
        Raises:
            ValueError: If no valid tickers provided
            RuntimeError: If all data retrievals fail
        """
        # Validate and limit ticker list
        validated_tickers = self.ticker_provider.validate_ticker_list(tickers)[:max_tickers]
        
        if not validated_tickers:
            raise ValueError("No valid ticker symbols provided")
        
        logger.info(f"Processing {len(validated_tickers)} tickers: {validated_tickers}")
        
        results = {}
        errors = []
        
        for i, ticker in enumerate(validated_tickers):
            try:
                logger.info(f"Processing {ticker} ({i+1}/{len(validated_tickers)})")
                ticker_data = self.get_single_ticker_data(ticker)
                results[ticker] = ticker_data
                
                # Add small delay between requests to be respectful
                import time
                time.sleep(0.5)
                
            except Exception as ticker_error:
                logger.error(f"Failed to process {ticker}: {ticker_error}")
                errors.append(f"{ticker}: {str(ticker_error)}")
                continue
        
        if not results:
            raise RuntimeError(f"Failed to retrieve data for any ticker. Errors: {errors}")
        
        logger.info(f"Successfully processed {len(results)}/{len(validated_tickers)} tickers")
        return results
    
    def get_sp500_sample(self, count: int = 5) -> Dict[str, Dict[str, Any]]:
        """Get data for a sample of S&P 500 companies.
        
        Args:
            count: Number of companies to analyze
            
        Returns:
            Dictionary mapping ticker symbols to their data
        """
        tickers = self.ticker_provider.get_sp500_tickers(count)
        return self.get_multiple_tickers_data(tickers)
    
    def get_sector_analysis(self, tickers: List[str]) -> Dict[str, Any]:
        """Perform sector-wide analysis across multiple tickers.
        
        Args:
            tickers: List of ticker symbols to analyze
            
        Returns:
            Analysis results including trends and confidence metrics
        """
        logger.info(f"Performing sector analysis for {len(tickers)} tickers")
        
        try:
            ticker_data = self.get_multiple_tickers_data(tickers)
            
            # Extract metrics for analysis
            revenue_growths = []
            pe_ratios = []
            roes = []
            strong_performers = []
            weak_performers = []
            
            for ticker, data in ticker_data.items():
                fundamentals = data["fundamentals"]
                
                # Collect metrics
                revenue_growth = fundamentals.get("revenue_growth", 0)
                pe_ratio = fundamentals.get("pe_ratio", 0)
                roe = fundamentals.get("roe", 0)
                
                revenue_growths.append(revenue_growth)
                pe_ratios.append(pe_ratio)
                roes.append(roe)
                
                # Classification logic
                if revenue_growth > 0.1 and roe > 0.15:
                    strong_performers.append(ticker)
                elif revenue_growth < -0.05 or roe < 0.05:
                    weak_performers.append(ticker)
            
            # Calculate sector metrics
            avg_revenue_growth = sum(revenue_growths) / len(revenue_growths) if revenue_growths else 0
            avg_pe = sum(pe_ratios) / len(pe_ratios) if pe_ratios else 0
            avg_roe = sum(roes) / len(roes) if roes else 0
            
            # Determine confidence level
            confidence = FinancialDataValidator._calculate_sector_confidence(
                avg_revenue_growth, avg_roe, len(ticker_data), len(strong_performers)
            )
            
            # Determine overall trend
            trend = FinancialDataValidator._determine_sector_trend(avg_revenue_growth, avg_roe)
            
            sector_analysis = {
                "overall_trend": trend,
                "confidence": confidence,
                "strong_performers": strong_performers,
                "weak_performers": weak_performers,
                "key_metrics": {
                    "avg_revenue_growth": round(avg_revenue_growth, 3),
                    "avg_pe_ratio": round(avg_pe, 2),
                    "avg_roe": round(avg_roe, 3)
                },
                "sample_size": len(ticker_data),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Sector analysis completed: {trend} trend, {confidence:.2f} confidence")
            return sector_analysis
            
        except Exception as e:
            logger.error(f"Sector analysis failed: {e}")
            raise RuntimeError(f"Sector analysis failed: {str(e)}")

    @property
    def available_tickers(self) -> List[str]:
        """Get list of available S&P 500 tickers."""
        return self.ticker_provider.get_sp500_tickers()


class FinancialDataValidator:
    """Extended validator for financial analysis."""
    
    @staticmethod
    def _calculate_sector_confidence(avg_revenue_growth: float, avg_roe: float, 
                                   sample_size: int, strong_count: int) -> float:
        """Calculate confidence score for sector analysis."""
        
        # Base confidence from sample size
        base_confidence = min(0.8, 0.3 + (sample_size / 10) * 0.5)
        
        # Adjust based on metrics consistency
        metrics_score = 0
        if avg_revenue_growth > 0.05:
            metrics_score += 0.2
        if avg_roe > 0.12:
            metrics_score += 0.2
        if strong_count >= sample_size * 0.4:
            metrics_score += 0.3
        
        return min(0.95, base_confidence + metrics_score)
    
    @staticmethod
    def _determine_sector_trend(avg_revenue_growth: float, avg_roe: float) -> str:
        """Determine overall sector trend."""
        if avg_revenue_growth > 0.05 and avg_roe > 0.12:
            return "positive"
        elif avg_revenue_growth < -0.02 or avg_roe < 0.08:
            return "negative"
        else:
            return "neutral"


# Global instance for easy access (Dependency Injection principle)
_sp500_service: Optional[SP500DataService] = None

def get_sp500_data_service() -> SP500DataService:
    """Get or create global SP500DataService instance with default dependencies."""
    global _sp500_service
    if _sp500_service is None:
        # What: Capture optional third-party keys so we know which premium sources are available
        # Why: Prevents accidental outbound calls to providers without authentication
        # How: Ask the central API-key registry for the Alpha Vantage key (others can be added later)
        # Data: Returns {"alpha_vantage": "..."} when configured, otherwise empty dict
        api_keys = get_available_api_keys("alpha_vantage")
        if api_keys:
            logger.info("Alpha Vantage key detected for SP500DataService (refactored) optional features")
        
        # Create with dependency injection
        _sp500_service = SP500DataService(
            data_provider=YahooFinanceProvider(
                rate_limiter=APIRateLimiter(requests_per_minute=12)
            ),
            validator=FinancialDataValidator(),
            ticker_provider=TickerListProvider()
        )
        logger.info("Created global SP500DataService instance")
    
    return _sp500_service


if __name__ == "__main__":
    """Test the refactored service."""
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Test with dependency injection
        service = SP500DataService()
        
        # Test single ticker
        print("Testing single ticker data retrieval...")
        aapl_data = service.get_single_ticker_data("AAPL")
        print(f"✅ Single ticker test passed: {len(aapl_data['market_data'])} data points")
        
        # Test multiple tickers
        print("Testing multiple tickers data retrieval...")
        sample_data = service.get_sp500_sample(count=3)
        print(f"✅ Multiple tickers test passed: {len(sample_data)} companies")
        
        # Test sector analysis
        print("Testing sector analysis...")
        sector_analysis = service.get_sector_analysis(["AAPL", "MSFT", "GOOGL"])
        print(f"✅ Sector analysis test passed: {sector_analysis['overall_trend']} trend")
        
        print("\n🎉 All tests passed! Refactored service is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Service test failed: {e}", exc_info=True)
