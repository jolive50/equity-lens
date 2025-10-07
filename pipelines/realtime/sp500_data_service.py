"""Refactored S&B 500 data service following SOLID principles.

This service orchestrates data retrieval and analysis for multiple S&P 500 companies,
using dependency injection and interface segregation for maintainable, testable code.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Protocol, Union
from datetime import datetime

try:
    from .data_sources import (
        YahooFinanceProvider, 
        APIRateLimiter, 
        DataValidationService,
        FinancialDataProvider
    )
except ImportError:
    from data_sources import (
        YahooFinanceProvider, 
        APIRateLimiter, 
        DataValidationService,
        FinancialDataProvider
    )

from .api_keys import get_available_api_keys

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DataQualitySummary:
    """Structured view of the validation metrics for a ticker."""

    historical_data_points: int
    fundamental_metrics_count: int
    news_articles_count: int
    validation_timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert the summary into the dict shape expected by downstream code."""
        return {
            "historical_data_points": self.historical_data_points,
            "fundamental_metrics_count": self.fundamental_metrics_count,
            "news_articles_count": self.news_articles_count,
            "validation_timestamp": self.validation_timestamp,
        }


@dataclass(frozen=True, slots=True)
class ProcessedTickerData:
    """Normalized payload for a single ticker."""

    ticker: str
    market_data: List[Dict[str, Any]]
    fundamentals: Dict[str, float]
    news_data: List[Dict[str, Any]]
    data_quality: DataQualitySummary

    def to_dict(self) -> Dict[str, Any]:
        """Return the dict representation expected by the public API."""
        return {
            "ticker": self.ticker,
            "market_data": self.market_data,
            "fundamentals": self.fundamentals,
            "news_data": self.news_data,
            "data_quality": self.data_quality.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SectorAnalysisSummary:
    """Structured sector insights derived from multiple tickers."""

    overall_trend: str
    confidence: float
    strong_performers: List[str]
    weak_performers: List[str]
    key_metrics: Dict[str, float]
    sample_size: int
    analysis_timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_trend": self.overall_trend,
            "confidence": self.confidence,
            "strong_performers": self.strong_performers,
            "weak_performers": self.weak_performers,
            "key_metrics": self.key_metrics,
            "sample_size": self.sample_size,
            "analysis_timestamp": self.analysis_timestamp,
        }


class TickerDataProcessor(Protocol):
    """Protocol for objects able to validate and standardize ticker data."""

    def process_ticker_data(
        self,
        ticker: str,
        historical_data: List[Dict[str, Any]],
        fundamentals: Dict[str, float],
        news_data: List[Dict[str, Any]],
    ) -> Union[ProcessedTickerData, Dict[str, Any]]:
        ...


class SP500ServiceError(RuntimeError):
    """Raised when the S&P 500 data service cannot fulfill a request."""


def _serialise_ticker_payload(
    payload: Union[ProcessedTickerData, Dict[str, Any]]
) -> Dict[str, Any]:
    """Convert validator output into the public dict representation."""

    if isinstance(payload, ProcessedTickerData):
        return payload.to_dict()
    return payload


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


class DataProcessingValidator(TickerDataProcessor):
    """Validates and processes financial data quality."""
    
    @staticmethod
    def process_ticker_data(
        ticker: str,
        historical_data: List[Dict[str, Any]],
        fundamentals: Dict[str, float],
        news_data: List[Dict[str, Any]],
    ) -> ProcessedTickerData:
        """Process and validate data for a single ticker."""
        
        # Validate each data component
        validated_historical = DataValidationService.validate_historical_data(historical_data)
        validated_fundamentals = DataValidationService.validate_fundamentals(fundamentals)

        quality = DataQualitySummary(
            historical_data_points=len(validated_historical),
            fundamental_metrics_count=len(validated_fundamentals),
            news_articles_count=len(news_data),
            validation_timestamp=datetime.utcnow().isoformat(),
        )

        return ProcessedTickerData(
            ticker=ticker.upper(),
            market_data=validated_historical,
            fundamentals=validated_fundamentals,
            news_data=news_data,  # News payload already structured
            data_quality=quality,
        )


class SP500DataService:
    """Main service for S&P 500 data retrieval (Open/Closed Principle).
    
    This service can be extended with new data providers without modifying
    its core functionality.
    """
    
    def __init__(self, 
                 data_provider: Optional[FinancialDataProvider] = None,
                 validator: Optional[TickerDataProcessor] = None,
                 ticker_provider: Optional[TickerListProvider] = None,
                 request_interval: float = 0.5):
        """Initialize with dependency injection.
        
        Args:
            data_provider: Financial data provider implementation
            validator: Data validator instance
            ticker_provider: Ticker list provider instance
            request_interval: Friendly delay between provider requests (seconds)
        """
        self.data_provider = data_provider or YahooFinanceProvider(
            rate_limiter=APIRateLimiter(requests_per_minute=12)
        )
        self.validator: TickerDataProcessor = validator or DataProcessingValidator()
        self.ticker_provider = ticker_provider or TickerListProvider()
        self._request_interval = max(0.0, request_interval)
        
        logger.info(
            "SP500DataService initialized with dependencies (request interval %.2fs)",
            self._request_interval,
        )

    @property
    def request_interval(self) -> float:
        """Expose the polite delay applied between provider calls."""
        return self._request_interval

    def _throttle(self) -> None:
        """Respect upstream rate limits by pausing between calls when configured."""
        if self._request_interval > 0:
            time.sleep(self._request_interval)
    
    def get_single_ticker_data(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing historical data, fundamentals, and news
            
        Raises:
            ValueError: If ticker is invalid
            SP500ServiceError: If data retrieval fails
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
            processed_payload = self.validator.process_ticker_data(
                ticker, historical_data, fundamentals, news_data
            )

            result = _serialise_ticker_payload(processed_payload)

            quality = result.get("data_quality", {})
            logger.info(
                "Successfully retrieved data for %s (prices=%s fundamentals=%s news=%s)",
                ticker,
                quality.get("historical_data_points"),
                quality.get("fundamental_metrics_count"),
                quality.get("news_articles_count"),
            )
            return result

        except Exception as e:
            logger.error(f"Failed to retrieve data for {ticker}: {e}")
            raise SP500ServiceError(f"Data retrieval failed for {ticker}: {str(e)}") from e
    
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
            SP500ServiceError: If all data retrievals fail
        """
        # Validate and limit ticker list
        validated_tickers = self.ticker_provider.validate_ticker_list(tickers)[:max_tickers]
        
        if not validated_tickers:
            raise ValueError("No valid ticker symbols provided")
        
        logger.info(f"Processing {len(validated_tickers)} tickers: {validated_tickers}")
        
        results: Dict[str, Dict[str, Any]] = {}
        errors = []
        
        for i, ticker in enumerate(validated_tickers):
            try:
                logger.info(f"Processing {ticker} ({i+1}/{len(validated_tickers)})")
                ticker_data = self.get_single_ticker_data(ticker)
                results[ticker] = ticker_data
                
                # Add small delay between requests to be respectful
                self._throttle()
                
            except Exception as ticker_error:
                logger.error(f"Failed to process {ticker}: {ticker_error}")
                errors.append(f"{ticker}: {str(ticker_error)}")
                continue
        
        if not results:
            raise SP500ServiceError(
                f"Failed to retrieve data for any ticker. Errors: {errors}"
            )
        
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

        Raises:
            SP500ServiceError: If the sector analysis cannot be completed
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
            
            summary = FinancialDataValidator.build_sector_summary(
                avg_revenue_growth=avg_revenue_growth,
                avg_pe=avg_pe,
                avg_roe=avg_roe,
                sample_size=len(ticker_data),
                strong_performers=strong_performers,
                weak_performers=weak_performers,
            )
            
            logger.info(
                "Sector analysis completed: %s trend, %.2f confidence",
                summary.overall_trend,
                summary.confidence,
            )
            return summary.to_dict()
            
        except Exception as e:
            logger.error(f"Sector analysis failed: {e}")
            raise SP500ServiceError(f"Sector analysis failed: {str(e)}") from e

    @property
    def available_tickers(self) -> List[str]:
        """Get list of available S&P 500 tickers."""
        return self.ticker_provider.get_sp500_tickers()


class FinancialDataValidator:
    """Extended validator for financial analysis."""

    @classmethod
    def build_sector_summary(
        cls,
        *,
        avg_revenue_growth: float,
        avg_pe: float,
        avg_roe: float,
        sample_size: int,
        strong_performers: List[str],
        weak_performers: List[str],
    ) -> SectorAnalysisSummary:
        """Aggregate metrics into a structured sector summary."""

        confidence = cls._calculate_sector_confidence(
            avg_revenue_growth, avg_roe, sample_size, len(strong_performers)
        )
        trend = cls._determine_sector_trend(avg_revenue_growth, avg_roe)
        key_metrics = {
            "avg_revenue_growth": round(avg_revenue_growth, 3),
            "avg_pe_ratio": round(avg_pe, 2),
            "avg_roe": round(avg_roe, 3),
        }

        return SectorAnalysisSummary(
            overall_trend=trend,
            confidence=confidence,
            strong_performers=strong_performers,
            weak_performers=weak_performers,
            key_metrics=key_metrics,
            sample_size=sample_size,
            analysis_timestamp=datetime.utcnow().isoformat(),
        )
    
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
        # What: Capture any premium data-provider keys that are available for optional enrichment
        # Why: Keeps the service aware of extra integrations without forcing every environment to provide them
        # How: Reuse the central API key helper and log the result for observability
        # Data: Returns {"alpha_vantage": "..."} when the key exists; empty dict otherwise
        api_keys = get_available_api_keys("alpha_vantage")
        if api_keys:
            logger.info("Alpha Vantage key detected for SP500DataService optional features")
        
        # Create with dependency injection
        _sp500_service = SP500DataService(
            data_provider=YahooFinanceProvider(
                rate_limiter=APIRateLimiter(requests_per_minute=12)
            ),
            validator=DataProcessingValidator(),
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
        print(f"Single ticker test passed: {len(aapl_data['market_data'])} data points")
        
        # Test multiple tickers
        print("Testing multiple tickers data retrieval...")
        sample_data = service.get_sp500_sample(count=3)
        print(f"Multiple tickers test passed: {len(sample_data)} companies")
        
        # Test sector analysis
        print("Testing sector analysis...")
        sector_analysis = service.get_sector_analysis(["AAPL", "MSFT", "GOOGL"])
        print(f"Sector analysis test passed: {sector_analysis['overall_trend']} trend")
        
        print("\nAll tests passed! Refactored service is working correctly.")
        
    except Exception as e:
        print(f"Test failed: {e}")
        logger.error(f"Service test failed: {e}", exc_info=True)
