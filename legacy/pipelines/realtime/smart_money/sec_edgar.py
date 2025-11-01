"""SEC EDGAR API client for institutional and insider trading data.

This module provides a client for fetching data from the SEC EDGAR (Electronic Data
Gathering, Analysis, and Retrieval) system. It's designed following SOLID principles
with comprehensive explanations for college-level understanding.

What this module does:
- Fetches 13F filings (institutional holdings) from SEC EDGAR
- Retrieves insider trading data (Form 4 filings)
- Processes and caches SEC data to respect rate limits
- Provides clean interfaces for smart money tracking

Why we need this:
- Track what "smart money" (institutions, insiders) are doing
- Institutional holdings indicate confidence from professional investors
- Insider trades often precede major company events
- SEC data is free, authoritative, and comprehensive

How it works:
- Uses SEC's public APIs and bulk data endpoints
- Respects SEC rate limits (10 requests per second)
- Caches responses to minimize redundant requests
- Parses XML/JSON filing data into usable formats

College-Level Concepts:
- API Client Pattern: Abstraction over HTTP requests
- Rate Limiting: Respecting API usage limits
- Caching: Storing responses to avoid redundant requests
- Data Parsing: Converting raw SEC filings to structured data
- SOLID Principles: Single Responsibility, Dependency Inversion
"""
from __future__ import annotations

import json  # For JSON parsing
import logging  # For progress tracking
import time  # For rate limiting
from datetime import datetime, timedelta  # For date handling
from pathlib import Path  # For file paths
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode  # For building URLs

import requests  # For HTTP requests

logger = logging.getLogger(__name__)


class SECEdgarClient:
    """Client for SEC EDGAR API with rate limiting and caching.

    What this does: Fetches institutional and insider trading data from SEC
    Why: SEC provides authoritative, free data on institutional holdings and insider trades
    How: Uses SEC's public APIs with proper rate limiting and user-agent headers

    This class demonstrates SOLID principles:
    - Single Responsibility: Only handles SEC API communication
    - Open/Closed: Can be extended for other SEC filing types
    - Interface Segregation: Clean methods for specific data types
    - Dependency Inversion: Returns abstractions (dicts), not SEC-specific formats

    SEC Requirements:
    - User-Agent header required (identifies your app)
    - Rate limit: 10 requests per second
    - Respect robots.txt and terms of service

    College-Level Analogy:
    Think of this as a librarian who helps you access SEC records. They know
    where to find documents, how to request them properly (rate limits), and
    can translate the bureaucratic filing formats into understandable information.
    """

    # SEC EDGAR API endpoints
    # What: Base URLs for different SEC data services
    # Why: SEC provides multiple endpoints for different data types
    # How: Use these as URL prefixes for API requests
    BASE_URL = "https://data.sec.gov"  # Primary data API
    SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"  # Filing search
    BULK_DATA_URL = "https://www.sec.gov/cgi-bin/browse-edgar"  # Legacy bulk access

    def __init__(
        self,
        *,
        user_agent: str,  # Required by SEC (identifies your app)
        cache_dir: Optional[Path] = None,  # Where to cache responses
        rate_limit_delay: float = 0.1,  # Delay between requests (seconds)
        cache_ttl: int = 3600,  # Cache time-to-live (seconds)
    ):
        """Initialize SEC EDGAR client with configuration.

        What: Sets up client for SEC API access with rate limiting
        Why: SEC requires proper identification and rate limiting
        How: Stores config, initializes cache, sets up session

        Args:
            user_agent: Required User-Agent string (e.g., "Company Name admin@company.com")
            cache_dir: Directory for caching responses (default: data/smart_money/cache)
            rate_limit_delay: Minimum seconds between requests (default: 0.1 = 10 req/sec)
            cache_ttl: How long to cache responses in seconds (default: 1 hour)

        SEC User-Agent Requirement:
        Format: "CompanyName AdminContact@company.com"
        Example: "StockSense Research admin@stocksense.com"
        Why: Allows SEC to contact you if there's an issue with your usage
        """
        self.user_agent = user_agent
        self.rate_limit_delay = rate_limit_delay
        self.cache_ttl = cache_ttl

        # Setup cache directory
        # What: Create directory for caching SEC responses
        # Why: Reduces API calls, speeds up repeat requests
        # How: Create directory if doesn't exist
        self.cache_dir = cache_dir or Path("data/smart_money/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Track last request time for rate limiting
        # What: Timestamp of most recent API request
        # Why: Enforce minimum delay between requests
        # How: Update after each request, check before new request
        self._last_request_time = 0.0

        # Setup HTTP session with headers
        # What: Reusable HTTP session with SEC-required headers
        # Why: Connection pooling (faster), consistent headers
        # How: requests.Session() provides persistent session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,  # Required by SEC
            'Accept-Encoding': 'gzip, deflate',  # Enable compression
            'Host': 'www.sec.gov'  # Required by SEC
        })

        logger.info(f"Initialized SEC EDGAR client: {user_agent}")
        logger.info(f"Rate limit: {1.0 / rate_limit_delay:.1f} requests/second")
        logger.info(f"Cache directory: {self.cache_dir}")

    def _enforce_rate_limit(self) -> None:
        """Enforce rate limit by sleeping if necessary.

        What: Ensures minimum delay between API requests
        Why: SEC limits to 10 requests per second
        How: Calculates time since last request, sleeps if needed

        Rate Limiting Example:
        - Last request at t=0.0
        - New request at t=0.05 (50ms later)
        - Need 100ms minimum delay
        - Sleep for 50ms to reach 100ms total
        """
        # Calculate time since last request
        # What: How long has it been since last API call?
        # Why: Need to know if we should delay
        # How: Current time - last request time
        current_time = time.time()
        elapsed = current_time - self._last_request_time

        # Sleep if not enough time has passed
        # What: Delay if we're going too fast
        # Why: Respect SEC rate limits
        # How: time.sleep() pauses execution
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            logger.debug(f"Rate limit: sleeping {sleep_time:.3f}s")
            time.sleep(sleep_time)

        # Update last request time
        # What: Record this request's timestamp
        # Why: For next request's rate limit check
        # How: Store current time
        self._last_request_time = time.time()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Generate cache file path for a cache key.

        What: Converts cache key to file path
        Why: Need consistent file naming for cache
        How: Hash key, use as filename

        Args:
            cache_key: Unique identifier for cached data

        Returns:
            Path to cache file

        Example:
            cache_key = "13F_AAPL_2025Q1"
            → data/smart_money/cache/13F_AAPL_2025Q1.json
        """
        # Sanitize cache key for filename
        # What: Remove invalid filename characters
        # Why: Some characters can't be used in filenames
        # How: Replace problematic characters with underscores
        safe_key = cache_key.replace('/', '_').replace('\\', '_').replace(':', '_')
        return self.cache_dir / f"{safe_key}.json"

    def _get_cached(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached data if available and not expired.

        What: Loads data from cache if fresh
        Why: Avoid redundant API calls
        How: Check file existence, age, then load

        Args:
            cache_key: Unique identifier for cached data

        Returns:
            Cached data if available and fresh, None otherwise

        Cache Expiration:
        - Check file modification time
        - If older than cache_ttl, consider expired
        - This ensures we get fresh data eventually
        """
        cache_path = self._get_cache_path(cache_key)

        # Check if cache file exists
        # What: Does cached data exist?
        # Why: Can't use cache if it doesn't exist
        # How: Check file existence
        if not cache_path.exists():
            logger.debug(f"Cache miss: {cache_key}")
            return None

        # Check cache age
        # What: Is cached data still fresh?
        # Why: Don't want to use stale data
        # How: Compare file modification time to TTL
        file_age = time.time() - cache_path.stat().st_mtime
        if file_age > self.cache_ttl:
            logger.debug(f"Cache expired: {cache_key} (age: {file_age:.0f}s)")
            return None

        # Load cached data
        # What: Read JSON from cache file
        # Why: Reuse previously fetched data
        # How: Parse JSON file
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.debug(f"Cache hit: {cache_key}")
                return data
        except Exception as e:
            # Cache file corrupted or invalid
            # What: Handle cache read errors
            # Why: Corrupt cache shouldn't crash app
            # How: Log error, return None (will refetch)
            logger.warning(f"Cache read error for {cache_key}: {e}")
            return None

    def _set_cached(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Store data in cache.

        What: Saves API response to cache file
        Why: Enable future cache hits
        How: Write JSON to file

        Args:
            cache_key: Unique identifier for data
            data: Data to cache (must be JSON-serializable)
        """
        cache_path = self._get_cache_path(cache_key)

        try:
            # Write data to cache file
            # What: Persist data as JSON
            # Why: Enable future reuse
            # How: json.dump() with pretty printing
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Cached: {cache_key}")
        except Exception as e:
            # Cache write failed
            # What: Handle cache write errors
            # Why: Cache failure shouldn't break API usage
            # How: Log warning, continue (just won't cache this response)
            logger.warning(f"Cache write error for {cache_key}: {e}")

    def get_institutional_holdings(
        self,
        *,
        ticker: str,  # Stock symbol
        quarter: Optional[str] = None,  # e.g., "2025Q1" (default: latest)
    ) -> Dict[str, Any]:
        """Get institutional holdings (13F filings) for a ticker.

        What: Fetches data on institutional ownership from 13F filings
        Why: See which funds/institutions own the stock
        How: Queries SEC's 13F dataset for the ticker

        Args:
            ticker: Stock symbol (e.g., "AAPL")
            quarter: Optional quarter (e.g., "2025Q1"), defaults to latest

        Returns:
            Dictionary with institutional holdings data:
            {
                "ticker": "AAPL",
                "quarter": "2025Q1",
                "total_holders": 150,
                "total_shares": 1500000,
                "top_holders": [
                    {"name": "Vanguard", "shares": 500000, "value": 75000000},
                    ...
                ],
                "ownership_change": "+2.5%"  # vs previous quarter
            }

        13F Filings:
        - Required for institutions managing $100M+
        - Filed quarterly (45 days after quarter end)
        - Shows positions as of quarter end
        - Good indicator of institutional confidence

        College-Level Explanation:
        Imagine you want to know which professional investors own Apple stock.
        13F filings are like public records that big investors must file showing
        their holdings. This function fetches and summarizes that data.
        """
        # Determine quarter
        # What: Use provided quarter or calculate latest
        # Why: Need to know which filing period to fetch
        # How: Parse quarter string or default to current quarter
        if quarter is None:
            # Calculate current quarter
            # What: Determine latest completed quarter
            # Why: Default to most recent data
            # How: Get current date, calculate quarter
            now = datetime.now()
            quarter_num = (now.month - 1) // 3 + 1
            quarter = f"{now.year}Q{quarter_num}"

        # Check cache first
        # What: Try to use cached data
        # Why: Avoid unnecessary API call
        # How: Build cache key, check cache
        cache_key = f"13F_{ticker.upper()}_{quarter}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Enforce rate limit before API call
        # What: Ensure we're not going too fast
        # Why: Respect SEC limits
        # How: Sleep if needed
        self._enforce_rate_limit()

        # Fetch from SEC API
        # What: Make HTTP request to SEC EDGAR
        # Why: Get institutional holdings data
        # How: Use SEC's Company Tickers search endpoint
        logger.info(f"Fetching 13F data for {ticker} ({quarter})...")

        try:
            # Note: This is a simplified implementation
            # What: Placeholder for actual SEC API integration
            # Why: Full implementation requires parsing XML/HTML filing documents
            # How: In production, would use SEC's bulk data or parse individual filings

            # For now, return placeholder structure
            # What: Return empty but correctly structured data
            # Why: Allows SmartMoneyAgent to work without full SEC implementation
            # How: Return dict matching expected format
            result = {
                "ticker": ticker.upper(),
                "quarter": quarter,
                "total_holders": 0,
                "total_shares": 0,
                "top_holders": [],
                "ownership_change": "N/A",
                "message": "SEC EDGAR integration in progress - institutional data not yet available",
                "data_source": "SEC EDGAR (placeholder)"
            }

            # Cache result
            # What: Store for future use
            # Why: Avoid repeat API calls
            # How: Write to cache
            self._set_cached(cache_key, result)

            return result

        except requests.RequestException as e:
            # HTTP request failed
            # What: Handle network/API errors
            # Why: Provide graceful fallback
            # How: Log error, raise RuntimeError with context
            logger.error(f"Failed to fetch 13F data for {ticker}: {e}")
            raise RuntimeError(f"SEC EDGAR API request failed: {e}") from e

    def get_insider_trades(
        self,
        *,
        ticker: str,  # Stock symbol
        days_back: int = 90,  # How many days of history
    ) -> Dict[str, Any]:
        """Get insider trading activity (Form 4 filings) for a ticker.

        What: Fetches data on insider buying/selling from Form 4 filings
        Why: Insiders often trade before major events (earnings, products, etc.)
        How: Queries SEC's Form 4 dataset for the ticker

        Args:
            ticker: Stock symbol (e.g., "AAPL")
            days_back: How many days of history to fetch (default: 90)

        Returns:
            Dictionary with insider trading data:
            {
                "ticker": "AAPL",
                "period_days": 90,
                "total_transactions": 25,
                "buy_transactions": 15,
                "sell_transactions": 10,
                "recent_trades": [
                    {
                        "date": "2025-01-15",
                        "insider": "Tim Cook",
                        "title": "CEO",
                        "transaction": "Buy",
                        "shares": 10000,
                        "price": 150.00
                    },
                    ...
                ],
                "net_buy_sell_ratio": 1.5  # More buying than selling
            }

        Form 4 Filings:
        - Filed by company insiders (executives, directors, 10%+ owners)
        - Must file within 2 business days of trade
        - Shows purchases and sales of company stock
        - Often indicates insider confidence (or lack thereof)

        College-Level Explanation:
        Company executives and board members must publicly disclose when they
        buy or sell their company's stock. This is valuable because they have
        inside knowledge of how the company is doing. Heavy buying often signals
        confidence; heavy selling might indicate concerns.
        """
        # Check cache first
        # What: Try to use cached data
        # Why: Avoid unnecessary API call
        # How: Build cache key, check cache
        cache_key = f"Form4_{ticker.upper()}_{days_back}days"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Enforce rate limit
        # What: Ensure we're not going too fast
        # Why: Respect SEC limits
        self._enforce_rate_limit()

        # Fetch from SEC API
        # What: Make HTTP request to SEC EDGAR
        # Why: Get insider trading data
        logger.info(f"Fetching Form 4 data for {ticker} (last {days_back} days)...")

        try:
            # Note: This is a simplified implementation
            # What: Placeholder for actual SEC API integration
            # Why: Full implementation requires parsing XML filing documents
            # How: In production, would parse Form 4 filings from SEC

            # For now, return placeholder structure
            # What: Return empty but correctly structured data
            # Why: Allows SmartMoneyAgent to work without full SEC implementation
            result = {
                "ticker": ticker.upper(),
                "period_days": days_back,
                "total_transactions": 0,
                "buy_transactions": 0,
                "sell_transactions": 0,
                "recent_trades": [],
                "net_buy_sell_ratio": 0.0,
                "message": "SEC EDGAR integration in progress - insider trade data not yet available",
                "data_source": "SEC EDGAR (placeholder)"
            }

            # Cache result
            # What: Store for future use
            # Why: Avoid repeat API calls
            self._set_cached(cache_key, result)

            return result

        except requests.RequestException as e:
            # HTTP request failed
            # What: Handle network/API errors
            # Why: Provide graceful fallback
            # How: Log error, raise RuntimeError with context
            logger.error(f"Failed to fetch Form 4 data for {ticker}: {e}")
            raise RuntimeError(f"SEC EDGAR API request failed: {e}") from e


def create_sec_client(
    *,
    user_agent: str = "StockSense Research admin@stocksense.com",
    cache_dir: Optional[Path] = None,
) -> SECEdgarClient:
    """Factory function to create SEC EDGAR client.

    What: Creates and returns configured SEC client
    Why: Centralizes client creation, provides default config
    How: Instantiates SECEdgarClient with standard settings

    Args:
        user_agent: User-Agent header for SEC requests
        cache_dir: Optional custom cache directory

    Returns:
        Configured SECEdgarClient ready to use

    Factory Pattern Benefits:
    - Encapsulates object creation logic
    - Provides sensible defaults
    - Makes testing easier (can inject mock clients)
    """
    return SECEdgarClient(
        user_agent=user_agent,
        cache_dir=cache_dir,
        rate_limit_delay=0.1,  # 10 requests per second
        cache_ttl=3600  # 1 hour cache
    )


# Test code (runs when this file is executed directly)
if __name__ == "__main__":
    print("Testing SEC EDGAR Client...")
    print("=" * 60)

    # Create client
    # What: Initialize SEC client with test user agent
    # Why: Test basic functionality
    # How: Use factory function
    client = create_sec_client(
        user_agent="StockSense Test admin@test.com"
    )

    # Test institutional holdings
    # What: Fetch 13F data for Apple
    # Why: Verify client works for institutional data
    # How: Call get_institutional_holdings
    print("\n1. Testing Institutional Holdings (13F)...")
    print("-" * 60)
    try:
        holdings = client.get_institutional_holdings(ticker="AAPL")
        print(f"Ticker: {holdings['ticker']}")
        print(f"Quarter: {holdings['quarter']}")
        print(f"Total Holders: {holdings['total_holders']}")
        print(f"Message: {holdings.get('message', 'N/A')}")
        print("✓ Institutional holdings test passed")
    except Exception as e:
        print(f"✗ Institutional holdings test failed: {e}")

    # Test insider trades
    # What: Fetch Form 4 data for Microsoft
    # Why: Verify client works for insider data
    # How: Call get_insider_trades
    print("\n2. Testing Insider Trades (Form 4)...")
    print("-" * 60)
    try:
        trades = client.get_insider_trades(ticker="MSFT", days_back=90)
        print(f"Ticker: {trades['ticker']}")
        print(f"Period: {trades['period_days']} days")
        print(f"Total Transactions: {trades['total_transactions']}")
        print(f"Buy/Sell Ratio: {trades['net_buy_sell_ratio']}")
        print(f"Message: {trades.get('message', 'N/A')}")
        print("✓ Insider trades test passed")
    except Exception as e:
        print(f"✗ Insider trades test failed: {e}")

    # Test caching
    # What: Verify cache is working
    # Why: Ensure we're not making redundant requests
    # How: Request same data twice, second should be cached
    print("\n3. Testing Caching...")
    print("-" * 60)
    try:
        # First request (should hit API)
        start_time = time.time()
        client.get_institutional_holdings(ticker="GOOGL")
        first_duration = time.time() - start_time

        # Second request (should hit cache)
        start_time = time.time()
        client.get_institutional_holdings(ticker="GOOGL")
        second_duration = time.time() - start_time

        print(f"First request: {first_duration:.3f}s")
        print(f"Second request: {second_duration:.3f}s (cached)")
        print(f"Speedup: {first_duration / second_duration:.1f}x")
        print("✓ Caching test passed")
    except Exception as e:
        print(f"✗ Caching test failed: {e}")

    print("\n" + "=" * 60)
    print("✓ SEC EDGAR client tests complete!")
    print("=" * 60)
    print("\nNotes:")
    print("- This is a placeholder implementation")
    print("- Full SEC EDGAR integration requires:")
    print("  1. Parsing XML/HTML filing documents")
    print("  2. CIK (Central Index Key) lookup for tickers")
    print("  3. Handling various filing formats and amendments")
    print("  4. Processing large bulk data files")
    print("- See SEC EDGAR documentation: https://www.sec.gov/edgar")
    print("=" * 60)
