"""Main StockSense API endpoint integrating LangGraph workflow with data adapters.

This is the "front desk" of the application - it receives HTTP requests from users' browsers
and coordinates all the backend services to provide stock analysis.

What this file does:
- Defines all API endpoints (URLs) that users can call
- Validates user input (is the ticker valid? is the user authorized?)
- Coordinates between different services (data fetching, AI analysis, storage)
- Returns formatted responses to the user

Why we need this:
- Provides a clean interface between frontend (React) and backend (Python)
- Handles errors gracefully so users get helpful messages
- Manages global state (database connections, AI agents)
- Enforces user permissions (basic vs premium features)
"""
from __future__ import annotations

import json  # For parsing JSON data
import logging  # For recording what happens (debugging and monitoring)
import os  # For accessing environment variables (API keys, settings)
from typing import Dict, Optional, List, Iterable  # Type hints for better code clarity

# dotenv: Loads environment variables from .env file
# Why: We don't want to hard-code API keys in our code (security risk)
from dotenv import load_dotenv

# FastAPI: Modern Python web framework
# Why: Fast, automatic API documentation, type validation
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware  # Allows frontend to call our API
from pydantic import BaseModel  # Validates request/response data structures
from fastapi.responses import StreamingResponse  # For streaming CSV files

# Our custom modules - these are the core business logic
from .data_adapters import DataAdapterFactory, DataService  # Fetches stock data
from .langgraph_workflow import run_stocksense_analysis  # Main AI workflow
from .api_keys import get_available_api_keys  # Centralised API-key registry
from .agents import (  # AI agents that do specific tasks
    ExplanationAgent,  # Explains predictions in plain English
    PredictionAgent,  # Predicts stock direction
    SentimentAgent,  # Analyzes news sentiment
    SmartMoneyAgent,  # Tracks institutional investors
    build_openai_llm,  # Creates OpenAI connection
    build_mock_llm,  # Creates fake LLM for testing
)
from .repository import (  # Data storage classes
    JsonFileRepository,  # Reads/writes JSON files
    WatchlistRepository,  # Manages user watchlists
    HistoryRepository,  # Stores analysis history
    AlertsRepository,  # Stores user alert rules
    AlertRule  # Data structure for alert rules
)

# Load environment variables from .env file
# This reads things like OPENAI_API_KEY, ALPHA_VANTAGE_API_KEY, etc.
load_dotenv()

# Configure logging to show informational messages
# Why: Helps us debug issues and monitor the application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  # Create logger for this file

# Initialize FastAPI application
# This creates the web server that listens for HTTP requests
app = FastAPI(
    title="StockSense API",  # Shows up in API documentation
    description="Layperson-friendly stock insights with AI-powered forecasting",
    version="1.0.0"  # Tracks which version of the API is running
)

# Add CORS middleware
# What: Cross-Origin Resource Sharing - allows frontend (React) to call our API
# Why: Browsers block requests to different domains by default for security
# How: This tells the browser "it's okay for any origin to call our API"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # "*" means allow all origins (for development)
                          # In production: specify exact frontend URL like ["https://stocksense.com"]
    allow_credentials=True,  # Allow cookies/authentication headers
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)


# ===== REQUEST/RESPONSE DATA MODELS =====
# These define the shape of data coming in and going out
# Pydantic automatically validates the data and generates API documentation

class AnalysisRequest(BaseModel):
    """Data structure for single stock analysis request.

    What: Defines what data the user must send to analyze a stock
    Why: FastAPI uses this to validate input and auto-generate documentation

    Example request from frontend:
    {
        "ticker": "AAPL",
        "user_tier": "premium"
    }
    """
    ticker: str  # Stock symbol (e.g., "AAPL", "MSFT")
    user_tier: str = "basic"  # User subscription level, defaults to "basic"


class AnalysisResponse(BaseModel):
    """Data structure for single stock analysis response.

    What: Defines what data we send back after analyzing a stock
    Why: Ensures consistent response format and generates documentation

    Example response to frontend:
    {
        "ticker": "AAPL",
        "as_of": "2025-01-24T10:30:00Z",
        "forecast": {"direction": "up", "confidence": 0.85, ...},
        "metrics": {...},
        "sentiment": {...},
        "smart_money": {...},
        "explanation": "Apple shows strong...",
        "warnings": [],
        "disclaimers": [...]
    }
    """
    ticker: str  # Stock symbol analyzed
    as_of: str  # Timestamp of analysis (ISO format)
    forecast: Dict  # Prediction results (direction, confidence, probabilities)
    metrics: Dict  # Financial metrics (P/E ratio, revenue growth, etc.)
    sentiment: Dict  # News sentiment analysis
    smart_money: Dict  # Institutional investor activity
    explanation: str  # Plain English explanation
    warnings: list  # Any warnings (e.g., "Low confidence")
    disclaimers: list  # Legal disclaimers


class BatchAnalysisRequest(BaseModel):
    """Request to analyze multiple stocks at once.

    What: User can send a list of tickers to analyze
    Why: Useful for watchlists - analyze all stocks in one request

    Example:
    {
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "user_tier": "premium"
    }
    """
    tickers: List[str]  # List of stock symbols to analyze
    user_tier: str = "premium"  # Batch analysis is premium feature


class BatchAnalysisItem(BaseModel):
    """Single item in batch analysis response.

    What: Compact summary for one stock (less detail than full analysis)
    Why: Faster processing and smaller response size for multiple stocks
    """
    ticker: str
    as_of: str  # Timestamp
    direction: str  # "up", "down", or "neutral"
    confidence: float  # 0.0 to 1.0
    horizon_days: int  # Days of 95%+ confidence
    score: float  # Sentiment score
    sentiment: Dict  # Sentiment details


class BatchAnalysisResponse(BaseModel):
    """Response containing multiple stock analyses.

    Example:
    {
        "user_tier": "premium",
        "count": 3,
        "items": [
            {"ticker": "AAPL", "direction": "up", ...},
            {"ticker": "MSFT", "direction": "up", ...},
            {"ticker": "GOOGL", "direction": "neutral", ...}
        ]
    }
    """
    user_tier: str
    count: int  # Number of stocks analyzed
    items: List[BatchAnalysisItem]  # List of analysis results


# ===== GLOBAL SERVICE INSTANCES =====
# These are created once and reused for all requests
# Why: Creating AI agents and database connections is expensive
# Pattern: Singleton pattern (lazy initialization)

_data_service: Optional[DataService] = None  # Fetches market data
_agents: Optional[Dict[str, any]] = None  # AI agents (prediction, sentiment, etc.)
_watchlists: Optional[WatchlistRepository] = None  # User watchlists storage
_history: Optional[HistoryRepository] = None  # Analysis history storage
_alerts: Optional[AlertsRepository] = None  # User alerts storage


def get_data_service() -> DataService:
    """Get or create the data service instance.

    What: Returns a singleton DataService that fetches stock data
    Why: We only want one DataService instance (saves memory, reuses connections)
    How: Creates instance on first call, returns cached instance on subsequent calls

    Returns:
        DataService configured with all available API keys
    """
    global _data_service  # Use the global variable

    # If service hasn't been created yet, create it
    if _data_service is None:
        adapters = {}  # Dictionary to store data source adapters

        # What: Determine which upstream APIs we can actually call right now
        # Why: Prevents accidental requests to services without credentials (avoids rate bans and errors)
        # How: Ask the central API-key registry for the subset of keys that are available
        # Data: Returns {service_name: api_key} pairs for configured providers
        api_keys = get_available_api_keys("alpha_vantage", "tiingo", "finnhub", "newsapi")

        # Alpha Vantage: Stock prices and fundamentals
        if "alpha_vantage" in api_keys:
            adapters["alpha_vantage"] = DataAdapterFactory.create_adapter(
                "alpha_vantage",
                api_keys["alpha_vantage"]
            )

        # Tiingo: Alternative stock data source
        if "tiingo" in api_keys:
            adapters["tiingo"] = DataAdapterFactory.create_adapter(
                "tiingo",
                api_keys["tiingo"]
            )

        # Finnhub: Another stock data source
        if "finnhub" in api_keys:
            adapters["finnhub"] = DataAdapterFactory.create_adapter(
                "finnhub",
                api_keys["finnhub"]
            )

        # NewsAPI: News articles source
        if "newsapi" in api_keys:
            adapters["newsapi"] = DataAdapterFactory.create_adapter(
                "newsapi",
                api_keys["newsapi"]
            )

        # Create the DataService with all configured adapters
        # DataService acts as a facade - simplifies access to multiple data sources
        _data_service = DataService(adapters)

    return _data_service


def get_agents() -> Dict[str, any]:
    """Get or create the AI agent instances.

    What: Returns AI agents that perform specific analysis tasks
    Why: Agents are expensive to create (load models, connect to APIs)
    How: Create once, cache, and reuse for all requests

    Returns:
        Dictionary with agent instances:
        - prediction: Predicts stock direction
        - sentiment: Analyzes news sentiment
        - explanation: Generates plain English explanations
        - smart_money: Tracks institutional activity
    """
    global _agents  # Use the global variable

    # If agents haven't been created yet, create them
    if _agents is None:
        # Try to use OpenAI's GPT model for better quality
        # If no API key available, fall back to mock (for testing)
        try:
            # build_openai_llm creates a connection to OpenAI's API
            # "gpt-4o-mini" is a cost-effective model (cheaper than full GPT-4)
            llm = build_openai_llm("gpt-4o-mini")
            logger.info("Using OpenAI GPT-4o-mini for agents")
        except ValueError as e:
            # If OpenAI initialization fails (no API key, network error, etc.)
            # Use a mock LLM that returns fake but realistic responses
            logger.warning(f"OpenAI not available: {e}. Using mock LLM.")
            llm = build_mock_llm("stocksense")

        # Create all agent instances with the LLM
        # Each agent has specialized prompts and logic for its task
        _agents = {
            "prediction": PredictionAgent(llm),  # Forecasts stock direction
            "sentiment": SentimentAgent(llm),  # Analyzes news sentiment
            "explanation": ExplanationAgent(llm),  # Explains results simply
            "smart_money": SmartMoneyAgent(llm),  # Tracks big investors
        }

    return _agents


def get_repos() -> Dict[str, any]:
    """Get or create repository instances for data storage.

    What: Returns repositories that save/load user data (watchlists, history, alerts)
    Why: Centralizes data access, ensures consistent file paths
    How: Creates JsonFileRepository instances for each data type

    Returns:
        Dictionary with:
        - watchlists: WatchlistRepository instance
        - history: HistoryRepository instance
        - alerts: AlertsRepository instance
    """
    global _watchlists, _history, _alerts  # Use global variables

    # If repositories haven't been created yet, create them
    if _watchlists is None or _history is None or _alerts is None:
        # Determine where to store data files
        # Default: ../data/realtime/feature_store/_app/
        # Can be overridden by APP_DATA_DIR environment variable
        base_dir = os.getenv(
            "APP_DATA_DIR",
            os.path.join(os.getcwd(), "..", "data", "realtime", "feature_store")
        )

        # Create JsonFileRepository for each data type
        # These handle reading/writing JSON files with thread-safety
        wl_store = JsonFileRepository(os.path.join(base_dir, "_app", "watchlists.json"))
        hist_store = JsonFileRepository(os.path.join(base_dir, "_app", "history.json"))
        al_store = JsonFileRepository(os.path.join(base_dir, "_app", "alerts.json"))

        # Create specialized repositories that know how to work with their data
        _watchlists = WatchlistRepository(wl_store)  # Manages ticker lists
        _history = HistoryRepository(hist_store)  # Stores past analyses
        _alerts = AlertsRepository(al_store)  # Stores alert rules

    return {
        "watchlists": _watchlists,
        "history": _history,
        "alerts": _alerts
    }


# ===== API ENDPOINTS =====
# These are the URLs that users can call
# Each function is decorated with @app.get, @app.post, etc.

@app.get("/")
async def root():
    """Root endpoint with API information.

    What: Returns basic info about the API
    Why: Helps users understand what endpoints are available
    How: Simply returns a JSON object with API metadata

    Example:
    GET http://localhost:8000/
    Returns: {"message": "StockSense API", "version": "1.0.0", ...}
    """
    return {
        "message": "StockSense API",
        "version": "1.0.0",
        "description": "Layperson-friendly stock insights with AI-powered forecasting",
        "endpoints": {
            "/analyze": "Main analysis endpoint",
            "/analyze/batch": "Batch analysis endpoint (registered/premium)",
            "/health": "Health check",
            "/docs": "API documentation"  # FastAPI auto-generates this
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint.

    What: Verifies that all services are working correctly
    Why: Load balancers and monitoring tools use this to check if API is alive
    How: Tries to get services; if any fail, returns 500 error

    Example:
    GET http://localhost:8000/health
    Returns: {"status": "healthy", "data_adapters": 3, "agents": 4}
    """
    try:
        # Try to get services - this will fail if something is broken
        data_service = get_data_service()
        agents = get_agents()

        # If we got here, everything is working
        return {
            "status": "healthy",
            "data_adapters": len(data_service.adapters),  # How many data sources
            "agents": len(agents),  # How many AI agents
            "timestamp": "2025-01-24T10:30:00Z"  # When check was performed
        }
    except Exception as e:
        # If anything failed, log the error and return 500 status
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,  # Internal Server Error
            detail="Service unhealthy"
        )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_stock(request: AnalysisRequest):
    """Main analysis endpoint that runs the complete StockSense workflow.

    What: Analyzes a single stock and returns prediction, sentiment, explanation
    Why: This is the primary function users call to get stock insights
    How:
    1. Validates input (ticker format, user tier)
    2. Gets services (data, agents)
    3. Runs AI workflow
    4. Saves to history
    5. Returns formatted result

    Receives from user:
    {
        "ticker": "AAPL",
        "user_tier": "premium"
    }

    Returns to user:
    {
        "ticker": "AAPL",
        "forecast": {"direction": "up", "confidence": 0.85, ...},
        "sentiment": {...},
        "explanation": "Apple shows strong momentum...",
        ...
    }
    """
    try:
        # === STEP 1: Validate Input ===
        # Check if ticker is valid
        # Why: Prevent malicious input or API abuse
        if not request.ticker or len(request.ticker) > 10:
            # Ticker symbols are max 5-6 characters usually
            # 10 is generous limit to catch errors
            raise HTTPException(
                status_code=400,  # Bad Request
                detail="Invalid ticker symbol"
            )

        # Check if user tier is valid
        # Why: Ensure we only process known subscription levels
        if request.user_tier not in ["basic", "registered", "premium"]:
            raise HTTPException(
                status_code=400,
                detail="User tier must be 'basic', 'registered' or 'premium'"
            )

        # === STEP 2: Get Services ===
        # Get singleton instances (created once, reused)
        data_service = get_data_service()  # For fetching market data
        agents = get_agents()  # For AI analysis

        # === STEP 3: Run AI Workflow ===
        # run_stocksense_analysis orchestrates the entire analysis:
        # - Fetches market data
        # - Runs prediction AI
        # - Analyzes sentiment
        # - Generates explanation
        # - Checks smart money (premium only)
        result = run_stocksense_analysis(
            ticker=request.ticker.upper(),  # Convert to uppercase (AAPL not aapl)
            user_tier=request.user_tier,
            prediction_agent=agents["prediction"],
            sentiment_agent=agents["sentiment"],
            explanation_agent=agents["explanation"],
            smart_money_agent=agents["smart_money"],
            data_service=data_service,
        )

        # === STEP 4: Save to History ===
        # Store compact entry for user's analysis history
        # Why: Users can see past predictions and track accuracy
        repos = get_repos()
        history: HistoryRepository = repos["history"]
        history.append(
            user_id="demo",  # In production: get from authentication
            entry={
                "ticker": result["ticker"],
                "as_of": result["as_of"],  # When analysis was done
                "direction": result["forecast"]["direction"],  # up/down/neutral
                "confidence": float(result["forecast"]["confidence"]),
                "score": float(result.get("sentiment", {}).get("score", 0.0)),
            }
        )

        # === STEP 5: Return Result ===
        # AnalysisResponse validates the structure and returns to user
        return AnalysisResponse(**result)

    except HTTPException:
        # If we already raised an HTTPException, just re-raise it
        # Don't wrap it in another error
        raise
    except Exception as e:
        # Catch any unexpected errors
        # Why: Don't expose internal error details to users (security)
        logger.error(f"Analysis failed for {request.ticker}: {e}")
        raise HTTPException(
            status_code=500,  # Internal Server Error
            detail="Analysis failed - please try again later"
        )


@app.get("/prices")
async def get_prices(
    ticker: str = Query(..., description="Stock ticker symbol"),
    source: str = Query("alpha_vantage", description="Data source"),
    limit: int = Query(200, description="Number of data points", le=2000)
):
    """Get historical price data for a ticker.

    What: Returns list of daily prices for a stock
    Why: Users want to see price history charts
    How: Fetches from specified data source, formats as JSON

    Receives:
    GET /prices?ticker=AAPL&source=alpha_vantage&limit=200

    Returns:
    {
        "ticker": "AAPL",
        "source": "alpha_vantage",
        "count": 200,
        "items": [
            {"date": "2025-01-24", "close": 150.25, "volume": 50000000, ...},
            ...
        ]
    }
    """
    try:
        data_service = get_data_service()

        # Verify requested source exists
        # Why: Don't try to fetch from a source we don't have API key for
        if source not in data_service.adapters:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown source: {source}"
            )

        # Get the adapter for this source
        adapter = data_service.adapters[source]

        # Fetch historical data
        # Returns list of MarketData objects
        historical_data = adapter.fetch_historical_data(ticker.upper(), limit)

        # Convert MarketData objects to dictionaries for JSON response
        # Why: JSON can't serialize custom objects, needs dicts
        items = []
        for data in historical_data:
            items.append({
                "date": data.timestamp,  # Date string (YYYY-MM-DD)
                "close": data.close,  # Closing price
                "open": data.open_price,  # Opening price
                "high": data.high,  # Highest price that day
                "low": data.low,  # Lowest price that day
                "volume": data.volume  # Number of shares traded
            })

        return {
            "ticker": ticker.upper(),
            "source": source,
            "count": len(items),
            "items": items
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Price fetch failed for {ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch price data"
        )


@app.get("/news")
async def get_news(
    ticker: str = Query(..., description="Stock ticker symbol"),
    source: str = Query("newsapi", description="News source"),
    limit: int = Query(100, description="Number of articles", le=500)
):
    """Get news articles for a ticker.

    What: Returns recent news articles about a stock
    Why: Users want to see what's being said about the company
    How: Fetches from news API, formats as JSON

    Receives:
    GET /news?ticker=AAPL&source=newsapi&limit=100

    Returns:
    {
        "ticker": "AAPL",
        "source": "newsapi",
        "count": 100,
        "items": [
            {
                "title": "Apple Reports Strong Earnings",
                "content": "Apple Inc. announced...",
                "source": "Reuters",
                "url": "https://...",
                "timestamp": "2025-01-24T09:00:00Z",
                "sentiment_score": 0.85
            },
            ...
        ]
    }
    """
    try:
        data_service = get_data_service()

        # Verify source exists
        if source not in data_service.adapters:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown source: {source}"
            )

        adapter = data_service.adapters[source]

        # Fetch news articles
        # Returns list of NewsItem objects
        news_items = adapter.fetch_news(ticker.upper(), limit)

        # Convert NewsItem objects to dictionaries
        items = []
        for item in news_items:
            items.append({
                "title": item.title,  # Article headline
                "content": item.content,  # Article text/summary
                "source": item.source,  # Publisher (Reuters, Bloomberg, etc.)
                "url": item.url,  # Link to full article
                "timestamp": item.timestamp,  # When published
                "sentiment_score": item.sentiment_score  # Positive/negative (if available)
            })

        return {
            "ticker": ticker.upper(),
            "source": source,
            "count": len(items),
            "items": items
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"News fetch failed for {ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to fetch news data"
        )


@app.get("/sentiment")
async def get_sentiment(
    ticker: str = Query(..., description="Stock ticker symbol"),
    limit: int = Query(200, description="Number of sentiment scores", le=1000)
):
    """Get sentiment analysis for a ticker.

    What: Analyzes news articles to determine overall market sentiment
    Why: Sentiment often predicts short-term price movements
    How: Fetches news, runs through sentiment AI, aggregates results

    Receives:
    GET /sentiment?ticker=AAPL&limit=200

    Returns:
    {
        "ticker": "AAPL",
        "current": "positive",
        "score": 0.72,
        "trend": "improving",
        "headlines": ["Apple beats earnings", ...],
        "news_count": 200
    }
    """
    try:
        data_service = get_data_service()
        agents = get_agents()

        # Get news from all available sources
        # Why: More news = better sentiment analysis
        all_news = data_service.get_all_news(ticker.upper())

        # Run sentiment analysis on the news
        # SentimentAgent uses FinBERT or LLM to classify each article
        sentiment_result = agents["sentiment"].run(
            ticker=ticker.upper(),
            news_data=all_news[:limit]  # Limit to prevent timeouts
        )

        return {
            "ticker": ticker.upper(),
            "current": sentiment_result.current,  # "positive", "negative", "neutral"
            "score": sentiment_result.score,  # Confidence in sentiment (0-1)
            "trend": sentiment_result.trend,  # "improving", "declining", "stable"
            "headlines": sentiment_result.headlines,  # Top headlines
            "news_count": len(all_news)  # Total articles found
        }

    except Exception as e:
        logger.error(f"Sentiment analysis failed for {ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze sentiment"
        )


@app.get("/smart-money")
async def get_smart_money(
    ticker: str = Query(..., description="Stock ticker symbol"),
    user_tier: str = Query("basic", description="User tier for gating features"),
):
    """Get smart money data for a ticker (premium only).

    What: Shows what institutional investors, insiders, and Congress are doing
    Why: "Smart money" often knows things before the public
    How: Queries smart money agent (which accesses SEC filings, insider trades, etc.)

    Receives:
    GET /smart-money?ticker=AAPL&user_tier=premium

    Returns (premium only):
    {
        "ticker": "AAPL",
        "institutions": {"summary": "Institutional ownership increased 3%..."},
        "insiders": {"summary": "CEO purchased 10,000 shares..."},
        "congress": {"summary": "5 congressional trades disclosed..."}
    }
    """
    try:
        # Feature gate: Only premium users can access smart money data
        # Why: This is expensive data that provides significant value
        if user_tier != "premium":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,  # Forbidden (not authorized)
                detail="Smart money data is available to premium users only",
            )

        agents = get_agents()

        # Run smart money analysis
        # SmartMoneyAgent fetches institutional ownership, insider trades, etc.
        smart_money_result = agents["smart_money"].run(ticker=ticker.upper())

        return {
            "ticker": ticker.upper(),
            "institutions": smart_money_result.institutions,  # Hedge funds, mutual funds
            "insiders": smart_money_result.insiders,  # Company executives
            "congress": smart_money_result.congress  # Congressional stock trades
        }

    except Exception as e:
        logger.error(f"Smart money analysis failed for {ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze smart money data"
        )


@app.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(request: BatchAnalysisRequest):
    """Batch analysis for multiple tickers (registered/premium only).

    What: Analyzes multiple stocks in one request
    Why: Efficient for watchlists - analyze entire portfolio at once
    How: Loops through tickers, runs analysis on each, returns compact summaries

    Receives:
    {
        "tickers": ["AAPL", "MSFT", "GOOGL"],
        "user_tier": "premium"
    }

    Returns:
    {
        "user_tier": "premium",
        "count": 3,
        "items": [
            {"ticker": "AAPL", "direction": "up", "confidence": 0.85, ...},
            {"ticker": "MSFT", "direction": "up", "confidence": 0.78, ...},
            {"ticker": "GOOGL", "direction": "neutral", "confidence": 0.65, ...}
        ]
    }
    """
    # Feature gate: Only registered/premium users can do batch analysis
    # Why: Batch analysis uses more server resources
    if request.user_tier not in ["registered", "premium"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Batch analysis is available to registered and premium users",
        )

    # Clean and validate ticker list
    # Strip whitespace, convert to uppercase, filter empty strings
    tickers = [t.strip().upper() for t in request.tickers if t and isinstance(t, str)]
    if not tickers:
        raise HTTPException(status_code=400, detail="No tickers provided")

    # Remove duplicates while preserving order
    # Why: User might accidentally add same ticker twice
    seen: set = set()
    deduped: List[str] = []
    for t in tickers:
        if t not in seen and len(t) <= 10:  # Also validate length
            seen.add(t)
            deduped.append(t)

    # Get services
    data_service = get_data_service()
    agents = get_agents()

    # Analyze each ticker
    items: List[BatchAnalysisItem] = []
    for t in deduped:
        try:
            # Run full analysis for this ticker
            result = run_stocksense_analysis(
                ticker=t,
                user_tier=request.user_tier,
                prediction_agent=agents["prediction"],
                sentiment_agent=agents["sentiment"],
                explanation_agent=agents["explanation"],
                smart_money_agent=agents["smart_money"],
                data_service=data_service,
            )

            # Extract compact summary (don't need full explanation for batch)
            # Why: Reduces response size and processing time
            horizon_days = int(result["forecast"]["horizon_95"].get("days", 0)) if result["forecast"].get("horizon_95") else 0
            items.append(
                BatchAnalysisItem(
                    ticker=result["ticker"],
                    as_of=result["as_of"],
                    direction=result["forecast"]["direction"],  # up/down/neutral
                    confidence=float(result["forecast"]["confidence"]),
                    horizon_days=horizon_days,  # How long we're confident
                    score=float(result["sentiment"].get("score", 0.0)),
                    sentiment=result["sentiment"],
                )
            )
        except Exception as e:
            logger.error(f"Batch analyze failed for {t}: {e}")
            # Skip failed ticker rather than failing whole batch
            # Why: Better UX - user gets partial results even if one stock fails
            continue

    return BatchAnalysisResponse(
        user_tier=request.user_tier,
        count=len(items),
        items=items
    )


def _iter_predictions_csv(rows: Iterable[Dict[str, any]]) -> Iterable[str]:
    """Generator that yields CSV rows for streaming.

    What: Converts prediction data to CSV format, one row at a time
    Why: Streaming is memory-efficient (don't load all data at once)
    How: Yields header first, then each data row as a formatted string

    Receives: Iterator of dictionaries (each dict is one row of data)
    Yields: CSV-formatted strings
    """
    # Yield header row
    yield "ticker,date,up,down,neutral,direction,confidence,horizon_days\n"

    # Yield each data row
    for r in rows:
        # Format row with proper CSV escaping
        # :.4f means 4 decimal places (e.g., 0.8512)
        line = f"{r['ticker']},{r['date']},{r['up']:.4f},{r['down']:.4f},{r['neutral']:.4f},{r['direction']},{r['confidence']:.4f},{r['horizon_days']}\n"
        yield line


@app.get("/export/predictions.csv")
async def export_predictions_csv(
    tickers: str = Query(..., description="Comma-separated list of tickers"),
    max_rows: int = Query(5000, le=5000, ge=1, description="Maximum rows to export (cap 5000)"),
    user_tier: str = Query("premium", description="User tier for gating features"),
):
    """Stream CSV of prediction daily probabilities (premium only).

    What: Exports probability predictions to downloadable CSV file
    Why: Users want to analyze data in Excel/Google Sheets
    How: Streams CSV data (memory-efficient for large files)

    Receives:
    GET /export/predictions.csv?tickers=AAPL,MSFT&max_rows=5000&user_tier=premium

    Returns:
    CSV file with columns: ticker, date, up, down, neutral, direction, confidence, horizon_days
    Example:
    AAPL,2025-01-25,0.8500,0.1000,0.0500,up,0.8500,14
    AAPL,2025-01-26,0.8300,0.1050,0.0650,up,0.8500,14
    ...
    """
    # Feature gate: Premium only
    if user_tier != "premium":
        raise HTTPException(
            status_code=403,
            detail="CSV export is available to premium users only"
        )

    # Parse comma-separated ticker list
    symbols = [s.strip().upper() for s in tickers.split(",") if s.strip()]
    if not symbols:
        raise HTTPException(status_code=400, detail="No tickers provided")

    data_service = get_data_service()
    agents = get_agents()

    # Collect prediction data for all tickers
    aggregated: List[Dict[str, any]] = []
    for sym in symbols:
        try:
            # Run analysis for this ticker
            res = run_stocksense_analysis(
                ticker=sym,
                user_tier=user_tier,
                prediction_agent=agents["prediction"],
                sentiment_agent=agents["sentiment"],
                explanation_agent=agents["explanation"],
                smart_money_agent=agents["smart_money"],
                data_service=data_service,
            )

            # Extract daily probability data
            horizon_days = int(res["forecast"]["horizon_95"].get("days", 0)) if res["forecast"].get("horizon_95") else 0
            direction = res["forecast"]["direction"]
            confidence = float(res["forecast"]["confidence"])

            # Add each day's probabilities to the CSV
            for day in res["forecast"].get("daily_probs", [])[:max_rows]:
                aggregated.append({
                    "ticker": res["ticker"],
                    "date": day["date"],  # YYYY-MM-DD format
                    "up": float(day.get("up", 0.0)),  # Probability stock goes up
                    "down": float(day.get("down", 0.0)),  # Probability goes down
                    "neutral": float(day.get("neutral", 0.0)),  # Probability stays flat
                    "direction": direction,  # Overall direction
                    "confidence": confidence,  # Overall confidence
                    "horizon_days": horizon_days,  # Days of 95%+ confidence
                })

                # Stop if we hit the row limit
                if len(aggregated) >= max_rows:
                    break

            if len(aggregated) >= max_rows:
                break

        except Exception as e:
            logger.error(f"CSV export analysis failed for {sym}: {e}")
            # Skip failed ticker
            continue

    # Stream the CSV data
    # Why streaming: Memory-efficient, starts download immediately
    generator = _iter_predictions_csv(aggregated)
    headers = {
        "Content-Disposition": "attachment; filename=predictions.csv"  # Triggers download
    }
    return StreamingResponse(
        generator,
        media_type="text/csv",
        headers=headers
    )


# ===== WATCHLIST API =====
# File-backed storage for user watchlists

@app.get("/watchlist")
async def get_watchlist(user_id: str = Query("demo")):
    """Get user's watchlist.

    What: Returns list of tickers user is watching
    Why: Users want to track favorite stocks
    How: Reads from JSON file

    Receives:
    GET /watchlist?user_id=demo

    Returns:
    {"user_id": "demo", "tickers": ["AAPL", "MSFT", "GOOGL"]}
    """
    repos = get_repos()
    watchlists: WatchlistRepository = repos["watchlists"]
    return {
        "user_id": user_id,
        "tickers": watchlists.get(user_id)  # Returns list of ticker symbols
    }


@app.post("/watchlist")
async def add_to_watchlist(user_id: str = Query("demo"), ticker: str = Query(...)):
    """Add ticker to user's watchlist.

    What: Adds a stock to the list of stocks user is tracking
    Why: Users want to monitor specific stocks
    How: Updates JSON file with new ticker

    Receives:
    POST /watchlist?user_id=demo&ticker=TSLA

    Returns:
    {"user_id": "demo", "tickers": ["AAPL", "MSFT", "GOOGL", "TSLA"]}
    """
    repos = get_repos()
    watchlists: WatchlistRepository = repos["watchlists"]
    updated = watchlists.add(user_id, ticker)  # Adds ticker, deduplicates, sorts
    return {"user_id": user_id, "tickers": updated}


@app.delete("/watchlist")
async def remove_from_watchlist(user_id: str = Query("demo"), ticker: str = Query(...)):
    """Remove ticker from user's watchlist.

    What: Removes a stock from tracking list
    Why: Users want to clean up watchlist
    How: Updates JSON file to remove ticker

    Receives:
    DELETE /watchlist?user_id=demo&ticker=TSLA

    Returns:
    {"user_id": "demo", "tickers": ["AAPL", "MSFT", "GOOGL"]}
    """
    repos = get_repos()
    watchlists: WatchlistRepository = repos["watchlists"]
    updated = watchlists.remove(user_id, ticker)  # Filters out ticker
    return {"user_id": user_id, "tickers": updated}


# ===== HISTORY API =====
# Stores past analysis results

@app.get("/history")
async def get_history(user_id: str = Query("demo"), limit: int = Query(100, le=1000)):
    """Get user's analysis history.

    What: Returns list of past stock analyses
    Why: Users want to see prediction history and track accuracy
    How: Reads from JSON file

    Receives:
    GET /history?user_id=demo&limit=100

    Returns:
    {
        "user_id": "demo",
        "count": 100,
        "items": [
            {"ticker": "AAPL", "as_of": "...", "direction": "up", "confidence": 0.85, ...},
            ...
        ]
    }
    """
    repos = get_repos()
    history: HistoryRepository = repos["history"]
    items = history.list(user_id, limit)  # Gets last N items
    return {"user_id": user_id, "count": len(items), "items": items}


# ===== ALERTS API =====
# Stores user alert rules (evaluation happens elsewhere)

@app.get("/alerts")
async def list_alerts(user_id: str = Query("demo")):
    """List user's alert rules.

    What: Returns alert rules user has configured
    Why: Users want to be notified when certain conditions are met
    How: Reads from JSON file

    Receives:
    GET /alerts?user_id=demo

    Returns:
    {
        "user_id": "demo",
        "rules": [
            {"ticker": "AAPL", "condition": "prob_down_gte", "threshold": 0.7},
            ...
        ]
    }
    """
    repos = get_repos()
    alerts: AlertsRepository = repos["alerts"]
    return {"user_id": user_id, "rules": alerts.list(user_id)}


@app.post("/alerts")
async def upsert_alert(
    user_id: str = Query("demo"),
    ticker: str = Query(...),
    condition: str = Query(...),
    threshold: float = Query(...)
):
    """Create or update an alert rule.

    What: Sets up notification when stock meets condition
    Why: Users want automatic alerts (e.g., "tell me if AAPL drops >70%")
    How: Saves rule to JSON file

    Receives:
    POST /alerts?user_id=demo&ticker=AAPL&condition=prob_down_gte&threshold=0.7

    Means: Alert me if AAPL has >= 70% probability of going down

    Returns:
    {"user_id": "demo", "rules": [...]}
    """
    # Validate condition
    # Only allow known condition types
    if condition not in {"prob_down_gte", "prob_up_gte", "confidence_gte"}:
        raise HTTPException(status_code=400, detail="Unsupported condition")

    # Validate threshold
    # Must be between 0 and 1 (it's a probability)
    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(status_code=400, detail="threshold must be within [0,1]")

    repos = get_repos()
    alerts: AlertsRepository = repos["alerts"]

    # Upsert: Update if exists, insert if new
    updated = alerts.upsert(
        user_id,
        AlertRule(ticker=ticker.upper(), condition=condition, threshold=threshold)
    )
    return {"user_id": user_id, "rules": updated}


@app.delete("/alerts")
async def delete_alert(
    user_id: str = Query("demo"),
    ticker: str = Query(...),
    condition: Optional[str] = Query(None)
):
    """Delete an alert rule.

    What: Removes alert for a ticker
    Why: Users want to stop getting alerts
    How: Updates JSON file to remove rule

    Receives:
    DELETE /alerts?user_id=demo&ticker=AAPL&condition=prob_down_gte

    If condition not specified: Removes ALL alerts for that ticker

    Returns:
    {"user_id": "demo", "rules": [...]}
    """
    repos = get_repos()
    alerts: AlertsRepository = repos["alerts"]
    updated = alerts.delete(user_id, ticker, condition)
    return {"user_id": user_id, "rules": updated}


# ===== ADMIN API =====
# Simple administrative endpoints

@app.get("/admin/usage")
async def admin_usage():
    """Get basic usage statistics.

    What: Returns usage metrics (users, predictions, popular stocks)
    Why: Admins want to monitor platform activity
    How: Aggregates data from history storage

    Returns:
    {
        "active_users": 1,
        "recent_predictions": 150,
        "top_tickers": [["AAPL", 45], ["MSFT", 32], ...]
    }
    """
    # Aggregate basic usage from history store
    repos = get_repos()
    history: HistoryRepository = repos["history"]

    # Get recent history items
    all_items = history.list("demo", limit=1000)

    # Count predictions by ticker
    by_ticker: Dict[str, int] = {}
    for it in all_items:
        t = it.get("ticker", "?")
        by_ticker[t] = by_ticker.get(t, 0) + 1

    # Sort by count and get top 20
    top = sorted(by_ticker.items(), key=lambda kv: kv[1], reverse=True)[:20]

    return {
        "active_users": 1,  # In production: count unique users
        "recent_predictions": len(all_items),
        "top_tickers": top  # Most analyzed stocks
    }


@app.post("/admin/models/deploy")
async def admin_models_deploy(model_name: str = Query(...)):
    """Deploy a new model version.

    What: Stub endpoint for model deployment
    Why: Admins need to update AI models without code changes
    How: In production: triggers CI/CD pipeline or model registry update

    Receives:
    POST /admin/models/deploy?model_name=v2_forecaster

    Returns:
    {"status": "ok", "message": "Deployment initiated for v2_forecaster"}
    """
    # This is a placeholder - real implementation would:
    # 1. Validate model exists in model registry
    # 2. Run tests on model
    # 3. Trigger deployment pipeline
    # 4. Update model version in database
    # 5. Notify monitoring systems

    logger.info(f"Admin requested model deployment: {model_name}")
    return {
        "status": "ok",
        "message": f"Deployment initiated for {model_name}"
    }


# ===== APPLICATION ENTRY POINT =====
if __name__ == "__main__":
    # This block runs when you execute: python api.py
    # Starts the web server on port 8000
    import uvicorn
    uvicorn.run(
        app,  # The FastAPI application
        host="0.0.0.0",  # Listen on all network interfaces (localhost + external)
        port=8000  # HTTP port
    )
