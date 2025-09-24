"""Main StockSense API endpoint integrating LangGraph workflow with data adapters."""
from __future__ import annotations

import json
import logging
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .data_adapters import DataAdapterFactory, DataService
from .langgraph_workflow import run_stocksense_analysis
from .agents import (
    ExplanationAgent,
    PredictionAgent,
    SentimentAgent,
    SmartMoneyAgent,
    build_mock_llm,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="StockSense API",
    description="Layperson-friendly stock insights with AI-powered forecasting",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class AnalysisRequest(BaseModel):
    ticker: str
    user_tier: str = "basic"


class AnalysisResponse(BaseModel):
    ticker: str
    as_of: str
    forecast: Dict
    metrics: Dict
    sentiment: Dict
    smart_money: Dict
    explanation: str
    warnings: list
    disclaimers: list


# Global service instances (in production, these would be dependency injected)
_data_service: Optional[DataService] = None
_agents: Optional[Dict[str, any]] = None


def get_data_service() -> DataService:
    """Get or create the data service instance."""
    global _data_service
    if _data_service is None:
        # Initialize adapters with API keys from environment
        import os
        adapters = {}
        
        if os.getenv("ALPHA_VANTAGE_API_KEY"):
            adapters["alpha_vantage"] = DataAdapterFactory.create_adapter(
                "alpha_vantage", os.getenv("ALPHA_VANTAGE_API_KEY")
            )
        
        if os.getenv("TIINGO_API_KEY"):
            adapters["tiingo"] = DataAdapterFactory.create_adapter(
                "tiingo", os.getenv("TIINGO_API_KEY")
            )
        
        if os.getenv("FINNHUB_API_KEY"):
            adapters["finnhub"] = DataAdapterFactory.create_adapter(
                "finnhub", os.getenv("FINNHUB_API_KEY")
            )
        
        if os.getenv("NEWSAPI_KEY"):
            adapters["newsapi"] = DataAdapterFactory.create_adapter(
                "newsapi", os.getenv("NEWSAPI_KEY")
            )
        
        _data_service = DataService(adapters)
    
    return _data_service


def get_agents() -> Dict[str, any]:
    """Get or create the agent instances."""
    global _agents
    if _agents is None:
        # In production, this would use a real LLM like OpenAI GPT-4
        # For now, using mock LLM for demonstration
        llm = build_mock_llm("stocksense")
        
        _agents = {
            "prediction": PredictionAgent(llm),
            "sentiment": SentimentAgent(llm),
            "explanation": ExplanationAgent(llm),
            "smart_money": SmartMoneyAgent(llm),
        }
    
    return _agents


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "StockSense API",
        "version": "1.0.0",
        "description": "Layperson-friendly stock insights with AI-powered forecasting",
        "endpoints": {
            "/analyze": "Main analysis endpoint",
            "/health": "Health check",
            "/docs": "API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        data_service = get_data_service()
        agents = get_agents()
        
        return {
            "status": "healthy",
            "data_adapters": len(data_service.adapters),
            "agents": len(agents),
            "timestamp": "2025-01-24T10:30:00Z"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unhealthy")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_stock(request: AnalysisRequest):
    """Main analysis endpoint that runs the complete StockSense workflow."""
    try:
        # Validate input
        if not request.ticker or len(request.ticker) > 10:
            raise HTTPException(
                status_code=400, 
                detail="Invalid ticker symbol"
            )
        
        if request.user_tier not in ["basic", "premium"]:
            raise HTTPException(
                status_code=400,
                detail="User tier must be 'basic' or 'premium'"
            )
        
        # Get services
        data_service = get_data_service()
        agents = get_agents()
        
        # Run the StockSense analysis workflow
        result = run_stocksense_analysis(
            ticker=request.ticker.upper(),
            user_tier=request.user_tier,
            prediction_agent=agents["prediction"],
            sentiment_agent=agents["sentiment"],
            explanation_agent=agents["explanation"],
            smart_money_agent=agents["smart_money"],
        )
        
        return AnalysisResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed for {request.ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed - please try again later"
        )


@app.get("/prices")
async def get_prices(
    ticker: str = Query(..., description="Stock ticker symbol"),
    source: str = Query("alpha_vantage", description="Data source"),
    limit: int = Query(200, description="Number of data points", le=2000)
):
    """Get historical price data for a ticker."""
    try:
        data_service = get_data_service()
        
        if source not in data_service.adapters:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown source: {source}"
            )
        
        adapter = data_service.adapters[source]
        historical_data = adapter.fetch_historical_data(ticker.upper(), limit)
        
        # Convert to API format
        items = []
        for data in historical_data:
            items.append({
                "date": data.timestamp,
                "close": data.close,
                "open": data.open_price,
                "high": data.high,
                "low": data.low,
                "volume": data.volume
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
    """Get news articles for a ticker."""
    try:
        data_service = get_data_service()
        
        if source not in data_service.adapters:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown source: {source}"
            )
        
        adapter = data_service.adapters[source]
        news_items = adapter.fetch_news(ticker.upper(), limit)
        
        # Convert to API format
        items = []
        for item in news_items:
            items.append({
                "title": item.title,
                "content": item.content,
                "source": item.source,
                "url": item.url,
                "timestamp": item.timestamp,
                "sentiment_score": item.sentiment_score
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
    """Get sentiment analysis for a ticker."""
    try:
        data_service = get_data_service()
        agents = get_agents()
        
        # Get news from all sources
        all_news = data_service.get_all_news(ticker.upper())
        
        # Run sentiment analysis
        sentiment_result = agents["sentiment"].run(
            ticker=ticker.upper(),
            news_data=all_news[:limit]
        )
        
        return {
            "ticker": ticker.upper(),
            "current": sentiment_result.current,
            "score": sentiment_result.score,
            "trend": sentiment_result.trend,
            "headlines": sentiment_result.headlines,
            "news_count": len(all_news)
        }
        
    except Exception as e:
        logger.error(f"Sentiment analysis failed for {ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze sentiment"
        )


@app.get("/smart-money")
async def get_smart_money(
    ticker: str = Query(..., description="Stock ticker symbol")
):
    """Get smart money data for a ticker."""
    try:
        agents = get_agents()
        
        # Run smart money analysis
        smart_money_result = agents["smart_money"].run(ticker=ticker.upper())
        
        return {
            "ticker": ticker.upper(),
            "institutions": smart_money_result.institutions,
            "insiders": smart_money_result.insiders,
            "congress": smart_money_result.congress
        }
        
    except Exception as e:
        logger.error(f"Smart money analysis failed for {ticker}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to analyze smart money data"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
