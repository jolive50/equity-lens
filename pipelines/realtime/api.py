"""Main StockSense API endpoint integrating LangGraph workflow with data adapters."""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional, List, Iterable

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from .data_adapters import DataAdapterFactory, DataService
from .langgraph_workflow import run_stocksense_analysis
from .agents import (
    ExplanationAgent,
    PredictionAgent,
    SentimentAgent,
    SmartMoneyAgent,
    build_openai_llm,
    build_mock_llm,
)
from .repository import JsonFileRepository, WatchlistRepository, HistoryRepository, AlertsRepository, AlertRule

# Load environment variables
load_dotenv()

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


class BatchAnalysisRequest(BaseModel):
    tickers: List[str]
    user_tier: str = "premium"


class BatchAnalysisItem(BaseModel):
    ticker: str
    as_of: str
    direction: str
    confidence: float
    horizon_days: int
    score: float
    sentiment: Dict


class BatchAnalysisResponse(BaseModel):
    user_tier: str
    count: int
    items: List[BatchAnalysisItem]


# Global service instances (in production, these would be dependency injected)
_data_service: Optional[DataService] = None
_agents: Optional[Dict[str, any]] = None
_watchlists: Optional[WatchlistRepository] = None
_history: Optional[HistoryRepository] = None
_alerts: Optional[AlertsRepository] = None


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
        # Try to use OpenAI LLM, fallback to mock if API key not available
        try:
            llm = build_openai_llm("gpt-4o-mini")
            logger.info("Using OpenAI GPT-4o-mini for agents")
        except ValueError as e:
            logger.warning(f"OpenAI not available: {e}. Using mock LLM.")
            llm = build_mock_llm("stocksense")
        
        _agents = {
            "prediction": PredictionAgent(llm),
            "sentiment": SentimentAgent(llm),
            "explanation": ExplanationAgent(llm),
            "smart_money": SmartMoneyAgent(llm),
        }
    
    return _agents


def get_repos() -> Dict[str, any]:
    global _watchlists, _history, _alerts
    if _watchlists is None or _history is None or _alerts is None:
        base_dir = os.getenv("APP_DATA_DIR", os.path.join(os.getcwd(), "..", "data", "realtime", "feature_store"))
        wl_store = JsonFileRepository(os.path.join(base_dir, "_app", "watchlists.json"))
        hist_store = JsonFileRepository(os.path.join(base_dir, "_app", "history.json"))
        al_store = JsonFileRepository(os.path.join(base_dir, "_app", "alerts.json"))
        _watchlists = WatchlistRepository(wl_store)
        _history = HistoryRepository(hist_store)
        _alerts = AlertsRepository(al_store)
    return {"watchlists": _watchlists, "history": _history, "alerts": _alerts}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "StockSense API",
        "version": "1.0.0",
        "description": "Layperson-friendly stock insights with AI-powered forecasting",
        "endpoints": {
            "/analyze": "Main analysis endpoint",
            "/analyze/batch": "Batch analysis endpoint (registered/premium)",
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
        
        if request.user_tier not in ["basic", "registered", "premium"]:
            raise HTTPException(
                status_code=400,
                detail="User tier must be 'basic', 'registered' or 'premium'"
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
            data_service=data_service,
        )
        
        # Append compact history entry for the anonymous demo user
        repos = get_repos()
        history: HistoryRepository = repos["history"]
        history.append(
            user_id="demo", 
            entry={
                "ticker": result["ticker"],
                "as_of": result["as_of"],
                "direction": result["forecast"]["direction"],
                "confidence": float(result["forecast"]["confidence"]),
                "score": float(result.get("sentiment", {}).get("score", 0.0)),
            }
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
    ticker: str = Query(..., description="Stock ticker symbol"),
    user_tier: str = Query("basic", description="User tier for gating features"),
):
    """Get smart money data for a ticker."""
    try:
        if user_tier != "premium":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Smart money data is available to premium users only",
            )
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


@app.post("/analyze/batch", response_model=BatchAnalysisResponse)
async def analyze_batch(request: BatchAnalysisRequest):
    """Batch analysis for multiple tickers. Registered and Premium only.

    For each ticker, returns compact prediction summary suitable for watchlists.
    """
    # Role guard
    if request.user_tier not in ["registered", "premium"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Batch analysis is available to registered and premium users",
        )

    # Basic validation
    tickers = [t.strip().upper() for t in request.tickers if t and isinstance(t, str)]
    if not tickers:
        raise HTTPException(status_code=400, detail="No tickers provided")

    # De-duplicate while preserving order
    seen: set = set()
    deduped: List[str] = []
    for t in tickers:
        if t not in seen and len(t) <= 10:
            seen.add(t)
            deduped.append(t)

    data_service = get_data_service()
    agents = get_agents()

    items: List[BatchAnalysisItem] = []
    for t in deduped:
        try:
            result = run_stocksense_analysis(
                ticker=t,
                user_tier=request.user_tier,
                prediction_agent=agents["prediction"],
                sentiment_agent=agents["sentiment"],
                explanation_agent=agents["explanation"],
                smart_money_agent=agents["smart_money"],
                data_service=data_service,
            )

            horizon_days = int(result["forecast"]["horizon_95"].get("days", 0)) if result["forecast"].get("horizon_95") else 0
            items.append(
                BatchAnalysisItem(
                    ticker=result["ticker"],
                    as_of=result["as_of"],
                    direction=result["forecast"]["direction"],
                    confidence=float(result["forecast"]["confidence"]),
                    horizon_days=horizon_days,
                    score=float(result["sentiment"].get("score", 0.0)),
                    sentiment=result["sentiment"],
                )
            )
        except Exception as e:
            logger.error(f"Batch analyze failed for {t}: {e}")
            # Skip failed ticker rather than failing whole batch
            continue

    return BatchAnalysisResponse(user_tier=request.user_tier, count=len(items), items=items)


def _iter_predictions_csv(rows: Iterable[Dict[str, any]]) -> Iterable[str]:
    # Header
    yield "ticker,date,up,down,neutral,direction,confidence,horizon_days\n"
    for r in rows:
        line = f"{r['ticker']},{r['date']},{r['up']:.4f},{r['down']:.4f},{r['neutral']:.4f},{r['direction']},{r['confidence']:.4f},{r['horizon_days']}\n"
        yield line


@app.get("/export/predictions.csv")
async def export_predictions_csv(
    tickers: str = Query(..., description="Comma-separated list of tickers"),
    max_rows: int = Query(5000, le=5000, ge=1, description="Maximum rows to export (cap 5000)"),
    user_tier: str = Query("premium", description="User tier for gating features"),
):
    """Stream CSV of prediction daily probabilities for one or more tickers.

    Guarantees at most 5,000 rows to meet performance SLO.
    """
    if user_tier != "premium":
        raise HTTPException(status_code=403, detail="CSV export is available to premium users only")

    symbols = [s.strip().upper() for s in tickers.split(",") if s.strip()]
    if not symbols:
        raise HTTPException(status_code=400, detail="No tickers provided")

    data_service = get_data_service()
    agents = get_agents()

    aggregated: List[Dict[str, any]] = []
    for sym in symbols:
        try:
            res = run_stocksense_analysis(
                ticker=sym,
                user_tier=user_tier,
                prediction_agent=agents["prediction"],
                sentiment_agent=agents["sentiment"],
                explanation_agent=agents["explanation"],
                smart_money_agent=agents["smart_money"],
                data_service=data_service,
            )
            horizon_days = int(res["forecast"]["horizon_95"].get("days", 0)) if res["forecast"].get("horizon_95") else 0
            direction = res["forecast"]["direction"]
            confidence = float(res["forecast"]["confidence"])  # overall
            for day in res["forecast"].get("daily_probs", [])[:max_rows]:
                aggregated.append(
                    {
                        "ticker": res["ticker"],
                        "date": day["date"],
                        "up": float(day.get("up", 0.0)),
                        "down": float(day.get("down", 0.0)),
                        "neutral": float(day.get("neutral", 0.0)),
                        "direction": direction,
                        "confidence": confidence,
                        "horizon_days": horizon_days,
                    }
                )
                if len(aggregated) >= max_rows:
                    break
            if len(aggregated) >= max_rows:
                break
        except Exception as e:
            logger.error(f"CSV export analysis failed for {sym}: {e}")
            continue

    generator = _iter_predictions_csv(aggregated)
    headers = {
        "Content-Disposition": "attachment; filename=predictions.csv"
    }
    return StreamingResponse(generator, media_type="text/csv", headers=headers)


# Watchlists API (file-backed)
@app.get("/watchlist")
async def get_watchlist(user_id: str = Query("demo")):
    repos = get_repos()
    watchlists: WatchlistRepository = repos["watchlists"]
    return {"user_id": user_id, "tickers": watchlists.get(user_id)}


@app.post("/watchlist")
async def add_to_watchlist(user_id: str = Query("demo"), ticker: str = Query(...)):
    repos = get_repos()
    watchlists: WatchlistRepository = repos["watchlists"]
    updated = watchlists.add(user_id, ticker)
    return {"user_id": user_id, "tickers": updated}


@app.delete("/watchlist")
async def remove_from_watchlist(user_id: str = Query("demo"), ticker: str = Query(...)):
    repos = get_repos()
    watchlists: WatchlistRepository = repos["watchlists"]
    updated = watchlists.remove(user_id, ticker)
    return {"user_id": user_id, "tickers": updated}


# History API
@app.get("/history")
async def get_history(user_id: str = Query("demo"), limit: int = Query(100, le=1000)):
    repos = get_repos()
    history: HistoryRepository = repos["history"]
    items = history.list(user_id, limit)
    return {"user_id": user_id, "count": len(items), "items": items}


# Alerts API (store only; evaluation hooks are left to scheduler/webhook)
@app.get("/alerts")
async def list_alerts(user_id: str = Query("demo")):
    repos = get_repos()
    alerts: AlertsRepository = repos["alerts"]
    return {"user_id": user_id, "rules": alerts.list(user_id)}


@app.post("/alerts")
async def upsert_alert(user_id: str = Query("demo"), ticker: str = Query(...), condition: str = Query(...), threshold: float = Query(...)):
    if condition not in {"prob_down_gte", "prob_up_gte", "confidence_gte"}:
        raise HTTPException(status_code=400, detail="Unsupported condition")
    if not (0.0 <= threshold <= 1.0):
        raise HTTPException(status_code=400, detail="threshold must be within [0,1]")
    repos = get_repos()
    alerts: AlertsRepository = repos["alerts"]
    updated = alerts.upsert(user_id, AlertRule(ticker=ticker.upper(), condition=condition, threshold=threshold))
    return {"user_id": user_id, "rules": updated}


@app.delete("/alerts")
async def delete_alert(user_id: str = Query("demo"), ticker: str = Query(...), condition: Optional[str] = Query(None)):
    repos = get_repos()
    alerts: AlertsRepository = repos["alerts"]
    updated = alerts.delete(user_id, ticker, condition)
    return {"user_id": user_id, "rules": updated}


# Admin API (minimal)
@app.get("/admin/usage")
async def admin_usage():
    # Aggregate basic usage from history store
    repos = get_repos()
    history: HistoryRepository = repos["history"]
    all_items = history.list("demo", limit=1000)
    by_ticker: Dict[str, int] = {}
    for it in all_items:
        t = it.get("ticker", "?")
        by_ticker[t] = by_ticker.get(t, 0) + 1
    top = sorted(by_ticker.items(), key=lambda kv: kv[1], reverse=True)[:20]
    return {"active_users": 1, "recent_predictions": len(all_items), "top_tickers": top}


@app.post("/admin/models/deploy")
async def admin_models_deploy(model_name: str = Query(...)):
    # Stub deployment hook – in a real system, trigger CI/CD or model registry update
    logger.info(f"Admin requested model deployment: {model_name}")
    return {"status": "ok", "message": f"Deployment initiated for {model_name}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
