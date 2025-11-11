"""LangGraph workflow for FreshStart stock analysis.

JOSH's Component - Workflow Orchestration
Simplified LangGraph workflow connecting all agents.
"""
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd
from langgraph.graph import StateGraph

from agents.explanation_agent import ExplanationAgent
from agents.prediction_agent import PredictionAgent
from agents.reflection_agent import ReflectionAgent
from agents.sentiment_agent import SentimentAgent
from storage.database import Database
from storage.vector_store import NewsVectorStore

logger = logging.getLogger("freshstart.coordinator")

_db_client: Optional[Database] = None
_db_init_failed = False
_vector_store_client: Optional[NewsVectorStore] = None
_vector_store_init_failed = False

# Cache reuse configuration
PRICE_CACHE_TTL_DAYS = 1
PRICE_CACHE_MIN_ROWS = 30
NEWS_CACHE_TTL_MINUTES = 60


def _get_db_client() -> Optional[Database]:
    """Lazily initialize and return the shared database client."""
    global _db_client, _db_init_failed
    if _db_client or _db_init_failed:
        return _db_client

    try:
        _db_client = Database()
        logger.info("SQLite caching enabled")
    except Exception as exc:  # pragma: no cover - defensive log guard
        logger.warning("Database unavailable, continuing without cache: %s", exc)
        _db_init_failed = True
    return _db_client


def _get_vector_store() -> Optional[NewsVectorStore]:
    """Lazily initialize and return the shared ChromaDB client."""
    global _vector_store_client, _vector_store_init_failed
    if _vector_store_client or _vector_store_init_failed:
        return _vector_store_client

    try:
        _vector_store_client = NewsVectorStore()
        logger.info("ChromaDB vector store enabled")
    except Exception as exc:  # pragma: no cover - defensive log guard
        logger.warning("Vector store unavailable, continuing without embeddings: %s", exc)
        _vector_store_init_failed = True
    return _vector_store_client


def _cache_market_data(ticker: str, market_data: List[Dict[str, Any]]) -> None:
    """Persist fetched market data to SQLite for reuse."""
    db = _get_db_client()
    if not db or not market_data:
        return

    try:
        df = pd.DataFrame(market_data)
        if df.empty or "date" not in df.columns:
            return

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        df = df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        inserted = db.cache_prices(ticker.upper(), df)
        if inserted:
            logger.info("Cached %d price rows for %s", inserted, ticker)
    except Exception as exc:
        logger.warning("Failed to cache price data for %s: %s", ticker, exc)


def _load_cached_market_data(
    ticker: str,
    min_rows: int = 0,
    max_age_days: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Load cached market data with optional freshness requirements.

    Args:
        ticker: Stock symbol
        min_rows: Minimum rows required to consider cache valid
        max_age_days: Maximum age (based on latest trading day) for cache to be valid

    Returns:
        List of market data rows (may be empty if requirements unmet)
    """
    db = _get_db_client()
    if not db:
        return []

    try:
        cached = db.get_cached_prices(ticker.upper())
    except Exception as exc:
        logger.warning("Failed to load cached price data for %s: %s", ticker, exc)
        return []

    if cached.empty:
        return []

    if min_rows and len(cached) < min_rows:
        return []

    if max_age_days is not None:
        latest_row = cached.index.max()
        try:
            latest_dt = latest_row.to_pydatetime()
        except AttributeError:
            if isinstance(latest_row, datetime):
                latest_dt = latest_row
            else:
                latest_dt = datetime.combine(latest_row, datetime.min.time())

        if datetime.utcnow() - latest_dt > timedelta(days=max_age_days):
            return []

    cached = cached.reset_index()
    results: List[Dict[str, Any]] = []
    for _, row in cached.iterrows():
        try:
            date_str = row["date"].strftime("%Y-%m-%d")
        except AttributeError:
            date_str = str(row["date"])

        results.append(
            {
                "date": date_str,
                "open": round(float(row.get("open", 0.0)), 2),
                "high": round(float(row.get("high", 0.0)), 2),
                "low": round(float(row.get("low", 0.0)), 2),
                "close": round(float(row.get("close", 0.0)), 2),
                "volume": int(row.get("volume", 0)),
            }
        )
    return results


def _cache_news_articles(ticker: str, articles: List[Dict[str, Any]]) -> None:
    """Persist fetched news articles to SQLite."""
    db = _get_db_client()
    if not db or not articles:
        return

    payload = []
    for article in articles:
        sentiment = article.get("sentiment") or {}
        payload.append(
            {
                "title": article.get("title", ""),
                "url": article.get("url", ""),
                "published_at": article.get("timestamp", datetime.utcnow().isoformat()),
                "source": article.get("source", "unknown"),
                "summary": article.get("content", "") or article.get("summary", ""),
                "sentiment_label": sentiment.get("label"),
                "sentiment_score": sentiment.get("score"),
            }
        )

    try:
        inserted = db.cache_news(ticker.upper(), payload)
        if inserted:
            logger.info("Cached %d news articles for %s", inserted, ticker)
    except Exception as exc:
        logger.warning("Failed to cache news for %s: %s", ticker, exc)


def _persist_news_embeddings(ticker: str, articles: List[Dict[str, Any]]) -> None:
    """Store news articles in ChromaDB for semantic recall."""
    vector_store = _get_vector_store()
    if not vector_store or not articles:
        return

    normalized_articles = []
    for article in articles:
        sentiment = article.get("sentiment") or {}
        normalized_articles.append(
            {
                "title": article.get("title", ""),
                "content": article.get("content", "") or article.get("summary", ""),
                "timestamp": article.get("timestamp", datetime.utcnow().isoformat()),
                "source": article.get("source", "unknown"),
                "sentiment": sentiment.get("label", "neutral"),
                "sentiment_score": float(sentiment.get("score", 0.0)),
            }
        )

    try:
        vector_store.add_news_articles(ticker.upper(), normalized_articles)
        logger.info("Indexed %d news articles for %s", len(normalized_articles), ticker)
    except Exception as exc:
        logger.warning("Failed to index news for %s: %s", ticker, exc)


def _load_cached_news(
    ticker: str,
    limit: int = 50,
    max_age_minutes: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Retrieve cached news, optionally enforcing freshness."""
    db = _get_db_client()
    if not db:
        return []

    try:
        cached = db.get_cached_news(
            ticker.upper(),
            limit=limit,
            max_age_minutes=max_age_minutes
        )
    except Exception as exc:
        logger.warning("Failed to load cached news for %s: %s", ticker, exc)
        return []

    if not cached:
        return []

    results: List[Dict[str, Any]] = []
    for article in cached:
        sentiment_label = article.get("sentiment_label", "Neutral")
        results.append(
            {
                "title": article.get("title", ""),
                "content": article.get("summary", ""),
                "source": article.get("source", "unknown"),
                "timestamp": article.get("published_at", datetime.utcnow().isoformat()),
                "url": article.get("url", ""),
                "sentiment": {
                    "label": sentiment_label,
                    "score": float(article.get("sentiment_score", 0.0)),
                },
                "ticker_sentiment": [],
            }
        )

    return results


class StockAnalysisState(TypedDict, total=False):
    """State for stock analysis workflow."""
    ticker: str
    user_tier: str  # "basic" or "premium"
    market_data: List[Dict[str, Any]]
    fundamentals: Dict[str, float]
    news_data: List[Dict[str, Any]]
    prediction_result: Dict[str, Any]
    sentiment_result: Dict[str, Any]
    reflection_result: Dict[str, Any]
    explanation: str
    confidence_level: str
    warnings: List[str]


def create_freshstart_workflow(
    prediction_agent: PredictionAgent,
    sentiment_agent: SentimentAgent,
    reflection_agent: ReflectionAgent,
    explanation_agent: ExplanationAgent
) -> StateGraph:
    """Create the FreshStart analysis workflow.

    Args:
        prediction_agent: Agent for price predictions
        sentiment_agent: Agent for sentiment analysis
        reflection_agent: Agent for quality validation
        explanation_agent: Agent for generating explanations

    Returns:
        Compiled LangGraph workflow
    """
    builder = StateGraph(StockAnalysisState)

    def validate_input(state: StockAnalysisState) -> StockAnalysisState:
        """Validate input and set defaults."""
        if not state.get("ticker"):
            raise ValueError("Ticker is required")

        if not state.get("user_tier"):
            state["user_tier"] = "basic"

        if not state.get("warnings"):
            state["warnings"] = []

        logger.info(f"Starting analysis for {state['ticker']}")
        return state

    def fetch_data(state: StockAnalysisState) -> StockAnalysisState:
        """Fetch market and news data with SQLite caching."""
        ticker = state["ticker"]

        from data.fetchers.price_data import get_fundamentals, get_historical_data

        # Fetch price data (prefer cached when fresh)
        market_data: List[Dict[str, Any]] = []
        cached_recent_prices = _load_cached_market_data(
            ticker,
            min_rows=PRICE_CACHE_MIN_ROWS,
            max_age_days=PRICE_CACHE_TTL_DAYS
        )
        if cached_recent_prices:
            market_data = cached_recent_prices
            logger.info(
                "Using cached price data for %s (%d rows <= %d day old)",
                ticker,
                len(cached_recent_prices),
                PRICE_CACHE_TTL_DAYS,
            )
        else:
            try:
                market_data = get_historical_data(ticker, period="3mo")
                logger.info(f"Fetched {len(market_data)} days of data for {ticker}")
                _cache_market_data(ticker, market_data)
            except Exception as e:
                logger.error(f"Failed to fetch price data: {e}")
                cached_data = _load_cached_market_data(ticker)
                if cached_data:
                    warning = f"Price data API error ({str(e)}); using cached history"
                    logger.warning(warning)
                    state["warnings"].append(warning)
                    market_data = cached_data
                else:
                    state["warnings"].append(f"Data fetch error: {str(e)}")

        state["market_data"] = market_data

        # Fetch fundamental data independently so we still load it when price fetch fails
        fundamentals: Dict[str, float] = {}
        try:
            fundamentals = get_fundamentals(ticker)
            logger.info(f"Fetched {len(fundamentals)} fundamental metrics for {ticker}")
        except Exception as e:
            logger.error(f"Failed to fetch fundamentals: {e}")
            state["warnings"].append(f"Fundamentals fetch error: {str(e)}")

        state["fundamentals"] = fundamentals

        # Fetch news data using Tae's NewsDataFetcher (reuse cache when fresh)
        news_limit = 50
        cached_recent_news = _load_cached_news(
            ticker,
            limit=news_limit,
            max_age_minutes=NEWS_CACHE_TTL_MINUTES
        )
        if cached_recent_news:
            state["news_data"] = cached_recent_news
            logger.info(
                "Using cached news for %s (%d articles <= %d min old)",
                ticker,
                len(cached_recent_news),
                NEWS_CACHE_TTL_MINUTES,
            )
        else:
            try:
                from data.fetchers.news_data import NewsDataFetcher

                news_fetcher = NewsDataFetcher()
                news_articles = news_fetcher.fetch_news(ticker, limit=news_limit)
                state["news_data"] = news_articles

                logger.info(f"Fetched {len(news_articles)} news articles for {ticker}")
                _cache_news_articles(ticker, news_articles)
                _persist_news_embeddings(ticker, news_articles)

            except Exception as e:
                logger.error(f"Failed to fetch news: {e}")
                cached_news = _load_cached_news(ticker, limit=news_limit)
                if cached_news:
                    warning = f"News API error ({str(e)}); using cached articles"
                    logger.warning(warning)
                    state["warnings"].append(warning)
                    state["news_data"] = cached_news
                else:
                    state["warnings"].append(f"News fetch error: {str(e)}")
                    state["news_data"] = []

        return state

    def run_prediction(state: StockAnalysisState) -> StockAnalysisState:
        """Run prediction agent."""
        try:
            result = prediction_agent.run(
                ticker=state["ticker"],
                market_data=state["market_data"],
                fundamentals=state["fundamentals"]
            )

            state["prediction_result"] = {
                "direction": result.direction,
                "confidence": result.confidence,
                "narrative": result.narrative,
                "probabilities": result.probabilities,
                "metadata": result.metadata
            }

            logger.info(f"Prediction: {result.direction} ({result.confidence:.1%})")

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            state["warnings"].append(f"Prediction error: {str(e)}")
            state["prediction_result"] = {
                "direction": "neutral",
                "confidence": 0.5,
                "narrative": "Prediction unavailable",
                "probabilities": {"up": 0.33, "down": 0.33, "neutral": 0.34},
                "metadata": {"error": str(e)}
            }

        return state

    def run_sentiment(state: StockAnalysisState) -> StockAnalysisState:
        """Run sentiment agent."""
        try:
            result = sentiment_agent.run(
                ticker=state["ticker"],
                news_data=state["news_data"]
            )

            state["sentiment_result"] = {
                "current": result.current,
                "score": result.score,
                "trend": result.trend,
                "headlines": result.headlines
            }

            logger.info(f"Sentiment: {result.current} ({result.score:.1%})")

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            state["warnings"].append(f"Sentiment error: {str(e)}")
            state["sentiment_result"] = {
                "current": "neutral",
                "score": 0.5,
                "trend": "stable",
                "headlines": []
            }

        return state

    def run_reflection(state: StockAnalysisState) -> StockAnalysisState:
        """Run reflection agent for quality validation."""
        try:
            result = reflection_agent.run(
                prediction=state["prediction_result"],
                sentiment=state["sentiment_result"],
                filing={},  # Smart money data (placeholder)
                market_data=state["market_data"],
                fundamentals=state["fundamentals"]
            )

            state["reflection_result"] = result

            # Adjust confidence based on validation
            if not result["validation_passed"]:
                state["warnings"].extend(result["issues"])
                state["confidence_level"] = "low"
                logger.warning(f"Validation failed: {len(result['issues'])} issues")
            else:
                # Set confidence level based on prediction confidence
                pred_conf = state["prediction_result"]["confidence"]
                if pred_conf >= 0.75:
                    state["confidence_level"] = "high"
                elif pred_conf >= 0.6:
                    state["confidence_level"] = "medium"
                else:
                    state["confidence_level"] = "low"

        except Exception as e:
            logger.error(f"Reflection failed: {e}")
            state["confidence_level"] = "medium"
            state["reflection_result"] = {"validation_passed": True, "issues": []}

        return state

    def build_explanation(state: StockAnalysisState) -> StockAnalysisState:
        """Generate explanation."""
        try:
            explanation = explanation_agent.run(
                ticker=state["ticker"],
                prediction=state["prediction_result"],
                sentiment=state["sentiment_result"],
                smart_money={},  # Placeholder
                user_tier=state["user_tier"],
                confidence_level=state["confidence_level"]
            )

            state["explanation"] = explanation
            logger.info("Generated explanation")

        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            state["explanation"] = f"Analysis for {state['ticker']} completed with errors."

        return state

    # Add nodes to workflow
    builder.add_node("validate", validate_input)
    builder.add_node("fetch_data", fetch_data)
    builder.add_node("predict", run_prediction)
    builder.add_node("sentiment", run_sentiment)
    builder.add_node("reflect", run_reflection)
    builder.add_node("explain", build_explanation)

    # Define workflow edges
    builder.set_entry_point("validate")
    builder.add_edge("validate", "fetch_data")
    builder.add_edge("fetch_data", "predict")
    builder.add_edge("predict", "sentiment")
    builder.add_edge("sentiment", "reflect")
    builder.add_edge("reflect", "explain")
    builder.set_finish_point("explain")

    return builder


def run_stock_analysis(
    ticker: str,
    user_tier: str = "basic",
    config=None
) -> Dict[str, Any]:
    """Run complete stock analysis workflow.

    Args:
        ticker: Stock symbol
        user_tier: User subscription level
        config: Optional WorkflowConfig instance

    Returns:
        Dict with analysis results
    """
    from coordinator.config import WorkflowConfig

    # Load config
    if config is None:
        config = WorkflowConfig()

    # Create agents
    prediction_model = config.get_prediction_model()
    prediction_agent = PredictionAgent(model=prediction_model)
    sentiment_agent = SentimentAgent()
    reflection_agent = ReflectionAgent()
    explanation_agent = ExplanationAgent()

    # Create and compile workflow
    workflow = create_freshstart_workflow(
        prediction_agent=prediction_agent,
        sentiment_agent=sentiment_agent,
        reflection_agent=reflection_agent,
        explanation_agent=explanation_agent
    ).compile()

    # Run workflow
    result = workflow.invoke({
        "ticker": ticker,
        "user_tier": user_tier
    })

    # Format response
    return {
        "ticker": result["ticker"],
        "as_of": datetime.utcnow().isoformat() + "Z",
        "prediction": result["prediction_result"],
        "sentiment": result["sentiment_result"],
        "explanation": result["explanation"],
        "confidence_level": result["confidence_level"],
        "warnings": result["warnings"],
        "metadata": {
            "user_tier": user_tier,
            "reflection_passed": result.get("reflection_result", {}).get("validation_passed", True)
        }
    }


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)

    try:
        result = run_stock_analysis("AAPL", user_tier="basic")
        print("\n" + "=" * 60)
        print("FreshStart Analysis Result")
        print("=" * 60)
        print(f"Ticker: {result['ticker']}")
        print(f"Prediction: {result['prediction']['direction']} ({result['prediction']['confidence']:.1%})")
        print(f"Sentiment: {result['sentiment']['current']}")
        print(f"Confidence: {result['confidence_level']}")
        print("\nExplanation:")
        print(result['explanation'])
        print("=" * 60)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
