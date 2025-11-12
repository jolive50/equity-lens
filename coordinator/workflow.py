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
        logger.info("\n┌─────────────────────────────────────────────────────────────────┐")
        logger.info("│ 📝 NODE: validate_input                                        │")
        logger.info("└─────────────────────────────────────────────────────────────────┘")

        if not state.get("ticker"):
            logger.error("❌ Validation failed: Ticker is required")
            raise ValueError("Ticker is required")

        if not state.get("user_tier"):
            state["user_tier"] = "basic"
            logger.info("   ℹ️  User tier not specified, defaulting to 'basic'")

        if not state.get("warnings"):
            state["warnings"] = []

        logger.info(f"   ✅ Input validated: ticker={state['ticker']}, user_tier={state['user_tier']}")
        return state

    def fetch_data(state: StockAnalysisState) -> StockAnalysisState:
        """Fetch market and news data with SQLite caching."""
        import time
        node_start = time.time()

        logger.info("\n┌─────────────────────────────────────────────────────────────────┐")
        logger.info("│ 📊 NODE: fetch_data                                            │")
        logger.info("└─────────────────────────────────────────────────────────────────┘")
        logger.info(f"   📥 INPUT STATE:")
        logger.info(f"      Ticker: {state.get('ticker', 'N/A')}")
        logger.info(f"      User Tier: {state.get('user_tier', 'N/A')}")

        ticker = state["ticker"]

        from data.fetchers.price_data import get_fundamentals, get_historical_data

        # Fetch price data (prefer cached when fresh)
        logger.info("   📈 Fetching price data...")
        price_start = time.time()
        market_data: List[Dict[str, Any]] = []
        cached_recent_prices = _load_cached_market_data(
            ticker,
            min_rows=PRICE_CACHE_MIN_ROWS,
            max_age_days=PRICE_CACHE_TTL_DAYS
        )
        if cached_recent_prices:
            market_data = cached_recent_prices
            logger.info(
                "      ✓ Using cached price data: %d rows (≤%d day old) [%s]",
                len(cached_recent_prices),
                PRICE_CACHE_TTL_DAYS,
                "CACHE HIT"
            )
        else:
            try:
                market_data = get_historical_data(ticker, period="3mo")
                logger.info(f"      ✓ Fetched fresh price data: {len(market_data)} rows [API CALL]")
                _cache_market_data(ticker, market_data)
            except Exception as e:
                logger.error(f"      ✗ Failed to fetch price data: {e}")
                cached_data = _load_cached_market_data(ticker)
                if cached_data:
                    warning = f"Price data API error ({str(e)}); using cached history"
                    logger.warning(f"      ⚠️  {warning}")
                    state["warnings"].append(warning)
                    market_data = cached_data
                else:
                    state["warnings"].append(f"Data fetch error: {str(e)}")
        logger.info(f"      ⏱️  Price data fetch: {time.time() - price_start:.2f}s")

        state["market_data"] = market_data

        # Fetch fundamental data independently so we still load it when price fetch fails
        logger.info("   💼 Fetching fundamentals...")
        fundamentals_start = time.time()
        fundamentals: Dict[str, float] = {}
        try:
            fundamentals = get_fundamentals(ticker)
            logger.info(f"      ✓ Fetched {len(fundamentals)} fundamental metrics")
            if fundamentals:
                logger.info(f"         Metrics: {', '.join(list(fundamentals.keys())[:5])}...")
        except Exception as e:
            logger.error(f"      ✗ Failed to fetch fundamentals: {e}")
            state["warnings"].append(f"Fundamentals fetch error: {str(e)}")
        logger.info(f"      ⏱️  Fundamentals fetch: {time.time() - fundamentals_start:.2f}s")

        state["fundamentals"] = fundamentals

        # Fetch news data using Tae's NewsDataFetcher (reuse cache when fresh)
        logger.info("   📰 Fetching news articles...")
        news_start = time.time()
        news_limit = 50
        cached_recent_news = _load_cached_news(
            ticker,
            limit=news_limit,
            max_age_minutes=NEWS_CACHE_TTL_MINUTES
        )
        if cached_recent_news:
            state["news_data"] = cached_recent_news
            logger.info(
                "      ✓ Using cached news: %d articles (≤%d min old) [CACHE HIT]",
                len(cached_recent_news),
                NEWS_CACHE_TTL_MINUTES,
            )
        else:
            try:
                from data.fetchers.news_data import NewsDataFetcher

                news_fetcher = NewsDataFetcher()
                news_articles = news_fetcher.fetch_news(ticker, limit=news_limit)
                state["news_data"] = news_articles

                logger.info(f"      ✓ Fetched fresh news: {len(news_articles)} articles [API CALL]")
                _cache_news_articles(ticker, news_articles)
                _persist_news_embeddings(ticker, news_articles)

            except Exception as e:
                logger.error(f"      ✗ Failed to fetch news: {e}")
                cached_news = _load_cached_news(ticker, limit=news_limit)
                if cached_news:
                    warning = f"News API error ({str(e)}); using cached articles"
                    logger.warning(f"      ⚠️  {warning}")
                    state["warnings"].append(warning)
                    state["news_data"] = cached_news
                else:
                    state["warnings"].append(f"News fetch error: {str(e)}")
                    state["news_data"] = []
        logger.info(f"      ⏱️  News fetch: {time.time() - news_start:.2f}s")

        logger.info(f"   ✅ Data fetch complete ({time.time() - node_start:.2f}s total)")
        logger.info(f"      Summary: {len(market_data)} price rows, {len(fundamentals)} metrics, {len(state.get('news_data', []))} articles")

        logger.info(f"\n   📤 OUTPUT STATE:")
        logger.info(f"      Market data: {len(market_data)} rows")
        logger.info(f"      Fundamentals: {len(fundamentals)} metrics")
        logger.info(f"      News data: {len(state.get('news_data', []))} articles")
        logger.info(f"      Warnings: {len(state.get('warnings', []))}")

        return state

    def run_prediction(state: StockAnalysisState) -> StockAnalysisState:
        """Run prediction agent."""
        import time
        node_start = time.time()

        logger.info("\n┌─────────────────────────────────────────────────────────────────┐")
        logger.info("│ 🔮 NODE: run_prediction (PredictionAgent)                     │")
        logger.info("└─────────────────────────────────────────────────────────────────┘")

        try:
            logger.info("   📥 INPUT STATE:")
            logger.info(f"      Ticker: {state.get('ticker', 'N/A')}")
            logger.info(f"      Market data: {len(state['market_data'])} rows")
            logger.info(f"      Fundamentals: {len(state['fundamentals'])} metrics")

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

            logger.info(f"   ✅ Prediction complete ({time.time() - node_start:.2f}s)")
            logger.info(f"      Model: {result.metadata.get('model', 'Unknown')}")
            logger.info(f"      Direction: {result.direction.upper()}")
            logger.info(f"      Confidence: {result.confidence:.1%}")
            logger.info(f"      Probabilities: ↑{result.probabilities['up']:.1%} ↓{result.probabilities['down']:.1%} →{result.probabilities.get('neutral', 0):.1%}")

            logger.info(f"\n   📤 OUTPUT STATE:")
            logger.info(f"      Prediction stored in state['prediction_result']")
            logger.info(f"      Direction: {result.direction}")
            logger.info(f"      Confidence: {result.confidence:.1%}")

        except Exception as e:
            logger.error(f"   ❌ Prediction failed ({time.time() - node_start:.2f}s): {e}")
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
        import time
        node_start = time.time()

        logger.info("\n┌─────────────────────────────────────────────────────────────────┐")
        logger.info("│ 💬 NODE: run_sentiment (SentimentAgent)                       │")
        logger.info("└─────────────────────────────────────────────────────────────────┘")

        try:
            logger.info("   📥 INPUT STATE:")
            logger.info(f"      Ticker: {state.get('ticker', 'N/A')}")
            logger.info(f"      News articles: {len(state['news_data'])}")
            logger.info(f"      Prediction direction: {state.get('prediction_result', {}).get('direction', 'N/A')}")

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

            logger.info(f"   ✅ Sentiment analysis complete ({time.time() - node_start:.2f}s)")
            if sentiment_agent.sentiment_model:
                model_info = sentiment_agent.sentiment_model.get_model_info()
                logger.info(f"      Model: {model_info['name']}")
            logger.info(f"      Sentiment: {result.current.upper()}")
            logger.info(f"      Score: {result.score:.2f} (0=negative, 0.5=neutral, 1=positive)")
            logger.info(f"      Trend: {result.trend}")
            logger.info(f"      Headlines analyzed: {len(result.headlines)}")

            logger.info(f"\n   📤 OUTPUT STATE:")
            logger.info(f"      Sentiment stored in state['sentiment_result']")
            logger.info(f"      Current sentiment: {result.current}")
            logger.info(f"      Score: {result.score:.2f}")
            logger.info(f"      Trend: {result.trend}")

        except Exception as e:
            logger.error(f"   ❌ Sentiment analysis failed ({time.time() - node_start:.2f}s): {e}")
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
        import time
        node_start = time.time()

        logger.info("\n┌─────────────────────────────────────────────────────────────────┐")
        logger.info("│ 🔍 NODE: run_reflection (ReflectionAgent)                     │")
        logger.info("└─────────────────────────────────────────────────────────────────┘")

        try:
            logger.info("   📥 INPUT STATE:")
            logger.info(f"      Prediction: {state.get('prediction_result', {}).get('direction', 'N/A')} ({state.get('prediction_result', {}).get('confidence', 0):.1%})")
            logger.info(f"      Sentiment: {state.get('sentiment_result', {}).get('current', 'N/A')} ({state.get('sentiment_result', {}).get('score', 0):.2f})")
            logger.info(f"   🔎 Validating analysis quality...")

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
                logger.warning(f"   ⚠️  Validation failed: {len(result['issues'])} issues")
                for idx, issue in enumerate(result['issues'], 1):
                    logger.warning(f"      {idx}. {issue}")
            else:
                # Set confidence level based on prediction confidence
                pred_conf = state["prediction_result"]["confidence"]
                if pred_conf >= 0.75:
                    state["confidence_level"] = "high"
                elif pred_conf >= 0.6:
                    state["confidence_level"] = "medium"
                else:
                    state["confidence_level"] = "low"
                logger.info(f"   ✅ Validation passed")

            logger.info(f"   ✅ Reflection complete ({time.time() - node_start:.2f}s)")
            logger.info(f"      Confidence level: {state['confidence_level'].upper()}")

            logger.info(f"\n   📤 OUTPUT STATE:")
            logger.info(f"      Reflection result stored in state['reflection_result']")
            logger.info(f"      Validation passed: {result['validation_passed']}")
            logger.info(f"      Confidence level: {state['confidence_level']}")
            logger.info(f"      Total warnings: {len(state['warnings'])}")

        except Exception as e:
            logger.error(f"   ❌ Reflection failed ({time.time() - node_start:.2f}s): {e}")
            state["confidence_level"] = "medium"
            state["reflection_result"] = {"validation_passed": True, "issues": []}

        return state

    def build_explanation(state: StockAnalysisState) -> StockAnalysisState:
        """Generate explanation with ChromaDB and SQLite context."""
        import time
        node_start = time.time()

        logger.info("\n┌─────────────────────────────────────────────────────────────────┐")
        logger.info("│ 📝 NODE: build_explanation (ExplanationAgent)                 │")
        logger.info("└─────────────────────────────────────────────────────────────────┘")

        try:
            logger.info("   📥 INPUT STATE:")
            logger.info(f"      Ticker: {state.get('ticker', 'N/A')}")
            logger.info(f"      User tier: {state['user_tier']}")
            logger.info(f"      Confidence level: {state['confidence_level']}")
            logger.info(f"      Prediction: {state.get('prediction_result', {}).get('direction', 'N/A')}")
            logger.info(f"      Sentiment: {state.get('sentiment_result', {}).get('current', 'N/A')}")

            ticker = state["ticker"]

            # Fetch ChromaDB data (similar news and statistics)
            logger.info("   🔍 Fetching ChromaDB context...")
            similar_news = []
            news_statistics = None
            vector_store = _get_vector_store()
            if vector_store:
                try:
                    # Search for similar news based on current sentiment
                    sentiment_current = state["sentiment_result"].get("current", "neutral")
                    search_query = f"{ticker} stock {sentiment_current} news"
                    similar_news = vector_store.search_similar_news(
                        query=search_query,
                        ticker=ticker,
                        n_results=5
                    )
                    logger.info(f"      ✓ Found {len(similar_news)} similar news articles")

                    # Get news statistics for ticker
                    news_statistics = vector_store.get_ticker_statistics(ticker)
                    logger.info(f"      ✓ Retrieved news statistics: {news_statistics.get('total_articles', 0)} total articles")
                except Exception as e:
                    logger.warning(f"      ⚠️  ChromaDB query failed: {e}")

            # Fetch SQLite data (historical analyses)
            logger.info("   💾 Fetching SQLite historical analyses...")
            historical_analyses = []
            db = _get_db_client()
            if db:
                try:
                    historical_analyses = db.get_analysis_history(ticker.upper(), limit=5)
                    logger.info(f"      ✓ Found {len(historical_analyses)} historical analyses")
                except Exception as e:
                    logger.warning(f"      ⚠️  SQLite query failed: {e}")

            logger.info("   📄 Generating explanation with historical context...")

            explanation = explanation_agent.run(
                ticker=state["ticker"],
                prediction=state["prediction_result"],
                sentiment=state["sentiment_result"],
                smart_money={},  # Placeholder
                user_tier=state["user_tier"],
                confidence_level=state["confidence_level"],
                similar_news=similar_news,
                historical_analyses=historical_analyses,
                news_statistics=news_statistics
            )

            state["explanation"] = explanation
            logger.info(f"   ✅ Explanation generated ({time.time() - node_start:.2f}s)")
            logger.info(f"      Length: {len(explanation)} characters")
            logger.info(f"      Context used: {len(similar_news)} similar news, {len(historical_analyses)} historical analyses")

            logger.info(f"\n   📤 OUTPUT STATE (FINAL):")
            logger.info(f"      Explanation stored in state['explanation']")
            logger.info(f"      Complete analysis ready for response")
            logger.info(f"      Fields populated: ticker, prediction, sentiment, explanation, confidence_level, warnings")

        except Exception as e:
            logger.error(f"   ❌ Explanation generation failed ({time.time() - node_start:.2f}s): {e}")
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
    import time
    workflow_start = time.time()

    logger.info("\n" + "╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "🔷 WORKFLOW: Initializing" + " " * 28 + "║")
    logger.info("╚" + "═" * 78 + "╝")

    from coordinator.config import WorkflowConfig

    # Load config
    if config is None:
        # Try to load from config.yaml first, fall back to defaults
        import os
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        if os.path.exists(config_path):
            config = WorkflowConfig.from_yaml(config_path)
            logger.info(f"📄 Loaded configuration from {config_path}")
        else:
            config = WorkflowConfig()
            logger.info(f"📄 Using default configuration (config.yaml not found)")

    logger.info(f"\n📋 WORKFLOW: Configuration loaded")
    logger.info(f"   Prediction models: {config.prediction_models}")
    logger.info(f"   Use prediction ensemble: {config.use_ensemble}")
    logger.info(f"   Prediction strategy: {config.prediction_ensemble_strategy}")
    logger.info(f"   Sentiment models: {config.sentiment_models}")
    logger.info(f"   Use sentiment ensemble: {config.use_sentiment_ensemble}")
    logger.info(f"   Sentiment strategy: {config.sentiment_ensemble_strategy}")
    logger.info(f"   Reflection enabled: {config.reflection_enabled}")

    # Create agents
    logger.info(f"\n🤖 WORKFLOW: Creating agents...")

    logger.info("   1️⃣  Creating PredictionAgent...")
    agent_start = time.time()
    prediction_model = config.get_prediction_model()
    prediction_agent = PredictionAgent(model=prediction_model)
    logger.info(f"      ✓ PredictionAgent ready ({time.time() - agent_start:.2f}s)")
    logger.info(f"      Model: {prediction_model.__class__.__name__}")

    logger.info("   2️⃣  Creating SentimentAgent...")
    agent_start = time.time()
    sentiment_model = config.get_sentiment_model()
    sentiment_agent = SentimentAgent(sentiment_model=sentiment_model)
    logger.info(f"      ✓ SentimentAgent ready ({time.time() - agent_start:.2f}s)")
    if sentiment_agent.sentiment_model:
        model_info = sentiment_agent.sentiment_model.get_model_info()
        if model_info.get('type') == 'ensemble':
            logger.info(f"      Model: {model_info['name']} (strategy: {model_info['strategy']})")
            logger.info(f"      Ensemble contains: {', '.join(model_info['models'])}")
        else:
            logger.info(f"      Model: {model_info['name']} (v{model_info.get('version', 'unknown')})")

    logger.info("   3️⃣  Creating ReflectionAgent...")
    agent_start = time.time()
    reflection_agent = ReflectionAgent()
    logger.info(f"      ✓ ReflectionAgent ready ({time.time() - agent_start:.2f}s)")

    logger.info("   4️⃣  Creating ExplanationAgent...")
    agent_start = time.time()
    explanation_agent = ExplanationAgent()
    logger.info(f"      ✓ ExplanationAgent ready ({time.time() - agent_start:.2f}s)")

    # Create and compile workflow
    logger.info(f"\n🔧 WORKFLOW: Building LangGraph workflow...")
    compile_start = time.time()
    workflow = create_freshstart_workflow(
        prediction_agent=prediction_agent,
        sentiment_agent=sentiment_agent,
        reflection_agent=reflection_agent,
        explanation_agent=explanation_agent
    ).compile()
    logger.info(f"   ✓ Workflow compiled ({time.time() - compile_start:.2f}s)")

    # Run workflow
    logger.info("\n" + "╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 15 + f"🚀 WORKFLOW: Executing for {ticker}" + " " * (48 - len(ticker)) + "║")
    logger.info("╚" + "═" * 78 + "╝")

    invoke_start = time.time()
    result = workflow.invoke({
        "ticker": ticker,
        "user_tier": user_tier
    })
    invoke_time = time.time() - invoke_start

    logger.info("\n" + "╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 18 + "✅ WORKFLOW: Execution Complete" + " " * 25 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info(f"⏱️  Workflow execution: {invoke_time:.2f}s")
    logger.info(f"⏱️  Total workflow time: {time.time() - workflow_start:.2f}s")

    # Save analysis to database
    db = _get_db_client()
    if db:
        try:
            analysis_data = {
                "prediction_direction": result["prediction_result"]["direction"],
                "prediction_confidence": result["prediction_result"]["confidence"],
                "prediction_model": result["prediction_result"].get("model", "unknown"),
                "sentiment_score": result["sentiment_result"]["score"],
                "sentiment_label": result["sentiment_result"]["current"],
                "sentiment_model": result["sentiment_result"].get("model", "unknown"),
                "explanation": result["explanation"],
                "reflection_warnings": "; ".join(result["warnings"]),
                "raw_data": result
            }
            db.save_analysis(ticker.upper(), analysis_data)
            logger.info(f"Saved analysis result for {ticker}")
        except Exception as exc:
            logger.warning(f"Failed to save analysis for {ticker}: {exc}")

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
