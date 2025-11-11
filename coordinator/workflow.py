"""LangGraph workflow for FreshStart stock analysis.

JOSH's Component - Workflow Orchestration
Simplified LangGraph workflow connecting all agents.
"""
import logging
from typing import Dict, Any, TypedDict, List
from datetime import datetime
from langgraph.graph import StateGraph

from agents.prediction_agent import PredictionAgent
from agents.sentiment_agent import SentimentAgent
from agents.reflection_agent import ReflectionAgent
from agents.explanation_agent import ExplanationAgent

logger = logging.getLogger("freshstart.coordinator")


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
        from storage.database import Database
        from storage.vector_store import VectorStore

        db = Database()
        logger.info("SQLite caching enabled")

        # Fetch price data (check cache first)
        try:
            from data.fetchers.price_data import get_historical_data, get_fundamentals
            import pandas as pd

            # Check if cached price data is fresh
            if db.is_price_cache_fresh(ticker, max_age_hours=1):
                cached_df = db.get_cached_prices(ticker)
                if not cached_df.empty:
                    # Convert DataFrame to list of dicts for workflow
                    market_data = [
                        {
                            'date': str(idx),
                            'Open': row['open'],
                            'High': row['high'],
                            'Low': row['low'],
                            'Close': row['close'],
                            'Volume': row['volume']
                        }
                        for idx, row in cached_df.iterrows()
                    ]
                    state["market_data"] = market_data
                    logger.info(f"Using cached price data for {ticker} ({len(market_data)} days)")
                else:
                    raise ValueError("Empty cached data")
            else:
                # Cache miss or stale - fetch fresh data
                market_data = get_historical_data(ticker, period="3mo")
                state["market_data"] = market_data
                logger.info(f"Fetched {len(market_data)} days of data for {ticker}")

                # Cache the fetched data
                if market_data:
                    df = pd.DataFrame(market_data)
                    if 'date' in df.columns:
                        df.set_index('date', inplace=True)
                    db.cache_prices(ticker, df)

            # Always fetch fresh fundamentals (they change less frequently)
            fundamentals = get_fundamentals(ticker)
            state["fundamentals"] = fundamentals
            logger.info(f"Fetched {len(fundamentals)} fundamental metrics for {ticker}")

        except Exception as e:
            logger.error(f"Failed to fetch price data: {e}")
            state["warnings"].append(f"Data fetch error: {str(e)}")
            state["market_data"] = []
            state["fundamentals"] = {}

        # Fetch news data with caching
        try:
            from data.fetchers.news_data import NewsDataFetcher

            # Check if cached news is fresh
            if db.is_news_cache_fresh(ticker, max_age_hours=1):
                cached_news = db.get_cached_news(ticker, limit=50)
                if cached_news:
                    state["news_data"] = cached_news
                    logger.info(f"Using cached news for {ticker} ({len(cached_news)} articles)")
                else:
                    raise ValueError("Empty cached news")
            else:
                # Cache miss or stale - fetch fresh news
                news_fetcher = NewsDataFetcher()
                news_articles = news_fetcher.fetch_news(ticker, limit=50)
                state["news_data"] = news_articles
                logger.info(f"Fetched {len(news_articles)} news articles for {ticker}")

                # Cache the fetched news
                if news_articles:
                    db.cache_news(ticker, news_articles)

            # Index news in ChromaDB vector store
            try:
                vector_store = VectorStore()
                logger.info("ChromaDB vector store enabled")
                for article in state["news_data"]:
                    vector_store.add_document(
                        ticker=ticker,
                        doc_id=article.get('url', ''),
                        text=f"{article.get('title', '')} {article.get('summary', '')}",
                        metadata=article
                    )
                logger.info(f"Indexed {len(state['news_data'])} news articles for {ticker}")
            except Exception as e:
                logger.warning(f"ChromaDB indexing failed (non-critical): {e}")

        except Exception as e:
            logger.error(f"Failed to fetch news: {e}")
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
