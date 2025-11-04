"""LangGraph workflow orchestration for StockSense analysis."""

import logging
from typing import Dict, List, Literal, Optional, TypedDict, Any
from langgraph.graph import StateGraph

from agents import (
    PredictionAgent,
    SentimentAgent,
    ReflectionAgent,
    ExplanationAgent
)


logger = logging.getLogger(__name__)


class StockAnalysisState(TypedDict, total=False):
    """State object for StockSense analysis workflow (simplified for single-ticker MVP).

    Attributes:
        ticker: Stock symbol
        user_tier: User subscription level (basic/premium)
        market_data: Historical price/volume data
        news_data: News articles
        fundamentals: Company financial metrics
        prediction_result: Prediction agent output
        sentiment_result: Sentiment agent output
        reflection_result: Reflection agent output
        explanation: Final explanation text
        confidence_level: Overall confidence (high/medium/low)
        daily_probs: Daily probability curve
        horizon_95: 95% confidence horizon
        warnings: List of warning messages
    """
    ticker: str
    user_tier: Literal["basic", "premium"]
    market_data: Optional[Dict]
    news_data: List[Dict]
    fundamentals: Dict[str, float]
    prediction_result: Optional[Dict]
    sentiment_result: Optional[Dict]
    reflection_result: Optional[Dict]
    explanation: Optional[str]
    confidence_level: Literal["high", "medium", "low"]
    daily_probs: List[Dict]
    horizon_95: Optional[Dict]
    warnings: List[str]


def create_stocksense_workflow(
    prediction_agent: PredictionAgent,
    sentiment_agent: SentimentAgent,
    reflection_agent: ReflectionAgent,
    explanation_agent: ExplanationAgent
) -> StateGraph:
    """Create simplified LangGraph workflow for single-ticker analysis.

    Args:
        prediction_agent: Agent for ML predictions
        sentiment_agent: Agent for news sentiment
        reflection_agent: Agent for quality validation
        explanation_agent: Agent for plain English explanations

    Returns:
        StateGraph builder (call .compile() to get runnable workflow)
    """
    builder = StateGraph(StockAnalysisState)

    def validate_input(state: StockAnalysisState) -> StockAnalysisState:
        """Validate input ticker and set defaults."""
        if not state.get("ticker"):
            raise ValueError("ticker is required")

        if not state.get("user_tier"):
            state["user_tier"] = "basic"

        if not state.get("warnings"):
            state["warnings"] = []

        logger.info(f"Validating analysis for {state['ticker']}")
        return state

    def collect_data(state: StockAnalysisState) -> StockAnalysisState:
        """Collect market data, news, and fundamentals.

        Note: In MVP, data fetching happens before workflow invocation.
        This node validates data presence.
        """
        ticker = state["ticker"]

        if not state.get("market_data"):
            raise ValueError(f"market_data required for {ticker}")

        if not state.get("news_data"):
            state["warnings"].append("No news data available")
            state["news_data"] = []

        if not state.get("fundamentals"):
            raise ValueError(f"fundamentals required for {ticker}")

        logger.info(
            f"Data collected for {ticker}: "
            f"{len(state.get('market_data', []))} market records, "
            f"{len(state.get('news_data', []))} news articles"
        )
        return state

    def run_prediction(state: StockAnalysisState) -> StockAnalysisState:
        """Run ML prediction using LSTM forecaster."""
        ticker = state["ticker"]
        market_data = state["market_data"]
        fundamentals = state["fundamentals"]

        logger.info(f"Running prediction for {ticker}")

        try:
            prediction = prediction_agent.run(
                ticker=ticker,
                market_data=market_data,
                fundamentals=fundamentals
            )

            state["prediction_result"] = {
                "direction": prediction.direction,
                "confidence": prediction.confidence,
                "narrative": prediction.narrative
            }

            if prediction.daily_probs:
                state["daily_probs"] = prediction.daily_probs

            if prediction.horizon_95:
                state["horizon_95"] = prediction.horizon_95

            logger.info(
                f"Prediction for {ticker}: {prediction.direction} "
                f"({prediction.confidence:.2%} confidence)"
            )

        except Exception as e:
            logger.error(f"Prediction failed for {ticker}: {e}")
            state["prediction_result"] = {
                "direction": "neutral",
                "confidence": 0.0,
                "narrative": f"Prediction failed: {str(e)}"
            }
            state["warnings"].append(f"Prediction error: {str(e)}")

        return state

    def run_sentiment(state: StockAnalysisState) -> StockAnalysisState:
        """Run sentiment analysis using FinBERT."""
        ticker = state["ticker"]
        news_data = state["news_data"]

        if not news_data:
            logger.warning(f"No news data for {ticker}, skipping sentiment")
            state["sentiment_result"] = {
                "current": "neutral",
                "score": 0.5,
                "trend": "stable",
                "headlines": []
            }
            state["warnings"].append("No news data for sentiment analysis")
            return state

        logger.info(f"Running sentiment for {ticker} with {len(news_data)} articles")

        try:
            sentiment = sentiment_agent.run(
                ticker=ticker,
                news_data=news_data
            )

            state["sentiment_result"] = {
                "current": sentiment.current,
                "score": sentiment.score,
                "trend": sentiment.trend,
                "headlines": sentiment.headlines
            }

            logger.info(
                f"Sentiment for {ticker}: {sentiment.current} "
                f"({sentiment.score:.2%} score)"
            )

        except Exception as e:
            logger.error(f"Sentiment analysis failed for {ticker}: {e}")
            state["sentiment_result"] = {
                "current": "neutral",
                "score": 0.5,
                "trend": "stable",
                "headlines": []
            }
            state["warnings"].append(f"Sentiment error: {str(e)}")

        return state

    def run_reflection(state: StockAnalysisState) -> StockAnalysisState:
        """Run quality validation with ReflectionAgent."""
        ticker = state["ticker"]
        prediction = state.get("prediction_result", {})
        sentiment = state.get("sentiment_result", {})
        market_data = state.get("market_data")
        fundamentals = state.get("fundamentals", {})

        logger.info(f"Running reflection validation for {ticker}")

        try:
            reflection = reflection_agent.run(
                prediction=prediction,
                sentiment=sentiment,
                filing={},  # SmartMoney not in MVP
                market_data=market_data,
                fundamentals=fundamentals
            )

            state["reflection_result"] = reflection

            # Apply confidence adjustment
            if "confidence_adjustment" in reflection and "prediction_result" in state:
                original_confidence = state["prediction_result"]["confidence"]
                adjusted_confidence = max(
                    0.0,
                    min(1.0, original_confidence + reflection["confidence_adjustment"])
                )
                state["prediction_result"]["confidence"] = adjusted_confidence

                logger.info(
                    f"Confidence adjusted: {original_confidence:.2%} → "
                    f"{adjusted_confidence:.2%}"
                )

            # Add reflection warnings to state
            if reflection.get("issues"):
                state["warnings"].extend(reflection["issues"])

            # Determine overall confidence level
            final_confidence = state["prediction_result"]["confidence"]
            if final_confidence >= 0.8:
                state["confidence_level"] = "high"
            elif final_confidence >= 0.6:
                state["confidence_level"] = "medium"
            else:
                state["confidence_level"] = "low"

        except Exception as e:
            logger.error(f"Reflection failed for {ticker}: {e}")
            state["reflection_result"] = {
                "validation_passed": False,
                "confidence_adjustment": 0.0,
                "issues": [f"Reflection error: {str(e)}"],
                "recommendations": []
            }
            state["warnings"].append(f"Reflection error: {str(e)}")

        return state

    def build_explanation(state: StockAnalysisState) -> StockAnalysisState:
        """Generate plain English explanation."""
        ticker = state["ticker"]
        prediction = state.get("prediction_result", {})
        sentiment = state.get("sentiment_result", {})
        user_tier = state.get("user_tier", "basic")
        confidence_level = state.get("confidence_level", "low")

        logger.info(f"Building explanation for {ticker}")

        try:
            explanation = explanation_agent.run(
                ticker=ticker,
                prediction=prediction,
                sentiment=sentiment,
                smart_money={},  # SmartMoney not in MVP
                user_tier=user_tier,
                confidence_level=confidence_level
            )

            state["explanation"] = explanation

            logger.info(f"Explanation generated for {ticker} ({len(explanation)} chars)")

        except Exception as e:
            logger.error(f"Explanation generation failed for {ticker}: {e}")
            state["explanation"] = (
                f"Analysis for {ticker.upper()}: {prediction.get('direction', 'neutral').upper()} "
                f"with {prediction.get('confidence', 0.0):.0%} confidence. "
                f"Explanation generation encountered an error."
            )
            state["warnings"].append(f"Explanation error: {str(e)}")

        return state

    # Build workflow graph
    builder.add_node("validate", validate_input)
    builder.add_node("fetch_data", collect_data)
    builder.add_node("predict", run_prediction)
    builder.add_node("sentiment", run_sentiment)
    builder.add_node("reflect", run_reflection)
    builder.add_node("explain", build_explanation)

    # Define linear pipeline
    builder.set_entry_point("validate")
    builder.add_edge("validate", "fetch_data")
    builder.add_edge("fetch_data", "predict")
    builder.add_edge("predict", "sentiment")
    builder.add_edge("sentiment", "reflect")
    builder.add_edge("reflect", "explain")
    builder.set_finish_point("explain")

    return builder


def run_stocksense_analysis(
    *,
    ticker: str,
    user_tier: Literal["basic", "premium"],
    market_data: List[Dict],
    news_data: List[Dict],
    fundamentals: Dict[str, float],
    prediction_agent: PredictionAgent,
    sentiment_agent: SentimentAgent,
    reflection_agent: ReflectionAgent,
    explanation_agent: ExplanationAgent
) -> Dict[str, Any]:
    """Run complete StockSense analysis workflow.

    Args:
        ticker: Stock symbol
        user_tier: User subscription level
        market_data: Historical price/volume data
        news_data: News articles
        fundamentals: Company financial metrics
        prediction_agent: Agent for predictions
        sentiment_agent: Agent for sentiment
        reflection_agent: Agent for validation
        explanation_agent: Agent for explanations

    Returns:
        Dictionary with analysis results:
            - ticker: Stock symbol
            - direction: Predicted direction (up/down/neutral)
            - confidence: Confidence score (0.0-1.0)
            - confidence_level: Confidence level (high/medium/low)
            - narrative: Prediction narrative
            - sentiment: Sentiment results
            - explanation: Plain English explanation
            - warnings: List of warning messages
            - daily_probs: Optional daily probability curve
            - horizon_95: Optional 95% confidence horizon
    """
    # Create workflow
    workflow = create_stocksense_workflow(
        prediction_agent=prediction_agent,
        sentiment_agent=sentiment_agent,
        reflection_agent=reflection_agent,
        explanation_agent=explanation_agent
    ).compile()

    # Initialize state
    initial_state = {
        "ticker": ticker,
        "user_tier": user_tier,
        "market_data": market_data,
        "news_data": news_data,
        "fundamentals": fundamentals,
        "warnings": []
    }

    # Run workflow
    logger.info(f"Starting StockSense analysis for {ticker}")
    result_state = workflow.invoke(initial_state)

    # Format response
    prediction = result_state.get("prediction_result", {})
    sentiment = result_state.get("sentiment_result", {})

    response = {
        "ticker": ticker,
        "direction": prediction.get("direction", "neutral"),
        "confidence": prediction.get("confidence", 0.0),
        "confidence_level": result_state.get("confidence_level", "low"),
        "narrative": prediction.get("narrative", ""),
        "sentiment": sentiment,
        "explanation": result_state.get("explanation", ""),
        "warnings": result_state.get("warnings", [])
    }

    # Add optional fields if present
    if "daily_probs" in result_state:
        response["daily_probs"] = result_state["daily_probs"]

    if "horizon_95" in result_state:
        response["horizon_95"] = result_state["horizon_95"]

    logger.info(f"StockSense analysis complete for {ticker}: {response['direction']}")

    return response
