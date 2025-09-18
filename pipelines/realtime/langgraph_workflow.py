"""LangGraph orchestration for the multi-agent forecasting workflow."""
from __future__ import annotations

from typing import Dict, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .agents import ExplanationAgent, PredictionAgent, SentimentAgent


class PredictionState(TypedDict, total=False):
    symbol: str
    market_data: str
    news: str
    prediction: str
    confidence: float
    sentiment: str
    explanation: str
    mode: Literal["full", "trend-only"]


def create_prediction_workflow(
    *,
    prediction_agent: PredictionAgent,
    sentiment_agent: SentimentAgent,
    explanation_agent: ExplanationAgent,
    confidence_threshold: float = 0.95,
) -> StateGraph:
    """Compose the LangGraph workflow for the AI capability."""

    builder = StateGraph(PredictionState)

    def gather_data(state: PredictionState) -> PredictionState:
        if "market_data" not in state:
            raise ValueError("market_data must be provided to the workflow")
        if "news" not in state:
            state["news"] = ""
        return state

    def run_prediction(state: PredictionState) -> PredictionState:
        result = prediction_agent.run(symbol=state["symbol"], market_data=state["market_data"])
        state["prediction"] = result.narrative
        state["confidence"] = result.confidence
        return state

    def run_sentiment(state: PredictionState) -> PredictionState:
        state["sentiment"] = sentiment_agent.run(symbol=state["symbol"], news=state.get("news", ""))
        return state

    def build_explanation(state: PredictionState) -> PredictionState:
        state["mode"] = "full" if state.get("confidence", 0.0) >= confidence_threshold else "trend-only"
        state["explanation"] = explanation_agent.run(
            prediction=state["prediction"],
            sentiment=state["sentiment"],
            mode=state["mode"],
        )
        return state

    builder.add_node("gather", gather_data)
    builder.add_node("predict", run_prediction)
    builder.add_node("sentiment", run_sentiment)
    builder.add_node("explain", build_explanation)

    builder.add_edge(START, "gather")
    builder.add_edge("gather", "predict")
    builder.add_edge("predict", "sentiment")
    builder.add_edge("sentiment", "explain")
    builder.add_edge("explain", END)

    return builder


def run_prediction_workflow(
    *,
    symbol: str,
    market_data: str,
    news: str,
    prediction_agent: PredictionAgent,
    sentiment_agent: SentimentAgent,
    explanation_agent: ExplanationAgent,
    confidence_threshold: float = 0.95,
) -> Dict[str, str]:
    """Convenience helper that executes the LangGraph workflow end-to-end."""

    workflow = create_prediction_workflow(
        prediction_agent=prediction_agent,
        sentiment_agent=sentiment_agent,
        explanation_agent=explanation_agent,
        confidence_threshold=confidence_threshold,
    ).compile()

    result_state = workflow.invoke(
        {
            "symbol": symbol,
            "market_data": market_data,
            "news": news,
        }
    )

    return {
        "prediction": result_state["prediction"],
        "confidence": f"{result_state['confidence']:.0%}",
        "sentiment": result_state["sentiment"],
        "explanation": result_state["explanation"],
        "mode": result_state["mode"],
    }


if __name__ == "__main__":
    from .agents import build_mock_llm

    mock_llm = build_mock_llm("demo")

    prediction_agent = PredictionAgent(mock_llm)
    sentiment_agent = SentimentAgent(mock_llm)
    explanation_agent = ExplanationAgent(mock_llm)

    output = run_prediction_workflow(
        symbol="AAPL",
        market_data="Price oscillating upward with strong volume. Confidence 96%.",
        news="Apple announces new product line with strong pre-orders.",
        prediction_agent=prediction_agent,
        sentiment_agent=sentiment_agent,
        explanation_agent=explanation_agent,
    )

    import json

    print(json.dumps(output, indent=2))
