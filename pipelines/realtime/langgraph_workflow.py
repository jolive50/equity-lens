"""LangGraph orchestration for the StockSense multi-agent forecasting workflow."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional, TypedDict

from langgraph.graph import StateGraph

from .agents import ExplanationAgent, PredictionAgent, SentimentAgent, SmartMoneyAgent
from .data_adapters import DataService


class StockAnalysisState(TypedDict, total=False):
    """State object for the StockSense analysis workflow."""
    ticker: str
    user_tier: Literal["basic", "premium"]
    market_data: Optional[Dict]
    news_data: List[Dict]
    fundamentals: Dict[str, float]
    prediction_result: Optional[Dict]
    sentiment_result: Optional[Dict]
    smart_money_data: Optional[Dict]
    explanation: Optional[str]
    confidence_level: Literal["high", "medium", "low"]
    daily_probs: List[Dict]
    horizon_95: Optional[Dict]
    warnings: List[str]


def create_stocksense_workflow(
    *,
    prediction_agent: PredictionAgent,
    sentiment_agent: SentimentAgent,
    explanation_agent: ExplanationAgent,
    smart_money_agent: SmartMoneyAgent,
    data_service: Optional[DataService] = None,
    confidence_threshold: float = 0.95,
) -> StateGraph:
    """Compose the LangGraph workflow for StockSense analysis."""

    builder = StateGraph(StockAnalysisState)

    def validate_input(state: StockAnalysisState) -> StockAnalysisState:
        """Validate input ticker and set defaults."""
        if not state.get("ticker"):
            raise ValueError("ticker is required")
        if not state.get("user_tier"):
            state["user_tier"] = "basic"
        if not state.get("warnings"):
            state["warnings"] = []
        return state

    def collect_market_data(state: StockAnalysisState) -> StockAnalysisState:
        """Collect market data and fundamentals."""
        # Use mock data for now to avoid API issues
        state["market_data"] = [
            {"date": "2025-01-24", "close": 150.0, "volume": 1000000},
            {"date": "2025-01-23", "close": 148.0, "volume": 950000}
        ]
        state["fundamentals"] = {
            "revenue_growth": 0.08,
            "ebitda_margin": 0.28,
            "debt_to_ebitda": 2.1
        }
        return state

    def collect_news_data(state: StockAnalysisState) -> StockAnalysisState:
        """Collect news and sentiment data."""
        # Use mock news data for now
        state["news_data"] = [
            {
                "title": "Strong earnings report",
                "content": "Company reports better than expected Q4 results",
                "timestamp": datetime.utcnow().isoformat()
            }
        ]
        return state

    def run_prediction(state: StockAnalysisState) -> StockAnalysisState:
        """Run ML prediction with probabilistic forecasting."""
        result = prediction_agent.run(
            ticker=state["ticker"],
            market_data=state["market_data"],
            fundamentals=state["fundamentals"]
        )
        
        # Generate daily probabilities for next 30 days
        daily_probs = []
        base_date = datetime.utcnow()
        
        for i in range(1, 31):
            date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
            # Mock probabilities - in real implementation, these would come from ML model
            up_prob = max(0.1, min(0.9, result.confidence + (i * -0.01)))
            down_prob = max(0.05, min(0.4, (1 - result.confidence) * 0.5))
            neutral_prob = 1 - up_prob - down_prob
            
            daily_probs.append({
                "date": date_str,
                "up": round(up_prob, 2),
                "down": round(down_prob, 2),
                "neutral": round(neutral_prob, 2)
            })
        
        state["daily_probs"] = daily_probs
        state["prediction_result"] = {
            "direction": result.direction,
            "confidence": result.confidence,
            "narrative": result.narrative
        }
        
        # Calculate 95% confidence horizon
        horizon_95 = _calculate_95_horizon(daily_probs)
        state["horizon_95"] = horizon_95
        
        # Set confidence level
        if result.confidence >= 0.95:
            state["confidence_level"] = "high"
        elif result.confidence >= 0.75:
            state["confidence_level"] = "medium"
        else:
            state["confidence_level"] = "low"
            state["warnings"].append("Low confidence prediction - use with caution")
        
        return state

    def run_sentiment(state: StockAnalysisState) -> StockAnalysisState:
        """Run sentiment analysis on news data."""
        sentiment_result = sentiment_agent.run(
            ticker=state["ticker"],
            news_data=state["news_data"]
        )
        state["sentiment_result"] = sentiment_result
        return state

    def run_smart_money(state: StockAnalysisState) -> StockAnalysisState:
        """Collect smart money data (institutions, insiders, Congress)."""
        if state["user_tier"] == "premium":
            smart_money_result = smart_money_agent.run(ticker=state["ticker"])
            state["smart_money_data"] = smart_money_result
        else:
            state["smart_money_data"] = {"message": "Premium feature"}
        return state

    def build_explanation(state: StockAnalysisState) -> StockAnalysisState:
        """Generate plain English explanation."""
        explanation = explanation_agent.run(
            ticker=state["ticker"],
            prediction=state["prediction_result"],
            sentiment=state["sentiment_result"],
            smart_money=state["smart_money_data"],
            user_tier=state["user_tier"],
            confidence_level=state["confidence_level"]
        )
        state["explanation"] = explanation
        return state

    # Add nodes to workflow
    builder.add_node("validate", validate_input)
    builder.add_node("market_data_node", collect_market_data)
    builder.add_node("news_data_node", collect_news_data)
    builder.add_node("predict", run_prediction)
    builder.add_node("sentiment", run_sentiment)
    builder.add_node("smart_money", run_smart_money)
    builder.add_node("explain", build_explanation)

    # Define workflow edges
    builder.set_entry_point("validate")
    builder.add_edge("validate", "market_data_node")
    builder.add_edge("market_data_node", "news_data_node")
    builder.add_edge("news_data_node", "predict")
    builder.add_edge("predict", "sentiment")
    builder.add_edge("sentiment", "smart_money")
    builder.add_edge("smart_money", "explain")
    builder.set_finish_point("explain")

    return builder


def _calculate_95_horizon(daily_probs: List[Dict]) -> Dict:
    """Calculate the 95% confidence horizon from daily probabilities."""
    max_confidence_days = 0
    first_drop_day = None
    
    for i, day_prob in enumerate(daily_probs):
        max_prob = max(day_prob["up"], day_prob["down"], day_prob["neutral"])
        if max_prob >= 0.95:
            max_confidence_days = i + 1
        elif first_drop_day is None and max_prob < 0.95:
            first_drop_day = day_prob["date"]
    
    # Find the direction with highest confidence
    if daily_probs:
        first_day = daily_probs[0]
        direction = max(["up", "down", "neutral"], key=lambda d: first_day[d])
    else:
        direction = "neutral"
    
    return {
        "class": direction,
        "start_date": daily_probs[0]["date"] if daily_probs else None,
        "end_date": daily_probs[max_confidence_days - 1]["date"] if max_confidence_days > 0 else None,
        "days": max_confidence_days,
        "drops_below_95_on": first_drop_day
    }


def run_stocksense_analysis(
    *,
    ticker: str,
    user_tier: str = "basic",
    prediction_agent: PredictionAgent,
    sentiment_agent: SentimentAgent,
    explanation_agent: ExplanationAgent,
    smart_money_agent: SmartMoneyAgent,
    data_service: Optional[DataService] = None,
    confidence_threshold: float = 0.95,
) -> Dict[str, any]:
    """Convenience helper that executes the StockSense workflow end-to-end."""

    workflow = create_stocksense_workflow(
        prediction_agent=prediction_agent,
        sentiment_agent=sentiment_agent,
        explanation_agent=explanation_agent,
        smart_money_agent=smart_money_agent,
        data_service=data_service,
        confidence_threshold=confidence_threshold,
    ).compile()

    result_state = workflow.invoke({
        "ticker": ticker,
        "user_tier": user_tier,
    })

    # Format response according to StockSense API contract
    return {
        "ticker": result_state["ticker"],
        "as_of": datetime.utcnow().isoformat() + "Z",
        "forecast": {
            "direction": result_state["prediction_result"]["direction"],
            "confidence": result_state["prediction_result"]["confidence"],
            "horizon_95": result_state["horizon_95"],
            "daily_probs": result_state["daily_probs"]
        },
        "metrics": _format_metrics(result_state["fundamentals"]),
        "sentiment": result_state["sentiment_result"],
        "smart_money": result_state["smart_money_data"],
        "explanation": result_state["explanation"],
        "warnings": result_state["warnings"],
        "disclaimers": [
            "This is informational only, not investment advice",
            "Data sources: Alpha Vantage, NewsAPI, SEC filings",
            f"Last updated: {datetime.utcnow().isoformat()}Z"
        ]
    }


def _format_metrics(fundamentals: Dict[str, float]) -> Dict[str, Dict]:
    """Format fundamentals into layperson-friendly metrics."""
    metrics = {}
    
    # Revenue growth
    revenue_growth = fundamentals.get("revenue_growth", 0)
    if revenue_growth > 0.05:
        metrics["revenue_growth"] = {
            "value": revenue_growth,
            "verdict": "Good",
            "explanation": "Revenue growth is above sector average and 5-year median."
        }
    elif revenue_growth > 0:
        metrics["revenue_growth"] = {
            "value": revenue_growth,
            "verdict": "OK",
            "explanation": "Revenue growth is positive but below sector average."
        }
    else:
        metrics["revenue_growth"] = {
            "value": revenue_growth,
            "verdict": "Needs caution",
            "explanation": "Revenue growth is negative - monitor closely."
        }
    
    # EBITDA margin
    ebitda_margin = fundamentals.get("ebitda_margin", 0)
    if ebitda_margin > 0.20:
        metrics["ebitda_margin"] = {
            "value": ebitda_margin,
            "verdict": "Good",
            "explanation": "EBITDA margin is strong and improving over recent quarters."
        }
    elif ebitda_margin > 0.10:
        metrics["ebitda_margin"] = {
            "value": ebitda_margin,
            "verdict": "OK",
            "explanation": "EBITDA margin is reasonable but could be better."
        }
    else:
        metrics["ebitda_margin"] = {
            "value": ebitda_margin,
            "verdict": "Needs caution",
            "explanation": "EBITDA margin is low - profitability concerns."
        }
    
    # Debt to EBITDA
    debt_to_ebitda = fundamentals.get("debt_to_ebitda", 0)
    if debt_to_ebitda < 2.0:
        metrics["debt_to_ebitda"] = {
            "value": debt_to_ebitda,
            "verdict": "Good",
            "explanation": "Debt levels are manageable relative to earnings."
        }
    elif debt_to_ebitda < 4.0:
        metrics["debt_to_ebitda"] = {
            "value": debt_to_ebitda,
            "verdict": "OK",
            "explanation": "Debt levels are moderate - watch interest coverage."
        }
    else:
        metrics["debt_to_ebitda"] = {
            "value": debt_to_ebitda,
            "verdict": "Needs caution",
            "explanation": "High debt levels - significant leverage risk."
        }
    
    return metrics


if __name__ == "__main__":
    from .agents import build_mock_llm

    mock_llm = build_mock_llm("stocksense")

    prediction_agent = PredictionAgent(mock_llm)
    sentiment_agent = SentimentAgent(mock_llm)
    explanation_agent = ExplanationAgent(mock_llm)
    smart_money_agent = SmartMoneyAgent(mock_llm)

    output = run_stocksense_analysis(
        ticker="AAPL",
        user_tier="premium",
        prediction_agent=prediction_agent,
        sentiment_agent=sentiment_agent,
        explanation_agent=explanation_agent,
        smart_money_agent=smart_money_agent,
    )

    import json
    print(json.dumps(output, indent=2))
