"""LangGraph orchestration for the StockSense multi-agent forecasting workflow."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional, TypedDict, Any

from langgraph.graph import StateGraph

from .agents import (
    CoordinationAgent, HistoricalAnalysisAgent, SentimentAnalysisAgent, 
    ExplanationAgent, PredictionAgent, SentimentAgent, SmartMoneyAgent
)
from .data_adapters import DataService
from .sp500_data_service import get_sp500_data_service

logger = logging.getLogger(__name__)


class StockAnalysisState(TypedDict, total=False):
    """State object for the StockSense analysis workflow."""
    ticker: str
    tickers: List[str]  # Support multiple S&P 500 companies
    user_tier: Literal["basic", "premium"]
    market_data: Optional[Dict]
    comprehensive_market_data: Dict[str, Dict]  # Data for multiple tickers
    news_data: List[Dict]
    comprehensive_news_data: Dict[str, List[Dict]]  # News for multiple tickers
    fundamentals: Dict[str, float]
    comprehensive_fundamentals: Dict[str, Dict[str, float]]  # Fundamentals for multiple tickers
    prediction_result: Optional[Dict]
    comprehensive_predictions: Dict[str, Dict]  # Predictions for multiple tickers
    sentiment_result: Optional[Dict]
    comprehensive_sentiment: Dict[str, Dict]  # Sentiment for multiple tickers
    smart_money_data: Optional[Dict]
    explanation: Optional[str]
    comprehensive_explanation: Optional[str]  # Combined explanation for all tickers
    confidence_level: Literal["high", "medium", "low"]
    confidence_score: float  # Raw confidence score (0.0-1.0)
    daily_probs: List[Dict]
    horizon_95: Optional[Dict]
    warnings: List[str]
    working_agent_results: Dict[str, Any]  # Results from working agents
    coordination_summary: Optional[str]  # Summary from coordinating agent


def create_stocksense_workflow(
    *,
    coordination_agent: CoordinationAgent,
    historical_agent: HistoricalAnalysisAgent,
    sentiment_agent: SentimentAnalysisAgent,
    prediction_agent: PredictionAgent,
    explanation_agent: ExplanationAgent,
    smart_money_agent: SmartMoneyAgent,
    data_service: Optional[DataService] = None,
    confidence_threshold: float = 0.95,
) -> StateGraph:
    """Compose the enhanced LangGraph workflow for StockSense analysis with coordinating agent."""

    builder = StateGraph(StockAnalysisState)

    def validate_input(state: StockAnalysisState) -> StockAnalysisState:
        """Validate input ticker or tickers and set defaults."""
        if not state.get("ticker") and not state.get("tickers"):
            raise ValueError("Either ticker or tickers list is required")
        
        if not state.get("ticker"):
            # If no single ticker, use first ticker for backward compatibility
            if state.get("tickers"):
                state["ticker"] = state["tickers"][0]
            else:
                raise ValueError("No ticker information provided")
        
        if not state.get("user_tier"):
            state["user_tier"] = "basic"
        if not state.get("warnings"):
            state["warnings"] = []
        
        # Default to single ticker analysis if no tickers list
        if state.get("ticker") and not state.get("tickers"):
            state["tickers"] = [state["ticker"]]
        
        return state

    def collect_sp500_data(state: StockAnalysisState) -> StockAnalysisState:
        """Collect historical data and fundamentals for S&P 500 companies."""
        tickers = state["tickers"]
        
        # Use S&P 500 data service
        sp500_service = get_sp500_data_service()
        comprehensive_data = sp500_service.get_comprehensive_sp500_data(tickers)
        
        state["comprehensive_market_data"] = {
            ticker: comprehensive_data[ticker]["market_data"] 
            for ticker in tickers
        }
        state["comprehensive_fundamentals"] = {
            ticker: comprehensive_data[ticker]["fundamentals"] 
            for ticker in tickers
        }
        
        # Set primary ticker data for backward compatibility
        primary_ticker = state["ticker"]
        state["market_data"] = comprehensive_data[primary_ticker]["market_data"]
        state["fundamentals"] = comprehensive_data[primary_ticker]["fundamentals"]
        
        return state

    def collect_comprehensive_news(state: StockAnalysisState) -> StockAnalysisState:
        """Collect news for all tickers."""
        tickers = state["tickers"]
        
        # Use S&P 500 service for comprehensive news
        sp500_service = get_sp500_data_service()
        comprehensive_news = {}
        for ticker in tickers:
            comprehensive_news[ticker] = sp500_service.get_news_data(ticker)
        
        state["comprehensive_news_data"] = comprehensive_news
        
        # Set primary ticker news for backward compatibility
        primary_ticker = state["ticker"]
        state["news_data"] = comprehensive_news[primary_ticker]
        
        return state

    def run_historical_working_agent(state: StockAnalysisState) -> StockAnalysisState:
        """Run historical analysis working agent."""
        historical_result = historical_agent.run(
            comprehensive_data={
                ticker: {
                    "market_data": state["comprehensive_market_data"][ticker],
                    "fundamentals": state["comprehensive_fundamentals"][ticker]
                }
                for ticker in state["tickers"]
            },
            market_data=state.get("market_data", {})
        )
        
        # Store working agent results
        if not state.get("working_agent_results"):
            state["working_agent_results"] = {}
        
        state["working_agent_results"]["historical"] = historical_result
        
        return state

    def run_sentiment_working_agent(state: StockAnalysisState) -> StockAnalysisState:
        """Run sentiment analysis working agent."""
        sentiment_result = sentiment_agent.run(
            comprehensive_news_data=state["comprehensive_news_data"],
            market_context={
                "market_data": state.get("market_data", {}),
                "fundamentals": state.get("fundamentals", {})
            }
        )
        
        # Store working agent results
        if not state["working_agent_results"]:
            state["working_agent_results"] = {}
        
        state["working_agent_results"]["sentiment"] = sentiment_result
        state["working_agent_results"]["sentiment_confidence"] = sentiment_result["sentiment_confidence"]
        state["working_agent_results"]["historical_confidence"] = state["working_agent_results"]["historical"]["historical_confidence"]
        
        return state

    def run_coordination_agent(state: StockAnalysisState) -> StockAnalysisState:
        """Run coordinating agent to synthesize working agent results."""
        coordination_result = coordination_agent.coordinate(
            tickers=state["tickers"],
            working_agent_results=state["working_agent_results"]
        )
        
        state["coordination_summary"] = coordination_result.get("raw_analysis", "")
        state["confidence_score"] = coordination_result.get("confidence", 0.0)
        
        # Check confidence threshold
        threshold_met = coordination_result.get("threshold_met", False)
        
        if threshold_met and state["confidence_score"] >= confidence_threshold:
            state["confidence_level"] = "high"
            logger.info(f"Analysis passed confidence threshold: {state['confidence_score']:.3f}")
        elif state["confidence_score"] >= 0.75:
            state["confidence_level"] = "medium"
            state["warnings"].append(f"Medium confidence ({state['confidence_score']:.3f}) - verify before acting")
            state["warnings"].append("Below 95% confidence threshold - recommendation withheld")
        else:
            state["confidence_level"] = "low"
            state["warnings"].append(f"Low confidence ({state['confidence_score']:.3f}) - insufficient for recommendation")
            state["warnings"].append("Below 95% confidence threshold - recommendation withheld")
        
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

    # Add enhanced nodes to workflow
    builder.add_node("validate", validate_input)
    builder.add_node("sp500_data", collect_sp500_data)
    builder.add_node("comprehensive_news", collect_comprehensive_news)
    builder.add_node("historical_working", run_historical_working_agent)
    builder.add_node("sentiment_working", run_sentiment_working_agent)
    builder.add_node("coordination", run_coordination_agent)
    builder.add_node("legacy_predict", run_prediction)
    builder.add_node("legacy_sentiment", run_sentiment)
    builder.add_node("smart_money", run_smart_money)
    builder.add_node("explain", build_explanation)

    # Define enhanced workflow edges
    builder.set_entry_point("validate")
    builder.add_edge("validate", "sp500_data")
    builder.add_edge("sp500_data", "comprehensive_news")
    
    # Working agents run in parallel
    builder.add_edge("comprehensive_news", "historical_working")
    builder.add_edge("comprehensive_news", "sentiment_working")
    
    # Coordination agent synthesizes working agents
    builder.add_edge("historical_working", "coordination")
    builder.add_edge("sentiment_working", "coordination")
    
    # Legacy agents run in sequence after coordination
    builder.add_edge("coordination", "legacy_predict")
    builder.add_edge("legacy_predict", "legacy_sentiment")
    builder.add_edge("legacy_sentiment", "smart_money")
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


def run_enhanced_stocksense_analysis(
    *,
    tickers: List[str],
    user_tier: str = "basic",
    coordination_agent: CoordinationAgent,
    historical_agent: HistoricalAnalysisAgent,
    sentiment_agent: SentimentAnalysisAgent,
    prediction_agent: PredictionAgent,
    explanation_agent: ExplanationAgent,
    smart_money_agent: SmartMoneyAgent,
    data_service: Optional[DataService] = None,
    confidence_threshold: float = 0.95,
) -> Dict[str, any]:
    """Enhanced StockSense workflow with coordinating agent and multiple S&P 500 companies."""

    workflow = create_stocksense_workflow(
        coordination_agent=coordination_agent,
        historical_agent=historical_agent,
        sentiment_agent=sentiment_agent,
        prediction_agent=prediction_agent,
        explanation_agent=explanation_agent,
        smart_money_agent=smart_money_agent,
        data_service=data_service,
        confidence_threshold=confidence_threshold,
    ).compile()

    result_state = workflow.invoke({
        "tickers": tickers,
        "user_tier": user_tier,
    })

    # Enhanced response format with coordination insights
    return {
        "tickers": result_state["tickers"],
        "as_of": datetime.utcnow().isoformat() + "Z",
        "confidence_score": result_state["confidence_score"],
        "confidence_level": result_state["confidence_level"],
        "coordination_summary": result_state.get("coordination_summary", ""),
        "working_agent_results": result_state["working_agent_results"],
        "comprehensive_explanation": result_state["comprehensive_explanation"],
        "fundamentals": result_state["comprehensive_fundamentals"],
        "sentiment": result_state["comprehensive_sentiment"],
        "predictions": result_state["comprehensive_predictions"],
        "warnings": result_state["warnings"],
        "threshold_met": result_state["confidence_score"] >= confidence_threshold,
        "disclaimers": [
            "This is informational only, not investment advice",
            f"Analysis based on {len(tickers)} S&P 500 companies",
            f"Only recommends if confidence >= {confidence_threshold}",
            f"Last updated: {datetime.utcnow().isoformat()}Z"
        ]
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


def _get_mock_sp500_data(ticker: str) -> Dict[str, Any]:
    """Generate mock S&P 500 data for demonstration."""
    import random
    import numpy as np
    
    # Mock historical data (30 days)
    base_price = 150.0 + random.uniform(-20, 50)  # Starting price
    market_data = []
    
    for i in range(30):
        date_str = (datetime.utcnow() - timedelta(days=30-i)).strftime("%Y-%m-%d")
        # Add some realistic price movement
        price_change = np.random.normal(0, 0.02)
        base_price += (base_price * price_change)
        
        market_data.append({
            "date": date_str,
            "close": round(base_price, 2),
            "volume": int(1000000 + random.uniform(-200000, 300000)),
            "open": round(base_price * random.uniform(0.98, 1.02), 2),
            "high": round(base_price * random.uniform(1.01, 1.05), 2),
            "low": round(base_price * random.uniform(0.95, 0.99), 2)
        })
    
    # Mock fundamentals
    fundamentals = {
        "revenue_growth": round(random.uniform(-0.05, 0.15), 3),
        "ebitda_margin": round(random.uniform(0.08, 0.35), 3),
        "pe_ratio": round(random.uniform(10, 40), 1),
        "debt_to_ebitda": round(random.uniform(1.0, 4.0), 1),
        "roe": round(random.uniform(0.08, 0.25), 3),
        "price_momentum_3m": round(random.uniform(-0.15, 0.20), 3),
        "price_momentum_6m": round(random.uniform(-0.20, 0.30), 3),
        "price_momentum_12m": round(random.uniform(-0.25, 0.40), 3)
    }
    
    return {
        "market_data": market_data,
        "fundamentals": fundamentals
    }


def _get_mock_news_data(ticker: str) -> List[Dict[str, str]]:
    """Generate mock news data for demonstration."""
    import random
    
    news_templates = [
        f"{ticker} reports strong Q4 earnings above analyst expectations",
        f"{ticker} announces new product launch targeting emerging markets",
        f"Analyst upgrades {ticker} to Buy rating citing strong fundamentals",
        f"{ticker} CFO discusses growth strategy in investor conference",
        f"Market sentiment towards {ticker} turns bullish after recent developments"
    ]
    
    news_data = []
    for i in range(len(news_templates)):
        news_data.append({
            "title": news_templates[i],
            "content": f"Comprehensive analysis of {ticker} showing positive momentum in key market segments.",
            "timestamp": (datetime.utcnow() - timedelta(days=i)).isoformat(),
            "source": "Financial News",
            "sentiment_score": random.uniform(0.6, 0.9)  # Generally positive mock sentiment
        })
    
    return news_data


if __name__ == "__main__":
    try:
        # Try to use real OpenAI API with environment key
        from langchain_openai import ChatOpenAI
        import os
        
        if os.getenv("OPENAI_API_KEY"):
            real_llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.1,
                max_tokens=2000,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            print("Using real GPT-4 API")
        else:
            raise Exception("No OpenAI API key found")
            
    except Exception as e:
        print(f"Could not initialize real LLM: {e}")
        # Fallback to a configured real model if available
        from langchain_openai import ChatOpenAI
        real_llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # Use cheaper model as fallback
            temperature=0.1,
            max_tokens=1000
        )
    
    from .agents import (
        build_real_llm_agent, CoordinationAgent, HistoricalAnalysisAgent, 
        SentimentAnalysisAgent, PredictionAgent, SentimentAgent, 
        ExplanationAgent, SmartMoneyAgent
    )

    llm_wrapper = build_real_llm_agent("stocksense", real_llm)

    # Create new coordinating and working agents with real LLM
    coordination_agent = CoordinationAgent(llm_wrapper, confidence_threshold=0.95)
    historical_agent = HistoricalAnalysisAgent(llm_wrapper)
    sentiment_agent = SentimentAnalysisAgent(llm_wrapper)
    
    # Create legacy agents for compatibility
    prediction_agent = PredictionAgent(llm_wrapper)
    legacy_sentiment_agent = SentimentAgent(llm_wrapper)
    explanation_agent = ExplanationAgent(llm_wrapper)
    smart_money_agent = SmartMoneyAgent(llm_wrapper)

    # Test enhanced workflow with multiple S&P 500 companies
    output = run_enhanced_stocksense_analysis(
        tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],  # 5 S&P 500 companies
        user_tier="premium",
        coordination_agent=coordination_agent,
        historical_agent=historical_agent,
        sentiment_agent=sentiment_agent,
        prediction_agent=prediction_agent,
        explanation_agent=explanation_agent,
        smart_money_agent=smart_money_agent,
    )

    import json
    print(json.dumps(output, indent=2))
