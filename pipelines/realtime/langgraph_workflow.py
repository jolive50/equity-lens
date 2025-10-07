"""LangGraph orchestration for the StockSense multi-agent forecasting workflow."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, List, Literal, Optional, TypedDict, Any

from langgraph.graph import StateGraph

from .agents import (
    CoordinationAgent, HistoricalAnalysisAgent, SentimentAnalysisAgent, 
    ExplanationAgent, PredictionAgent, SentimentAgent, SmartMoneyAgent
)
from .data_adapters import DataService
from .sp500_data_service import get_sp500_data_service
from .api_keys import get_api_key  # Centralised API key lookup

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
        # What: Make sure callers supplied at least one ticker to analyze
        # Why: Every downstream agent requires a symbol; failing early keeps errors readable
        # How: Check both `ticker` and bulk `tickers` entries and raise if both are missing
        # Data: Reads the incoming state dict; no external calls or side effects yet
        if not state.get("ticker") and not state.get("tickers"):
            raise ValueError("Either ticker or tickers list is required")
        
        # What: Fill in the legacy `ticker` field when only the new list format is provided
        # Why: Several parts of the workflow still expect a primary ticker for backwards compatibility
        # How: Copy the first symbol from the `tickers` list into the single-ticker slot
        # Data: Mutates the state dict so both keys reference the same base symbol
        if not state.get("ticker"):
            # If no single ticker, use first ticker for backward compatibility
            if state.get("tickers"):
                state["ticker"] = state["tickers"][0]
            else:
                raise ValueError("No ticker information provided")
        
        # What: Set sensible defaults when the caller leaves optional settings blank
        # Why: Guarantees consistent downstream behavior between basic and premium tiers
        # How: Populate `user_tier` and `warnings` with baseline values when absent
        # Data: Only touches local state dict; no network or disk I/O
        if not state.get("user_tier"):
            state["user_tier"] = "basic"
        if not state.get("warnings"):
            state["warnings"] = []
        
        # What: Normalize single-ticker requests into the list structure the rest of the graph consumes
        # Why: Simplifies later loops that assume `tickers` always exists
        # How: Wrap the primary ticker in a one-element list when the list was omitted
        # Data: Ensures state["tickers"] is always a list, even for single analyses
        # Default to single ticker analysis if no tickers list
        if state.get("ticker") and not state.get("tickers"):
            state["tickers"] = [state["ticker"]]
        
        return state

    def collect_sp500_data(state: StockAnalysisState) -> StockAnalysisState:
        """Collect historical data and fundamentals for S&P 500 companies."""
        tickers = state["tickers"]
        
        # What: Pull pre-aggregated price and fundamentals for each requested S&P 500 company
        # Why: Feeding the agents a consistent dataset avoids duplicate API calls and speeds up analysis
        # How: Ask the shared SP500 data service for a combined payload covering every ticker in the list
        # Data: Outputs a nested dict keyed by ticker with `market_data` and `fundamentals` sections
        # Use S&P 500 data service
        sp500_service = get_sp500_data_service()
        comprehensive_data = sp500_service.get_comprehensive_sp500_data(tickers)
        
        # What: Store ticker-by-ticker price histories inside the workflow state
        # Why: Historical, sentiment, and prediction agents all expect quick access to past prices
        # How: Build a dictionary comprehension that maps each ticker to its downloaded market dataset
        # Data: Each value is whatever structure the data service returned for `market_data`
        state["comprehensive_market_data"] = {
            ticker: comprehensive_data[ticker]["market_data"] 
            for ticker in tickers
        }
        # What: Preserve fundamentals alongside prices for every ticker in the batch
        # Why: Prediction and explanation agents reference valuation metrics while reasoning
        # How: Mirror the market-data comprehension but point to the fundamentals slice
        # Data: Each value contains ratios and balance-sheet metrics harvested earlier
        state["comprehensive_fundamentals"] = {
            ticker: comprehensive_data[ticker]["fundamentals"] 
            for ticker in tickers
        }
        
        # What: Keep backwards compatibility for parts of the app that only know about a single ticker
        # Why: The FastAPI endpoints and some UI components still expect `market_data` / `fundamentals` roots
        # How: Copy the primary ticker's info into the legacy slots while leaving the new structures intact
        # Data: The chosen primary ticker is whatever `validate_input` set earlier
        # Set primary ticker data for backward compatibility
        primary_ticker = state["ticker"]
        state["market_data"] = comprehensive_data[primary_ticker]["market_data"]
        state["fundamentals"] = comprehensive_data[primary_ticker]["fundamentals"]
        
        return state

    def collect_comprehensive_news(state: StockAnalysisState) -> StockAnalysisState:
        """Collect news for all tickers."""
        tickers = state["tickers"]
        
        # What: Gather recent headlines for each company under review
        # Why: Sentiment agent and explanation pipeline rely on up-to-date article context
        # How: Loop through the ticker list and ask the shared service for per-symbol news arrays
        # Data: Builds a dict keyed by ticker with lists of news article dictionaries inside
        # Use S&P 500 service for comprehensive news
        sp500_service = get_sp500_data_service()
        comprehensive_news = {}
        for ticker in tickers:
            comprehensive_news[ticker] = sp500_service.get_news_data(ticker)
        
        # What: Cache the multi-ticker news bundle on the state object for later steps
        # Why: Avoids repeating the service call and keeps intermediate results inspectable
        # How: Assign the assembled dictionary to `comprehensive_news_data`
        # Data: Downstream agents read from this structure when creating sentiment rollups
        state["comprehensive_news_data"] = comprehensive_news
        
        # What: Maintain the legacy single-ticker news field for older parts of the workflow
        # Why: Some agents and API payloads still expect `news_data` to exist even when analyzing many symbols
        # How: Copy the primary ticker's list into the classic slot without altering the bulk version
        # Data: Shares references to the same list objects, so edits stay in sync
        # Set primary ticker news for backward compatibility
        primary_ticker = state["ticker"]
        state["news_data"] = comprehensive_news[primary_ticker]
        
        return state

    def run_historical_working_agent(state: StockAnalysisState) -> StockAnalysisState:
        """Run historical analysis working agent."""
        # What: Ask the historical agent to crunch long-term market behavior
        # Why: The coordinating agent needs technical context before making recommendations
        # How: Bundle each ticker's market data and fundamentals into the shape the agent expects
        # Data: Passes a nested dict with per-ticker datasets plus a single-ticker shortcut
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
        # What: Initialize the shared results bucket if this is the first agent to report back
        # Why: Later agents append into the same structure, so we need a dict ready to merge into
        # How: Create the dictionary only when missing to preserve any prior state
        # Data: Mutates `state["working_agent_results"]` in place
        if not state.get("working_agent_results"):
            state["working_agent_results"] = {}
        
        # What: Cache the historical agent's full response for downstream coordination
        # Why: The coordination and explanation steps both re-use the raw narrative and scores
        # How: Store under a descriptive key so other functions can look it up easily
        # Data: The result payload typically includes technical summaries and confidence metrics
        state["working_agent_results"]["historical"] = historical_result
        
        return state

    def run_sentiment_working_agent(state: StockAnalysisState) -> StockAnalysisState:
        """Run sentiment analysis working agent."""
        # What: Feed the sentiment agent the latest headlines plus market context
        # Why: Combining qualitative news with quantitative prices yields richer sentiment scores
        # How: Supply both the multi-ticker news bundle and the primary ticker's market fundamentals
        # Data: Sends dictionaries full of article metadata and price metrics into the agent
        sentiment_result = sentiment_agent.run(
            comprehensive_news_data=state["comprehensive_news_data"],
            market_context={
                "market_data": state.get("market_data", {}),
                "fundamentals": state.get("fundamentals", {})
            }
        )
        
        # Store working agent results
        # What: Ensure the shared results dict exists before we append sentiment outputs
        # Why: Previous steps may not have populated it if sentiment runs first in a custom flow
        # How: Create an empty dictionary the first time we access the container
        # Data: Keeps accumulating agent responses under distinct keys
        if not state["working_agent_results"]:
            state["working_agent_results"] = {}
        
        # What: Persist both the headline-level analysis and confidence scores
        # Why: The coordinator weighs confidence when blending agent opinions
        # How: Copy the agent's dictionary into structured slots for easy reuse
        # Data: Includes top stories, sentiment trend, and numeric confidence values
        state["working_agent_results"]["sentiment"] = sentiment_result
        state["working_agent_results"]["sentiment_confidence"] = sentiment_result["sentiment_confidence"]
        state["working_agent_results"]["historical_confidence"] = state["working_agent_results"]["historical"]["historical_confidence"]
        
        return state

    def run_coordination_agent(state: StockAnalysisState) -> StockAnalysisState:
        """Run coordinating agent to synthesize working agent results."""
        # What: Invite the coordinator agent to blend historical and sentiment insights
        # Why: We want a single narrative and confidence score that balances multiple perspectives
        # How: Pass along every agent's raw output plus the ticker list so the coordinator can compare symbols
        # Data: Sends the aggregated `working_agent_results` dict and expects a combined analysis back
        coordination_result = coordination_agent.coordinate(
            tickers=state["tickers"],
            working_agent_results=state["working_agent_results"]
        )
        
        # What: Capture the coordinator's long-form reasoning and numeric confidence
        # Why: Later steps display the summary to users and use the score to gate recommendations
        # How: Store the text report under `coordination_summary` and the float under `confidence_score`
        # Data: Confidence typically ranges 0-1 while summary is a paragraph string
        state["coordination_summary"] = coordination_result.get("raw_analysis", "")
        state["confidence_score"] = coordination_result.get("confidence", 0.0)
        
        # Check confidence threshold
        threshold_met = coordination_result.get("threshold_met", False)
        
        if threshold_met and state["confidence_score"] >= confidence_threshold:
            # What: Flag the analysis as highly reliable for the rest of the pipeline
            # Why: Premium users only see recommendations when the system is nearly certain
            # How: Mark the qualitative level and log the achievement for observability
            # Data: Logs the numeric score while storing the "high" label on the state
            state["confidence_level"] = "high"
            logger.info(f"Analysis passed confidence threshold: {state['confidence_score']:.3f}")
        elif state["confidence_score"] >= 0.75:
            # What: Treat mid-range confidence as cautionary feedback instead of a green light
            # Why: Users still deserve context, but we want to warn them before acting
            # How: Set the level to medium and append plain-language warnings for the UI to display
            # Data: Warnings list accumulates human-readable strings about confidence gaps
            state["confidence_level"] = "medium"
            state["warnings"].append(f"Medium confidence ({state['confidence_score']:.3f}) - verify before acting")
            state["warnings"].append("Below 95% confidence threshold - recommendation withheld")
        else:
            # What: Label low-confidence runs so the UI can instruct users to hold off
            # Why: Prevents accidental trust in weak signals by surfacing red-flag messages
            # How: Assign the "low" level and add explicit warnings about insufficient evidence
            # Data: Warnings highlight the numeric score to keep messaging transparent
            state["confidence_level"] = "low"
            state["warnings"].append(f"Low confidence ({state['confidence_score']:.3f}) - insufficient for recommendation")
            state["warnings"].append("Below 95% confidence threshold - recommendation withheld")
        
        return state

    def run_prediction(state: StockAnalysisState) -> StockAnalysisState:
        """Run ML prediction with probabilistic forecasting."""
        # What: Ask the prediction agent for a probabilistic price-direction forecast
        # Why: This is the headline insight users care about—will the stock likely rise or fall?
        # How: Provide the primary ticker's market history and fundamentals as input features
        # Data: Sends dictionaries of price candles and valuation metrics; receives a ForecastResult object
        result = prediction_agent.run(
            ticker=state["ticker"],
            market_data=state["market_data"],
            fundamentals=state["fundamentals"]
        )
        
        # What: Persist the model's day-by-day probability curve for the next 30 trading days
        # Why: The frontend charts and alerting logic rely on real probability outputs from the forecaster
        # How: Reuse the `daily_probs` list returned by the ML model without altering the values
        # Data: Each entry already contains ISO dates and up/down/neutral probabilities generated by the model
        if result.daily_probs:
            state["daily_probs"] = result.daily_probs
        else:
            # What: Provide an empty curve when the ML forecaster could not supply probabilities
            # Why: Keeps the API contract intact while being honest about data availability
            # How: Store an empty list and surface a warning for the client
            state["daily_probs"] = []
            state["warnings"].append(
                "Daily probability curve unavailable because the ML forecaster was not able to produce a forecast."
            )
        # What: Store the core prediction summary for easy access
        # Why: Multiple agents and the UI reference direction, confidence, and narrative text
        # How: Copy the forecaster output into a plain dictionary
        # Data: Includes movement direction, numeric confidence, and an English explanation
        state["prediction_result"] = {
            "direction": result.direction,
            "confidence": result.confidence,
            "narrative": result.narrative
        }
        
        # What: Mirror the model's own 95% confidence horizon computation
        # Why: Keeps the API response aligned with the logic already baked into the forecaster
        # How: Copy the structured horizon information from the ForecastResult object
        # Data: Includes duration, direction class, start/end dates, and the first drop below 95%
        if result.horizon_95:
            state["horizon_95"] = result.horizon_95
        else:
            state["horizon_95"] = {
                "class": result.direction,
                "days": 0,
                "start_date": None,
                "end_date": None,
                "drops_below_95_on": None,
                "available": False
            }
        
        # Set confidence level
        # What: Translate numeric confidence into the friendly labels other components expect
        # Why: UI and alerts rely on "high/medium/low" buckets to decide messaging tone
        # How: Compare the confidence score against predetermined breakpoints and add warnings when low
        # Data: Updates `confidence_level` string and appends caution messages if needed
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
        # What: Produce a single-ticker sentiment snapshot for backwards-compatible consumers
        # Why: Older components expect this lightweight result even though multi-ticker sentiment exists
        # How: Reuse the sentiment agent with the legacy call signature (ticker + news list)
        # Data: Sends a ticker string and list of article dicts; receives a sentiment summary dict
        sentiment_result = sentiment_agent.run(
            ticker=state["ticker"],
            news_data=state["news_data"]
        )
        state["sentiment_result"] = sentiment_result
        return state

    def run_smart_money(state: StockAnalysisState) -> StockAnalysisState:
        """Collect smart money data (institutions, insiders, Congress)."""
        if state["user_tier"] == "premium":
            # What: Fetch premium-only institutional trading insights
            # Why: Paying users expect deeper visibility into hedge fund, insider, and congressional moves
            # How: Call the dedicated agent which aggregates filings and disclosures into a digestible format
            # Data: Returns nested dicts for each smart-money category keyed by activity type
            smart_money_result = smart_money_agent.run(ticker=state["ticker"])
            state["smart_money_data"] = smart_money_result
        else:
            # What: Provide a clear message when non-premium users hit this branch
            # Why: Helps the frontend show upgrade prompts instead of a blank panel
            # How: Store a simple dictionary with an explanatory string
            # Data: Keeps the state shape consistent while signalling feature gating
            state["smart_money_data"] = {"message": "Premium feature"}
        return state

    def build_explanation(state: StockAnalysisState) -> StockAnalysisState:
        """Generate plain English explanation."""
        # What: Craft a narrative that stitches together predictions, news, and smart-money signals
        # Why: Users need a digestible story that explains the numbers in everyday language
        # How: Call the explanation agent with all intermediate results plus the user's tier and confidence level
        # Data: Feeds in multiple dictionaries from prior steps; receives a string summary
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
    # What: Wire up the LangGraph state machine so tasks run in the intended order
    # Why: The graph controls execution flow, including parallel legs and final assembly
    # How: Set the entry node, then connect transitions that reflect data dependencies
    # Data: Each edge references node names defined above; no external resources touched here
    builder.set_entry_point("validate")
    builder.add_edge("validate", "sp500_data")
    builder.add_edge("sp500_data", "comprehensive_news")
    
    # What: Trigger both historical and sentiment working agents at the same time
    # Why: Parallelizing independent tasks shortens response time for users
    # How: Add separate edges from the news collection step to each working agent node
    # Data: Each branch receives the same state object and writes back its own results
    builder.add_edge("comprehensive_news", "historical_working")
    builder.add_edge("comprehensive_news", "sentiment_working")
    
    # What: Funnel both working-agent outputs into the coordinator for synthesis
    # Why: The coordinator needs perspectives from every specialist before drafting guidance
    # How: Connect each working node to the coordination node so both transitions feed the same step
    # Data: The state now includes historical and sentiment results when the coordinator runs
    builder.add_edge("historical_working", "coordination")
    builder.add_edge("sentiment_working", "coordination")
    
    # What: Continue the legacy single-ticker pipeline once the new coordination step finishes
    # Why: Maintains backwards compatibility while still benefiting from the enhanced front half
    # How: Chain prediction, sentiment, smart-money, and explanation nodes linearly
    # Data: State evolves with prediction results, sentiment summaries, premium data, then narratives
    builder.add_edge("coordination", "legacy_predict")
    builder.add_edge("legacy_predict", "legacy_sentiment")
    builder.add_edge("legacy_sentiment", "smart_money")
    builder.add_edge("smart_money", "explain")
    
    builder.set_finish_point("explain")

    return builder


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

    # What: Instantiate the LangGraph workflow configured with all enhanced agents
    # Why: The compiled graph orchestrates data collection, analysis, and coordination automatically
    # How: Call `create_stocksense_workflow` with the injected agent instances, then compile it
    # Data: Produces a runnable graph object that we can invoke with initial state
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

    # What: Kick off the workflow for the requested set of tickers
    # Why: Running the graph yields a rich state object containing every agent's output
    # How: Pass the initial state (tickers and user tier) into the compiled workflow's invoke method
    # Data: Returns a dictionary with comprehensive market, sentiment, and coordination results
    result_state = workflow.invoke({
        "tickers": tickers,
        "user_tier": user_tier,
    })

    # Enhanced response format with coordination insights
    # What: Shape the final payload so the API can respond with human-friendly fields
    # Why: Consumers expect standardized keys like `predictions`, `sentiment`, and disclaimers
    # How: Pull values out of the workflow state and assemble them into a top-level dictionary
    # Data: Includes timestamps, agent outputs, warnings, and compliance messaging
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

    # What: Build the workflow for a single ticker using the provided agents
    # Why: Keeps the API handler lightweight by hiding graph wiring logic in one helper
    # How: Call `create_stocksense_workflow` with the relevant agents and compile it
    # Data: Produces a runnable object that adheres to the same state contract as the enhanced version
    workflow = create_stocksense_workflow(
        prediction_agent=prediction_agent,
        sentiment_agent=sentiment_agent,
        explanation_agent=explanation_agent,
        smart_money_agent=smart_money_agent,
        data_service=data_service,
        confidence_threshold=confidence_threshold,
    ).compile()

    # What: Execute the workflow starting from the initial ticker and user tier
    # Why: Running the state machine gathers market data, sentiment, predictions, and explanations
    # How: Invoke the compiled graph with the starting state dictionary
    # Data: Returns the final state containing every intermediate result
    result_state = workflow.invoke({
        "ticker": ticker,
        "user_tier": user_tier,
    })

    # Format response according to StockSense API contract
    # What: Map the workflow state into the response shape the FastAPI endpoint expects
    # Why: Ensures clients receive fields like `forecast`, `metrics`, and `disclaimers`
    # How: Pull the relevant pieces from `result_state` and bundle them into a readable dictionary
    # Data: Combines prediction outputs, sentiment summaries, smart-money data, and warnings
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
    # What: Convert raw financial ratios into simple traffic-light verdicts
    # Why: Students and end users grasp "Good/OK/Needs caution" faster than raw percentages
    # How: Compare each metric against intuitive thresholds and attach explanations
    # Data: Reads `fundamentals` dict and returns a new dict keyed by metric name
    metrics = {}
    
    # What: Summarize revenue growth into a verdict
    # Why: Consistent top-line growth signals healthy demand and execution
    # How: Check growth against positive thresholds and assign a friendly explanation
    # Data: Stores the original value plus verdict text inside metrics["revenue_growth"]
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
    
    # What: Grade profitability using EBITDA margin
    # Why: High margins show the company can turn revenue into operating profit efficiently
    # How: Compare margin to healthy (>20%) and acceptable (>10%) ranges, then craft the explanation
    # Data: Writes the verdict bundle under metrics["ebitda_margin"]
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
    
    # What: Evaluate leverage using the Debt/EBITDA ratio
    # Why: Lower ratios mean the company can pay off debt with earnings more comfortably
    # How: Apply standard credit thresholds (<2 good, <4 acceptable) and warn when leverage is high
    # Data: Saves the annotated result in metrics["debt_to_ebitda"]
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
    # What: Allow the module to run as a standalone demo script
    # Why: Gives students a quick way to see the workflow output without wiring up FastAPI
    # How: Spin up LLM clients, instantiate agents, and execute the enhanced analysis pipeline
    # Data: Prints a JSON blob with predictions, sentiment, and coordination summaries
    try:
        # What: Prefer the high-accuracy GPT-4 backend when an API key is available
        # Why: Produces richer narratives and more reliable coordination summaries during demos
        # How: Fetch the OpenAI API key via the central helper and configure the ChatOpenAI client
        # Data: Sends prompts to OpenAI's API only when the key is present
        from langchain_openai import ChatOpenAI

        api_key = get_api_key("openai")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required to run the demo workflow")

        real_llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            max_tokens=2000,
            api_key=api_key
        )
        print("Using real GPT-4 API")
            
    except Exception as e:
        # What: Notify the developer that the preferred GPT-4 client could not be created
        # Why: Visibility into credential or network issues makes debugging smoother
        # How: Print the exception message directly to the console
        # Data: Includes the Python exception string only; no secret values are logged
        print(f"Could not initialize real LLM: {e}")
        # What: Fall back to a lighter GPT-3.5 model so the demo still runs
        # Why: Keeps the script usable when premium models are unavailable or cost-prohibitive
        # How: Instantiate ChatOpenAI with the 3.5 turbo model and modest generation settings
        # Data: Future prompt traffic will target the GPT-3.5 endpoint instead of GPT-4
        from langchain_openai import ChatOpenAI
        real_llm = ChatOpenAI(
            model="gpt-3.5-turbo",  # Use cheaper model as fallback
            temperature=0.1,
            max_tokens=1000,
            api_key=get_api_key("openai", required=True)
        )
    
    from .agents import (
        build_real_llm_agent, CoordinationAgent, HistoricalAnalysisAgent, 
        SentimentAnalysisAgent, PredictionAgent, SentimentAgent, 
        ExplanationAgent, SmartMoneyAgent
    )

    llm_wrapper = build_real_llm_agent("stocksense", real_llm)

    # Create new coordinating and working agents with real LLM
    # What: Instantiate the coordination stack that powers the enhanced workflow
    # Why: Each agent specializes in a portion of the analysis and needs the shared LLM wrapper
    # How: Pass the wrapper into each class constructor with any required configuration
    # Data: Agents reuse the same LLM session for consistent tone and cost control
    coordination_agent = CoordinationAgent(llm_wrapper, confidence_threshold=0.95)
    historical_agent = HistoricalAnalysisAgent(llm_wrapper)
    sentiment_agent = SentimentAnalysisAgent(llm_wrapper)
    
    # Create legacy agents for compatibility
    # What: Build the traditional single-agent components that certain endpoints still call
    # Why: Ensures backwards-compatible paths continue to function during manual tests
    # How: Reuse the same wrapper so fallbacks and narratives align with the coordinating agents
    # Data: These agents will rely on the LLM or internal models depending on configuration
    prediction_agent = PredictionAgent(llm_wrapper)
    legacy_sentiment_agent = SentimentAgent(llm_wrapper)
    explanation_agent = ExplanationAgent(llm_wrapper)
    smart_money_agent = SmartMoneyAgent(llm_wrapper)

    # Test enhanced workflow with multiple S&P 500 companies
    # What: Execute the multi-ticker workflow to showcase aggregated output
    # Why: Demonstrates how coordination combines several agent perspectives for a portfolio view
    # How: Call `run_enhanced_stocksense_analysis` with a list of well-known tickers
    # Data: Returns a comprehensive dictionary which we then pretty-print as JSON
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
