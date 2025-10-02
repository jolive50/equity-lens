"""LangChain-compatible agent definitions for the StockSense workflow.

These agents encapsulate prompts and chaining logic so that the
LangGraph orchestration layer can focus on routing state between them.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Structured output from the prediction agent."""

    direction: Literal["up", "down", "neutral"]
    confidence: float
    narrative: str


@dataclass
class SentimentResult:
    """Structured output from the sentiment agent."""
    
    current: Literal["positive", "neutral", "negative"]
    score: float
    trend: Literal["improving", "stable", "declining"]
    headlines: List[str]


@dataclass
class SmartMoneyResult:
    """Structured output from the smart money agent."""
    
    institutions: Dict[str, any]
    insiders: Dict[str, any]
    congress: Dict[str, any]


class PredictionAgent:
    """Generates probabilistic directional forecasts for stock symbols."""

    def __init__(
        self,
        llm: Runnable,
        *,
        prompt: Optional[ChatPromptTemplate] = None,
        use_ml_model: bool = True,
    ) -> None:
        self.use_ml_model = use_ml_model
        if use_ml_model:
            try:
                from .models.forecaster import create_forecaster
                self.forecaster = create_forecaster("gradient_boosting")
            except ImportError:
                logger.warning("ML forecaster not available, falling back to LLM-based prediction")
                self.forecaster = None
                self.use_ml_model = False
        else:
            self.forecaster = None
        
        base_prompt = prompt or ChatPromptTemplate.from_template(
            """You are an expert equity prediction specialist focused on probabilistic forecasting.

Analyze the following data for {ticker} and provide a directional forecast:

Market Data: {market_data}
Fundamentals: {fundamentals}

Provide your analysis in this exact format:
DIRECTION: [up|down|neutral]
CONFIDENCE: [0.0-1.0]
NARRATIVE: [2-3 sentence explanation of your reasoning]

Focus on:
- Technical indicators and momentum
- Fundamental strength/weakness
- Market regime and volatility
- Risk factors and catalysts

Be conservative with confidence - only assign high confidence (>0.9) when multiple factors strongly align."""
        )
        self._chain = base_prompt | llm | StrOutputParser()

    def run(self, *, ticker: str, market_data: Dict, fundamentals: Dict[str, float]) -> PredictionResult:
        # Try ML model first if available
        if self.use_ml_model and self.forecaster and isinstance(market_data, list):
            try:
                ml_result = self.forecaster.predict(market_data, fundamentals)
                
                # Convert ML result to PredictionResult format
                return PredictionResult(
                    direction=ml_result.direction,
                    confidence=ml_result.confidence,
                    narrative=f"ML model prediction based on technical indicators and market patterns. "
                             f"Confidence: {ml_result.confidence:.1%}. "
                             f"95% horizon: {ml_result.horizon_95.get('days', 0)} days."
                )
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}, falling back to LLM")
        
        # Fallback to LLM-based analysis
        raw_output = self._chain.invoke({
            "ticker": ticker,
            "market_data": str(market_data),
            "fundamentals": str(fundamentals)
        })
        
        direction = _extract_direction(raw_output)
        confidence = _extract_confidence(raw_output, default=0.5)
        narrative = _extract_narrative(raw_output)
        
        return PredictionResult(
            direction=direction,
            confidence=confidence,
            narrative=narrative
        )


class SentimentAgent:
    """Analyzes news sentiment and social media for stock symbols."""

    def __init__(
        self,
        llm: Runnable,
        *,
        prompt: Optional[ChatPromptTemplate] = None,
        use_finbert: bool = True,
    ) -> None:
        self.use_finbert = use_finbert
        if use_finbert:
            try:
                from .sentiment.finbert import create_sentiment_analyzer
                self.sentiment_analyzer = create_sentiment_analyzer()
            except ImportError:
                logger.warning("FinBERT not available, falling back to LLM-based sentiment")
                self.sentiment_analyzer = None
                self.use_finbert = False
        else:
            self.sentiment_analyzer = None
        
        base_prompt = prompt or ChatPromptTemplate.from_template(
            """You are a financial sentiment analyst specializing in news and social media analysis.

Analyze the sentiment for {ticker} based on the following news data:

News Data: {news_data}

Provide your analysis in this exact format:
CURRENT: [positive|neutral|negative]
SCORE: [0.0-1.0]
TREND: [improving|stable|declining]
HEADLINES: [3 most relevant headlines, one per line]

Consider:
- Earnings announcements and guidance
- Product launches and partnerships
- Regulatory news and legal issues
- Analyst upgrades/downgrades
- Market sentiment and social media buzz

Be objective and focus on factual sentiment rather than speculation."""
        )
        self._chain = base_prompt | llm | StrOutputParser()

    def run(self, *, ticker: str, news_data: List[Dict]) -> SentimentResult:
        # Try FinBERT first if available
        if self.use_finbert and self.sentiment_analyzer and news_data:
            try:
                finbert_result = self.sentiment_analyzer.process_news_articles(news_data)
                
                # Convert FinBERT result to SentimentResult format
                return SentimentResult(
                    current=finbert_result["current"],
                    score=finbert_result["score"],
                    trend=finbert_result["trend"],
                    headlines=finbert_result["headlines"]
                )
            except Exception as e:
                logger.warning(f"FinBERT analysis failed: {e}, falling back to LLM")
        
        # Fallback to LLM-based analysis
        raw_output = self._chain.invoke({
            "ticker": ticker,
            "news_data": str(news_data)
        })
        
        current = _extract_sentiment_current(raw_output)
        score = _extract_confidence(raw_output, default=0.5)
        trend = _extract_sentiment_trend(raw_output)
        headlines = _extract_headlines(raw_output)
        
        return SentimentResult(
            current=current,
            score=score,
            trend=trend,
            headlines=headlines
        )


class SmartMoneyAgent:
    """Tracks institutional, insider, and congressional trading activity."""

    def __init__(
        self,
        llm: Runnable,
        *,
        prompt: Optional[ChatPromptTemplate] = None,
        use_data_service: bool = True,
    ) -> None:
        self.use_data_service = use_data_service
        if use_data_service:
            try:
                from .smart_money import create_smart_money_service
                import os
                api_keys = {
                    "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
                    "finnhub": os.getenv("FINNHUB_API_KEY")
                }
                # Filter out None values
                api_keys = {k: v for k, v in api_keys.items() if v}
                self.smart_money_service = create_smart_money_service(api_keys)
            except ImportError:
                logger.warning("Smart money service not available, falling back to LLM-based analysis")
                self.smart_money_service = None
                self.use_data_service = False
        else:
            self.smart_money_service = None
        
        base_prompt = prompt or ChatPromptTemplate.from_template(
            """You are a smart money analyst tracking institutional flows, insider trading, and congressional disclosures.

Analyze smart money activity for {ticker}:

Provide your analysis in this exact format:
INSTITUTIONS: [ownership change, top buyers/sellers, summary]
INSIDERS: [recent activity, net position, summary]
CONGRESS: [recent trades, summary]

Focus on:
- Recent institutional ownership changes
- Insider buying/selling patterns
- Congressional trading disclosures
- Notable hedge fund positions
- Berkshire Hathaway or similar iconic investors

If no recent data is available, indicate "No recent activity" for that category."""
        )
        self._chain = base_prompt | llm | StrOutputParser()

    def run(self, *, ticker: str) -> SmartMoneyResult:
        # Try smart money service first if available
        if self.use_data_service and self.smart_money_service:
            try:
                institutional = self.smart_money_service.get_institutional_summary(ticker)
                insider = self.smart_money_service.get_insider_summary(ticker)
                congressional = self.smart_money_service.get_congressional_summary(ticker)
                
                return SmartMoneyResult(
                    institutions=institutional,
                    insiders=insider,
                    congress=congressional
                )
            except Exception as e:
                logger.warning(f"Smart money service failed: {e}, falling back to LLM")
        
        # Fallback to LLM-based analysis
        raw_output = self._chain.invoke({"ticker": ticker})
        
        institutions = _extract_smart_money_section(raw_output, "INSTITUTIONS")
        insiders = _extract_smart_money_section(raw_output, "INSIDERS")
        congress = _extract_smart_money_section(raw_output, "CONGRESS")
        
        return SmartMoneyResult(
            institutions=institutions,
            insiders=insiders,
            congress=congress
        )


class ExplanationAgent:
    """Generates plain English explanations for retail investors."""

    def __init__(
        self,
        llm: Runnable,
        *,
        prompt: Optional[ChatPromptTemplate] = None,
    ) -> None:
        base_prompt = prompt or ChatPromptTemplate.from_template(
            """You are a financial educator who explains complex market analysis in simple, clear language for retail investors.

Generate a comprehensive explanation for {ticker} based on:

Prediction: {prediction}
Sentiment: {sentiment}
Smart Money: {smart_money}
User Tier: {user_tier}
Confidence Level: {confidence_level}

Create a clear, educational explanation that:

1. Starts with the main directional call and confidence level
2. Explains the key factors driving the prediction in plain English
3. Discusses sentiment context and what it means
4. For premium users, includes smart money insights
5. Ends with appropriate risk warnings and disclaimers

Use simple language, avoid jargon, and be honest about limitations.
Always emphasize this is informational, not investment advice.

Keep explanations concise but comprehensive - aim for 3-4 paragraphs."""
        )
        self._chain = base_prompt | llm | StrOutputParser()

    def run(
        self, 
        *, 
        ticker: str,
        prediction: Dict,
        sentiment: Dict,
        smart_money: Dict,
        user_tier: str,
        confidence_level: str
    ) -> str:
        return self._chain.invoke({
            "ticker": ticker,
            "prediction": str(prediction),
            "sentiment": str(sentiment),
            "smart_money": str(smart_money),
            "user_tier": user_tier,
            "confidence_level": confidence_level
        }).strip()


def build_openai_llm(model: str = "gpt-4o-mini") -> Runnable:
    """Build OpenAI LLM using LangChain."""
    from langchain_openai import ChatOpenAI
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0.1,  # Low temperature for consistent financial analysis
        max_tokens=1000
    )


def build_real_llm_agent(tag: str, llm_model: Runnable) -> Runnable:
    """Utility to build a real LLM agent that processes actual prompts."""

    def _process(prompt) -> str:
        try:
            if hasattr(prompt, 'messages') and prompt.messages:
                # Extract the actual content and process with real LLM
                result = llm_model.invoke(prompt)
                if hasattr(result, 'content'):
                    return result.content
                else:
                    return str(result)
        elif isinstance(prompt, str):
                # Handle string prompts by creating a simple chat template
                chat_prompt = ChatPromptTemplate.from_template("{input}")
                formatted_prompt = chat_prompt.format(input=prompt)
                result = llm_model.invoke(formatted_prompt)
                return result.content if hasattr(result, 'content') else str(result)
        else:
                # Fallback processing
                result = llm_model.invoke(str(prompt))
                return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            logger.error(f"[{tag}] Error processing prompt: {e}")
            return f"[{tag}] Processing error: {str(e)[:100]}..."

    return RunnableLambda(_process)


# New agent classes for enhanced workflow

class CoordinationAgent:
    """Coordinates multiple analysis agents and synthesizes insights."""

    def __init__(
        self,
        llm: Runnable,
        *,
        confidence_threshold: float = 0.95,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        
        base_prompt = ChatPromptTemplate.from_template(
            """You are a coordination agent that synthesizes analysis from multiple working agents.
            
You will receive inputs from historical analysis agents and sentiment analysis agents for multiple S&P 500 companies.

Working Agent Results: {working_agent_results}

Your task is to:
1. Synthesize insights across all companies
2. Identify patterns and correlations
3. Generate a confidence-weighted summary
4. Only recommend actions if confidence >= {confidence_threshold}

Provide your analysis in this format:
OVERALL_CONFIDENCE: [0.0-1.0]
SYNTHESIS: [3-4 sentence summary of key insights across all companies]
TOP_PERFORMERS: [Companies with highest bullish prospects]
CAUTION_FLAGS: [Companies with concerning signals]
RECOMMENDATION: [Recommend action only if >={confidence_threshold} confidence, otherwise "HOLD"]
RISK_ASSESSMENT: [Market-wide risk factors]

Focus on fundamental trends, market sentiment convergence, and risk diversification."""
        )
        self._chain = base_prompt | llm | StrOutputParser()

    def coordinate(
        self, 
        *, 
        tickers: List[str], 
        working_agent_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Coordinate analysis across multiple agents and tickers."""
        
        # Extract overall confidence from working agents
        overall_confidence = 0.0
        if 'historical_confidence' in working_agent_results:
            overall_confidence += working_agent_results['historical_confidence'] * 0.6
        if 'sentiment_confidence' in working_agent_results:
            overall_confidence += working_agent_results['sentiment_confidence'] * 0.4
        
        # Only proceed if confidence meets threshold
        if overall_confidence < self.confidence_threshold:
            return {
                "confidence": overall_confidence,
                "recommendation": "INSUFFICIENT_CONFIDENCE",
                "threshold_met": False,
                "message": f"Confidence {overall_confidence:.3f} below threshold {self.confidence_threshold}"
            }
        
        raw_output = self._chain.invoke({
            "working_agent_results": str(working_agent_results),
            "confidence_threshold": self.confidence_threshold
        })
        
        return {
            "confidence": overall_confidence,
            "threshold_met": True,
            "raw_analysis": raw_output,
            "_extract_coordination_data": raw_output
        }


class HistoricalAnalysisAgent:
    """Analyzes historical data and fundamental metrics for S&P 500 companies."""

    def __init__(
        self,
        llm: Runnable,
        *,
        use_fundamental_analysis: bool = True,
    ) -> None:
        self.use_fundamental_analysis = use_fundamental_analysis
        
        base_prompt = ChatPromptTemplate.from_template(
            """You are a historical market analysis specialist focused on S&P 500 companies.

Analyze the historical data and fundamentals for multiple companies:

Companies Data: {comprehensive_data}
Market Data: {market_data}

Focus on these S&P 500 metrics:
- Revenue Growth Rate (YoY %)
- EBITDA Margin (%)
- P/E Ratio (Current vs 5-year average)
- Debt-to-EBITDA Ratio
- ROE (Return on Equity)
- Price Momentum (3M, 6M, 12M)

Provide analysis in this format:
OVERALL_TREND: [positive|neutral|negative]
TECHNICAL_SCORE: [0-100]
FUNDAMENTAL_SCORE: [0-100]
TOP_PERFORMERS: [Company tickers with strongest fundamentals]
CONCERN_COMPANIES: [Company tickers with emerging issues]
CONFIDENCE: [0.0-1.0]
NARRATIVE: [Comprehensive analysis of market fundamentals]

Emphasize:
- Relative performance vs SPY benchmark
- Sector rotation trends
- Valuation attractiveness
- Earnings growth sustainability"""
        )
        self._chain = base_prompt | llm | StrOutputParser()

    def run(self, *, comprehensive_data: Dict[str, Dict], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run historical analysis across multiple companies."""
        raw_output = self._chain.invoke({
            "comprehensive_data": str(comprehensive_data),
            "market_data": str(market_data)
        })
        
        # Extract metrics from analysis
        overall_trend = _extract_historical_trend(raw_output)
        technical_score = _extract_score(raw_output, "TECHNICAL_SCORE")
        fundamental_score = _extract_score(raw_output, "FUNDAMENTAL_SCORE")
        confidence = _extract_confidence(raw_output, default=0.5)
        narrative = _extract_narrative(raw_output)
        top_performers = _extract_ticker_list(raw_output, "TOP_PERFORMERS")
        concern_companies = _extract_ticker_list(raw_output, "CONCERN_COMPANIES")
        
        # Enhanced analysis combining LLM insights with quantitative metrics
        market_trend_analysis = self._analyze_market_trends(comprehensive_data)
        
        # Combine LLM analysis with quantitative trends
        combined_confidence = self._combine_confidence_scores(confidence, market_trend_analysis["confidence"])
        
        return {
            "overall_trend": overall_trend,
            "technical_score": technical_score,
            "fundamental_score": fundamental_score,
            "confidence": combined_confidence,
            "narrative": narrative,
            "top_performers": top_performers,
            "concern_companies": concern_companies,
            "historical_confidence": combined_confidence,
            "market_trends": market_trend_analysis  # Include comprehensive trend analysis
        }
    
    def _analyze_market_trends(self, comprehensive_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze market trends using quantitative fundamental metrics."""
        import statistics
        
        sector_metrics = {
            "revenue_growth": [],
            "ebitda_margin": [],
            "pe_ratios": [],
            "debt_to_equity": [],
            "roe": [],
            "price_momentum": []
        }
        
        ticker_fundamentals = {}
        
        # Aggregate metrics across all tickers
        for ticker, data in comprehensive_data.items():
            fundamentals = data["fundamentals"]
            ticker_fundamentals[ticker] = fundamentals
            
            # Collect various metrics
            if "revenue_growth" in fundamentals:
                sector_metrics["revenue_growth"].append(fundamentals["revenue_growth"])
            if "ebitda_margin" in fundamentals:
                sector_metrics["ebitda_margin"].append(fundamentals["ebitda_margin"])
            if "pe_ratio" in fundamentals and fundamentals["pe_ratio"] > 0:
                sector_metrics["pe_ratios"].append(fundamentals["pe_ratio"])
            if "debt_to_equity" in fundamentals:
                sector_metrics["debt_to_equity"].append(fundamentals["debt_to_equity"])
            if "roe" in fundamentals:
                sector_metrics["roe"].append(fundamentals["roe"])
            if "price_momentum_12m" in fundamentals:
                sector_metrics["price_momentum"].append(fundamentals["price_momentum_12m"])
        
        # Calculate sector-wide trends
        trends = {
            "avg_revenue_growth": statistics.mean(sector_metrics["revenue_growth"]) if sector_metrics["revenue_growth"] else 0.0,
            "avg_ebitda_margin": statistics.mean(sector_metrics["ebitda_margin"]) if sector_metrics["ebitda_margin"] else 0.0,
            "avg_pe_ratio": statistics.mean([p for p in sector_metrics["pe_ratios"] if p > 0]) if sector_metrics["pe_ratios"] else 0.0,
            "avg_roe": statistics.mean(sector_metrics["roe"]) if sector_metrics["roe"] else 0.0,
            "avg_price_momentum": statistics.mean(sector_metrics["price_momentum"]) if sector_metrics["price_momentum"] else 0.0
        }
        
        # Determine trend directions
        trends["revenue_trend"] = "positive" if trends["avg_revenue_growth"] > 0.05 else \
                                 "negative" if trends["avg_revenue_growth"] < -0.02 else "neutral"
        
        trends["profitability_trend"] = "strong" if trends["avg_roe"] > 0.15 else \
                                       "weak" if trends["avg_roe"] < 0.08 else "moderate"
        
        trends["valuation_trend"] = "expensive" if trends["avg_pe_ratio"] > 25 else \
                                   "cheap" if trends["avg_pe_ratio"] < 15 else "fair"
        
        trends["momentum_trend"] = "positive" if trends["avg_price_momentum"] > 0.1 else \
                                  "negative" if trends["avg_price_momentum"] < -0.05 else "neutral"
        
        # Calculate confidence based on trend consistency
        confidence_factors = []
        
        if trends["revenue_trend"] == "positive":
            confidence_factors.append(0.8)
        if trends["profitability_trend"] == "strong":
            confidence_factors.append(0.7)
        if trends["valuation_trend"] in ["fair", "cheap"]:
            confidence_factors.append(0.6)
        if trends["momentum_trend"] == "positive":
            confidence_factors.append(0.7)
        
        trends["confidence"] = statistics.mean(confidence_factors) if confidence_factors else 0.5
        
        return trends
    
    def _combine_confidence_scores(self, llm_confidence: float, trend_confidence: float) -> float:
        """Combine LLM and quantitative trend confidence scores."""
        # Weighted combination favoring quantitative analysis
        return (llm_confidence * 0.4 + trend_confidence * 0.6)


class SentimentAnalysisAgent:
    """Analyzes sentiment from news and social media for S&P 500 companies."""

    def __init__(
        self,
        llm: Runnable,
        *,
        use_finbert: bool = True,
    ) -> None:
        self.use_finbert = use_finbert
        
        base_prompt = ChatPromptTemplate.from_template(
            """You are a financial sentiment specialist analyzing S&P 500 companies.

Analyze sentiment trends from news across multiple companies:

Companies News: {comprehensive_news_data}
Current Market Context: {market_context}

Focus on:
- Earnings announcement sentiment
- Analyst upgrades/downgrades
- Sector-specific sentiment
- Crisis/news sentiment impact
- Social media sentiment trends

Provide analysis in this format:
OVERALL_SENTIMENT: [positive|neutral|negative]
SENTIMENT_SCORE: [0-100]
TREND_DIRECTION: [improving|stable|declining]
POSITIVE_DRIVERS: [Key positive narratives]
NEGATIVE_CONCERNS: [Key negative narratives]
CONFIDENCE: [0.0-1.0]
HEADLINES: [Top 3 most significant headlines across companies]

Consider sector rotation and macro sentiment impact."""
        )
        self._chain = base_prompt | llm | StrOutputParser()

    def run(self, *, comprehensive_news_data: Dict[str, List[Dict]], market_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run sentiment analysis across multiple companies."""
        
        # Try FinBERT first if available and we have news data
        if self.use_finbert and self.sentiment_analyzer and comprehensive_news_data:
            try:
                # Use enhanced FinBERT for sector-wide sentiment analysis
                finbert_result = self.sentiment_analyzer.analyze_sector_sentiment(comprehensive_news_data)
                
                # Extract headline summaries
                headlines = []
                for ticker, result in finbert_result["sector_sentiment"].items():
                    headlines.extend(result.get("headlines", [])[:1])  # Take top headline per ticker
                
                return {
                    "overall_sentiment": finbert_result["overall_sentiment"],
                    "sentiment_score": int(finbert_result["overall_score"] * 100),
                    "trend_direction": "improving" if finbert_result["overall_score"] > 0.6 else \
                                     "declining" if finbert_result["overall_score"] < 0.4 else "stable",
                    "confidence": finbert_result["convergence_score"],
                    "headlines": headlines[:3],
                    "positive_drivers": self._extract_positives_from_headlines(headlines),
                    "negative_concerns": self._extract_negatives_from_headlines(headlines),
                    "sentiment_confidence": finbert_result["convergence_score"],
                    "sector_analysis": finbert_result["sector_sentiment"],
                    "total_articles": finbert_result["total_articles"]
                }
            except Exception as e:
                logger.warning(f"FinBERT sector analysis failed: {e}, falling back to LLM")
        
        # Fallback to LLM-based analysis
        raw_output = self._chain.invoke({
            "comprehensive_news_data": str(comprehensive_news_data),
            "market_context": str(market_context or {})
        })
        
        # Extract metrics from analysis
        overall_sentiment = _extract_sentiment_current(raw_output)
        sentiment_score = _extract_score(raw_output, "SENTIMENT_SCORE")
        trend_direction = _extract_sentiment_trend(raw_output)
        confidence = _extract_confidence(raw_output, default=0.5)
        headlines = _extract_headlines(raw_output)
        positive_drivers = _extract_text_section(raw_output, "POSITIVE_DRIVERS")
        negative_concerns = _extract_text_section(raw_output, "NEGATIVE_CONCERNS")
        
        return {
            "overall_sentiment": overall_sentiment,
            "sentiment_score": sentiment_score,
            "trend_direction": trend_direction,
            "confidence": confidence,
            "headlines": headlines,
            "positive_drivers": positive_drivers,
            "negative_concerns": negative_concerns,
            "sentiment_confidence": confidence
        }
    
    def _extract_positives_from_headlines(self, headlines: List[str]) -> str:
        """Extract positive themes from headlines."""
        positive_keywords = ["strong", "growth", "beat", "exceed", "positive", "bullish", "upgrade"]
        positive_themes = []
        
        for headline in headlines:
            headline_lower = headline.lower()
            for keyword in positive_keywords:
                if keyword in headline_lower:
                    positive_themes.append(f"'{keyword}' trend")
                    break
        
        return ", ".join(set(positive_themes)) if positive_themes else "Mixed positive indicators"
    
    def _extract_negatives_from_headlines(self, headlines: List[str]) -> str:
        """Extract negative themes from headlines."""
        negative_keywords = ["weak", "miss", "down", "negative", "bearish", "downgrade", "concern", "volatility"]
        negative_themes = []
        
        for headline in headlines:
            headline_lower = headline.lower()
            for keyword in negative_keywords:
                if keyword in headline_lower:
                    negative_themes.append(f"'{keyword}' trend")
                    break
        
        return ", ".join(set(negative_themes)) if negative_themes else "No major concerns identified"




# Helper functions for parsing agent outputs

CONFIDENCE_PATTERN = re.compile(r"CONFIDENCE:\s*([0-9.]+)")
DIRECTION_PATTERN = re.compile(r"DIRECTION:\s*(up|down|neutral)", re.IGNORECASE)
SENTIMENT_PATTERN = re.compile(r"CURRENT:\s*(positive|neutral|negative)", re.IGNORECASE)
TREND_PATTERN = re.compile(r"TREND:\s*(improving|stable|declining)", re.IGNORECASE)


def _extract_confidence(raw_text: str, *, default: float) -> float:
    """Extract confidence score from agent output."""
    match = CONFIDENCE_PATTERN.search(raw_text)
    if not match:
        return default
    value = float(match.group(1))
    return max(0.0, min(1.0, value))


def _extract_direction(raw_text: str) -> Literal["up", "down", "neutral"]:
    """Extract direction from agent output."""
    match = DIRECTION_PATTERN.search(raw_text)
    if not match:
        return "neutral"
    return match.group(1).lower()


def _extract_narrative(raw_text: str) -> str:
    """Extract narrative explanation from agent output."""
    lines = raw_text.split('\n')
    narrative_lines = []
    in_narrative = False
    
    for line in lines:
        if line.strip().startswith('NARRATIVE:'):
            in_narrative = True
            narrative_lines.append(line.replace('NARRATIVE:', '').strip())
        elif in_narrative and line.strip():
            narrative_lines.append(line.strip())
        elif in_narrative and not line.strip():
            break
    
    return ' '.join(narrative_lines) if narrative_lines else "No narrative provided"


def _extract_sentiment_current(raw_text: str) -> Literal["positive", "neutral", "negative"]:
    """Extract current sentiment from agent output."""
    match = SENTIMENT_PATTERN.search(raw_text)
    if not match:
        return "neutral"
    return match.group(1).lower()


def _extract_sentiment_trend(raw_text: str) -> Literal["improving", "stable", "declining"]:
    """Extract sentiment trend from agent output."""
    match = TREND_PATTERN.search(raw_text)
    if not match:
        return "stable"
    return match.group(1).lower()


def _extract_headlines(raw_text: str) -> List[str]:
    """Extract headlines from agent output."""
    lines = raw_text.split('\n')
    headlines = []
    in_headlines = False
    
    for line in lines:
        if line.strip().startswith('HEADLINES:'):
            in_headlines = True
            headline = line.replace('HEADLINES:', '').strip()
            if headline:
                headlines.append(headline)
        elif in_headlines and line.strip() and not line.strip().startswith(('CURRENT:', 'SCORE:', 'TREND:')):
            headlines.append(line.strip())
        elif in_headlines and not line.strip():
            break
    
    return headlines[:3]  # Return max 3 headlines


def _extract_smart_money_section(raw_text: str, section: str) -> Dict[str, any]:
    """Extract a specific section from smart money agent output."""
    lines = raw_text.split('\n')
    section_data = []
    in_section = False
    
    for line in lines:
        if line.strip().startswith(f'{section}:'):
            in_section = True
            content = line.replace(f'{section}:', '').strip()
            if content:
                section_data.append(content)
        elif in_section and line.strip() and not line.strip().startswith(('INSTITUTIONS:', 'INSIDERS:', 'CONGRESS:')):
            section_data.append(line.strip())
        elif in_section and not line.strip():
            break
    
    # Parse the section data into structured format
    if not section_data:
        return {"summary": f"No recent {section.lower()} activity"}
    
    summary = ' '.join(section_data)
    return {"summary": summary}


# Additional helper functions for new agents

HISTORICAL_TREND_PATTERN = re.compile(r"OVERALL_TREND:\s*(positive|neutral|negative)", re.IGNORECASE)
SCORE_PATTERN = re.compile(r"(\w+_SCORE):\s*(\d+)", re.IGNORECASE)
TICKER_LIST_PATTERN = re.compile(r"(\w+):\s*\[([^\]]+)\]", re.IGNORECASE)

def _extract_historical_trend(raw_text: str) -> Literal["positive", "neutral", "negative"]:
    """Extract historical trend from agent output."""
    match = HISTORICAL_TREND_PATTERN.search(raw_text)
    if not match:
        return "neutral"
    return match.group(1).lower()


def _extract_score(raw_text: str, score_type: str) -> int:
    """Extract score from agent output."""
    pattern = re.compile(f"{score_type}:\\s*(\\d+)", re.IGNORECASE)
    match = pattern.search(raw_text)
    if not match:
        return 50  # Default neutral score
    return int(match.group(1))


def _extract_ticker_list(raw_text: str, section: str) -> List[str]:
    """Extract list of tickers from agent output."""
    pattern = re.compile(f"{section}:\\s*\\[([^\\]]+)\\]", re.IGNORECASE)
    match = pattern.search(raw_text)
    if not match:
        return []
    
    # Split by comma and clean up
    tickers = [ticker.strip().upper() for ticker in match.group(1).split(',')]
    return [ticker for ticker in tickers if ticker]


def _extract_text_section(raw_text: str, section: str) -> str:
    """Extract a text section from agent output."""
    lines = raw_text.split('\n')
    section_data = []
    in_section = False
    
    for line in lines:
        if line.strip().startswith(f'{section}:'):
            in_section = True
            content = line.replace(f'{section}:', '').strip()
            if content:
                section_data.append(content)
        elif in_section and line.strip() and not line.strip().startswith(('OVERALL_SENTIMENT:', 'SENTIMENT_SCORE:', 'TREND_DIRECTION:', 'CONFIDENCE:')):
            section_data.append(line.strip())
        elif in_section and not line.strip():
            break
    
    return ' '.join(section_data) if section_data else "No data available"
