"""LangChain-compatible agent definitions for the StockSense workflow.

These agents encapsulate prompts and chaining logic so that the
LangGraph orchestration layer can focus on routing state between them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda


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
    ) -> None:
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
    ) -> None:
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
    ) -> None:
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


def build_mock_llm(tag: str) -> Runnable:
    """Utility to build a lightweight mock LLM for offline demos."""

    def _mock(prompt: str) -> str:
        return f"[{tag}] Mock response for: {prompt[:200]}..."

    return RunnableLambda(lambda prompt: _mock(prompt["messages"][0]["content"]))


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
