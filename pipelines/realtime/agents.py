"""LangChain-compatible agent definitions for the LangGraph workflow.

These agents encapsulate prompts and chaining logic so that the
LangGraph orchestration layer can focus on routing state between them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda


@dataclass
class PredictionResult:
    """Structured output from the prediction agent."""

    narrative: str
    confidence: float


class PredictionAgent:
    """Generates quantitative guidance for a requested symbol."""

    def __init__(
        self,
        llm: Runnable,
        *,
        prompt: Optional[ChatPromptTemplate] = None,
    ) -> None:
        base_prompt = prompt or ChatPromptTemplate.from_template(
            """You are an equity prediction specialist.
Summarise the likely 14-day directional move for {symbol} using the following context.\n\n"""
            "Market Data:\n{market_data}\n\n"
            "Respond with two sentences that include a confidence percentage."""
        )
        self._chain = base_prompt | llm | StrOutputParser()

    def run(self, *, symbol: str, market_data: str) -> PredictionResult:
        raw_output = self._chain.invoke({"symbol": symbol, "market_data": market_data})
        return PredictionResult(
            narrative=raw_output.strip(),
            confidence=_extract_confidence(raw_output, default=0.5),
        )


class SentimentAgent:
    """Aggregates qualitative sentiment about the security."""

    def __init__(
        self,
        llm: Runnable,
        *,
        prompt: Optional[ChatPromptTemplate] = None,
    ) -> None:
        base_prompt = prompt or ChatPromptTemplate.from_template(
            """You analyse news headlines and filings for investor sentiment.
Classify the combined sentiment as bullish, bearish, or neutral and justify in a bullet list.\n\n"
            "Symbol: {symbol}\n"
            "News and Filings:\n{news}\n"""
        )
        self._chain = base_prompt | llm | StrOutputParser()

    def run(self, *, symbol: str, news: str) -> str:
        return self._chain.invoke({"symbol": symbol, "news": news}).strip()


class ExplanationAgent:
    """Turns model outputs into an end-user explanation."""

    def __init__(
        self,
        llm: Runnable,
        *,
        prompt: Optional[ChatPromptTemplate] = None,
    ) -> None:
        base_prompt = prompt or ChatPromptTemplate.from_template(
            """Explain the prediction for a retail investor.
Include the quantitative view, sentiment context, and highlight whether we are in full guidance or trend-only mode.\n\n"
            "Prediction Narrative:\n{prediction}\n\n"
            "Sentiment Summary:\n{sentiment}\n\n"
            "Mode: {mode}\n"""
        )
        self._chain = base_prompt | llm | StrOutputParser()

    def run(self, *, prediction: str, sentiment: str, mode: str) -> str:
        return self._chain.invoke(
            {
                "prediction": prediction,
                "sentiment": sentiment,
                "mode": mode,
            }
        ).strip()


def build_mock_llm(tag: str) -> Runnable:
    """Utility to build a lightweight mock LLM for offline demos."""

    def _mock(prompt: str) -> str:
        return f"[{tag}] {prompt[:200]}..."

    return RunnableLambda(lambda prompt: _mock(prompt["messages"][0]["content"]))


CONFIDENCE_PATTERN = re.compile(r"(\d{1,3})\s*%")


def _extract_confidence(raw_text: str, *, default: float) -> float:
    match = CONFIDENCE_PATTERN.search(raw_text)
    if not match:
        return default
    value = int(match.group(1)) / 100.0
    return max(0.0, min(1.0, value))
