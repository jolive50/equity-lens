"""Explanation agent for generating plain English explanations."""

import logging
from typing import Dict, Optional
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


logger = logging.getLogger(__name__)


class ExplanationAgent:
    """Generates plain English explanations for retail investors.

    Translates technical analysis into simple language using LLM.
    """

    def __init__(
        self,
        llm: Runnable,
        *,
        prompt: Optional[ChatPromptTemplate] = None
    ) -> None:
        """Initialize explanation agent.

        Args:
            llm: Language model for generating explanations
            prompt: Optional custom prompt template
        """
        base_prompt = prompt or ChatPromptTemplate.from_template(
            """You are a financial educator explaining stock analysis to retail investors.

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
        """Generate plain English explanation.

        Args:
            ticker: Stock symbol
            prediction: Prediction results
            sentiment: Sentiment analysis results
            smart_money: Smart money data
            user_tier: User subscription level (basic/registered/premium)
            confidence_level: Overall confidence (high/medium/low)

        Returns:
            Plain English explanation string
        """
        return self._chain.invoke({
            "ticker": ticker,
            "prediction": str(prediction),
            "sentiment": str(sentiment),
            "smart_money": str(smart_money),
            "user_tier": user_tier,
            "confidence_level": confidence_level
        }).strip()
