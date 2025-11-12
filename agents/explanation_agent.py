"""Explanation Agent for generating narratives.

JOSH's Component - Explanation Agent
Generates plain English explanations using OpenAI (for team use only).
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ExplanationAgent:
    """Agent that generates plain English explanations."""

    def __init__(self, llm=None):
        """Initialize ExplanationAgent.

        Args:
            llm: Optional LangChain LLM for text generation
        """
        self.llm = llm
        logger.info("ExplanationAgent initialized")

    def run(
        self,
        ticker: str,
        prediction: Dict[str, Any],
        sentiment: Dict[str, Any],
        smart_money: Dict[str, Any],
        user_tier: str,
        confidence_level: str
    ) -> str:
        """Generate plain English explanation.

        Args:
            ticker: Stock symbol
            prediction: Prediction results
            sentiment: Sentiment results
            smart_money: Smart money data
            user_tier: User subscription level
            confidence_level: Overall confidence

        Returns:
            String with explanation
        """
        logger.info("      ╔══════════════════════════════════════════════════════╗")
        logger.info("      ║  📝 ExplanationAgent Execution                     ║")
        logger.info("      ╚══════════════════════════════════════════════════════╝")
        logger.info(f"         Ticker: {ticker}")
        logger.info(f"         User tier: {user_tier}")
        logger.info(f"         Confidence level: {confidence_level}")
        logger.info(f"         → Input data:")
        logger.info(f"            Prediction direction: {prediction.get('direction', 'N/A')}")
        logger.info(f"            Prediction confidence: {prediction.get('confidence', 0):.1%}")
        logger.info(f"            Sentiment: {sentiment.get('current', 'N/A')} ({sentiment.get('score', 0):.2f})")

        # If LLM available, use it
        if self.llm:
            logger.info(f"         → Using LLM-based explanation generation")
            explanation = self._generate_llm_explanation(
                ticker, prediction, sentiment, smart_money, user_tier, confidence_level
            )
        else:
            logger.info(f"         → Using template-based explanation generation")
            explanation = self._generate_template_explanation(
                ticker, prediction, sentiment, smart_money, user_tier, confidence_level
            )

        logger.info("      ┌──────────────────────────────────────────────────┐")
        logger.info("      │  📝 Explanation Result                           │")
        logger.info("      └──────────────────────────────────────────────────┘")
        logger.info(f"         Generated explanation: {len(explanation)} characters")
        logger.info(f"         Preview: {explanation[:100]}...")

        return explanation

    def _generate_template_explanation(
        self,
        ticker: str,
        prediction: Dict[str, Any],
        sentiment: Dict[str, Any],
        smart_money: Dict[str, Any],
        user_tier: str,
        confidence_level: str
    ) -> str:
        """Generate template-based explanation."""
        direction = prediction.get("direction", "neutral")
        confidence = prediction.get("confidence", 0.5)
        sent_current = sentiment.get("current", "neutral")
        sent_score = sentiment.get("score", 0.5)

        explanation = f"""
Analysis for {ticker}:

Prediction: {direction.upper()} with {confidence:.1%} confidence
The ML model predicts {ticker} will move {direction}.
Confidence level: {confidence_level}

Sentiment: {sent_current.upper()} ({sent_score:.1%})
Recent news sentiment is {sent_current}.

"""

        if user_tier == "premium" and smart_money:
            explanation += "Smart Money: Premium data available.\n"

        explanation += """
Disclaimer: This is informational analysis only, not investment advice.
Always do your own research before making investment decisions.
"""

        return explanation.strip()

    def _generate_llm_explanation(
        self,
        ticker: str,
        prediction: Dict[str, Any],
        sentiment: Dict[str, Any],
        smart_money: Dict[str, Any],
        user_tier: str,
        confidence_level: str
    ) -> str:
        """Generate LLM-based explanation."""
        # TODO: Implement when OpenAI LLM is configured
        logger.info("LLM explanation not yet implemented, using template")
        return self._generate_template_explanation(
            ticker, prediction, sentiment, smart_money, user_tier, confidence_level
        )
