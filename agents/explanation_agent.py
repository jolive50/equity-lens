"""Explanation Agent for generating narratives.

JOSH's Component - Explanation Agent
Generates plain English explanations using OpenAI (for team use only).
"""
import logging
from typing import Any, Dict, List, Optional

from tools.explanation_tools import (
    generate_llm_explanation,
    generate_template_explanation,
    load_default_llm,
)

logger = logging.getLogger(__name__)


class ExplanationAgent:
    """Agent that generates plain English explanations."""

    def __init__(self, llm: Optional[Any] = None):
        """Initialize ExplanationAgent.

        Args:
            llm: Optional LangChain LLM for text generation
        """
        if llm is None:
            self.llm = load_default_llm(logger)
        else:
            self.llm = llm
            logger.info("ExplanationAgent initialized with provided LLM")

    def run(
        self,
        ticker: str,
        prediction: Dict[str, Any],
        sentiment: Dict[str, Any],
        smart_money: Dict[str, Any],
        user_tier: str,
        confidence_level: str,
        similar_news: Optional[List[Dict[str, Any]]] = None,
        historical_analyses: Optional[List[Dict[str, Any]]] = None,
        news_statistics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate plain English explanation."""
        logger.info("      ExplanationAgent Execution")
        logger.info("         Ticker: %s", ticker)
        logger.info("         User tier: %s", user_tier)
        logger.info("         Confidence level: %s", confidence_level)
        logger.info("         Input data:")
        logger.info("            Prediction direction: %s", prediction.get("direction", "N/A"))
        logger.info("            Prediction confidence: %.1f%%", prediction.get("confidence", 0) * 100)
        logger.info(
            "            Sentiment: %s (%.2f)",
            sentiment.get("current", "N/A"),
            sentiment.get("score", 0),
        )
        logger.info("            Similar news articles: %d", len(similar_news) if similar_news else 0)
        logger.info("            Historical analyses: %d", len(historical_analyses) if historical_analyses else 0)
        logger.info("            News statistics available: %s", bool(news_statistics))

        if self.llm:
            logger.info("         Using LLM-based explanation generation")
            explanation = generate_llm_explanation(
                self.llm,
                ticker,
                prediction,
                sentiment,
                smart_money,
                user_tier,
                confidence_level,
                similar_news=similar_news,
                historical_analyses=historical_analyses,
                news_statistics=news_statistics,
                fallback_template=generate_template_explanation,
                log=logger,
            )
        else:
            logger.info("         Using template-based explanation generation")
            explanation = generate_template_explanation(
                ticker, prediction, sentiment, smart_money, user_tier, confidence_level
            )

        logger.info("      Explanation Result")
        logger.info("         Generated explanation: %d characters", len(explanation))
        logger.info("         Preview: %s...", explanation[:100])

        return explanation

