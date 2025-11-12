"""Explanation Agent for generating narratives.

JOSH's Component - Explanation Agent
Generates plain English explanations using OpenAI (for team use only).
"""
import logging
import os
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ExplanationAgent:
    """Agent that generates plain English explanations."""

    def __init__(self, llm=None):
        """Initialize ExplanationAgent.

        Args:
            llm: Optional LangChain LLM for text generation
        """
        if llm is None:
            # Auto-initialize OpenAI if API key available
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    from langchain_openai import ChatOpenAI
                    self.llm = ChatOpenAI(
                        model="gpt-3.5-turbo",
                        temperature=0.7,
                        api_key=api_key
                    )
                    logger.info("ExplanationAgent initialized with OpenAI gpt-3.5-turbo")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI LLM: {e}. Using template-based generation.")
                    self.llm = None
            else:
                logger.info("ExplanationAgent initialized without LLM (OPENAI_API_KEY not found)")
                self.llm = None
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
        news_statistics: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate plain English explanation.

        Args:
            ticker: Stock symbol
            prediction: Prediction results
            sentiment: Sentiment results
            smart_money: Smart money data
            user_tier: User subscription level
            confidence_level: Overall confidence
            similar_news: Similar news from ChromaDB (semantic search results)
            historical_analyses: Historical analysis results from SQLite
            news_statistics: News statistics from ChromaDB for this ticker

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
        logger.info(f"            Similar news articles: {len(similar_news) if similar_news else 0}")
        logger.info(f"            Historical analyses: {len(historical_analyses) if historical_analyses else 0}")
        logger.info(f"            News statistics available: {bool(news_statistics)}")

        # If LLM available, use it
        if self.llm:
            logger.info(f"         → Using LLM-based explanation generation")
            explanation = self._generate_llm_explanation(
                ticker, prediction, sentiment, smart_money, user_tier, confidence_level,
                similar_news, historical_analyses, news_statistics
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
        confidence_level: str,
        similar_news: Optional[List[Dict[str, Any]]] = None,
        historical_analyses: Optional[List[Dict[str, Any]]] = None,
        news_statistics: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate LLM-based explanation using OpenAI gpt-3.5-turbo."""
        try:
            # Build context from ChromaDB and SQLite data
            context_parts = []

            # Add historical analysis context
            if historical_analyses and len(historical_analyses) > 0:
                recent_analysis = historical_analyses[0]
                context_parts.append(
                    f"Previous analysis for {ticker} predicted {recent_analysis.get('prediction_direction', 'unknown')} "
                    f"with {recent_analysis.get('prediction_confidence', 0):.1%} confidence. "
                    f"Sentiment was {recent_analysis.get('sentiment_label', 'unknown')}."
                )

            # Add news statistics context
            if news_statistics:
                total_articles = news_statistics.get('total_articles', 0)
                avg_sentiment = news_statistics.get('avg_sentiment_score', 0.5)
                dist = news_statistics.get('sentiment_distribution', {})
                context_parts.append(
                    f"News coverage: {total_articles} articles tracked. "
                    f"Overall sentiment distribution: {dist.get('positive', 0)} positive, "
                    f"{dist.get('neutral', 0)} neutral, {dist.get('negative', 0)} negative. "
                    f"Average sentiment score: {avg_sentiment:.2f}."
                )

            # Add similar news context
            if similar_news and len(similar_news) > 0:
                top_similar = similar_news[:3]
                context_parts.append("Recent similar news headlines:")
                for i, article in enumerate(top_similar, 1):
                    title = article.get('title', 'Unknown')
                    article_sentiment = article.get('sentiment', 'neutral')
                    context_parts.append(f"{i}. {title} (sentiment: {article_sentiment})")

            historical_context = "\n".join(context_parts) if context_parts else "No historical data available."

            # Construct prompt for OpenAI
            prompt = f"""You are a financial analyst explaining stock analysis results to investors.

Stock: {ticker}
User Tier: {user_tier}
Confidence Level: {confidence_level.upper()}

Current Analysis:
- Prediction: {prediction.get('direction', 'neutral').upper()} with {prediction.get('confidence', 0.5):.1%} confidence
- Model: {prediction.get('metadata', {}).get('model', 'unknown')}
- Sentiment: {sentiment.get('current', 'neutral').upper()} (score: {sentiment.get('score', 0.5):.2f})
- Trend: {sentiment.get('trend', 'stable')}

Historical Context:
{historical_context}

Generate a clear, professional explanation (3-4 paragraphs) that:
1. Summarizes the current prediction and sentiment
2. References historical context and news trends when available
3. Explains what this means for investors
4. Includes appropriate disclaimers

Keep the tone professional but accessible. Do not use bullet points."""

            # Call OpenAI via LangChain
            response = self.llm.invoke(prompt)

            # Extract content from response
            if hasattr(response, 'content'):
                explanation = response.content
            else:
                explanation = str(response)

            # Add disclaimer if not present
            if "disclaimer" not in explanation.lower() and "not investment advice" not in explanation.lower():
                explanation += "\n\nDisclaimer: This analysis is for informational purposes only and does not constitute investment advice. Always conduct your own research before making investment decisions."

            return explanation.strip()

        except Exception as e:
            logger.error(f"LLM explanation generation failed: {e}")
            logger.info("Falling back to template-based explanation")
            return self._generate_template_explanation(
                ticker, prediction, sentiment, smart_money, user_tier, confidence_level
            )
