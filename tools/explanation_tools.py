"""Helper functions for explanation generation."""

from typing import Any, Callable, Dict, List, Optional
import logging
import os

logger = logging.getLogger(__name__)


def load_default_llm(log: Optional[logging.Logger] = None):
    """Create the default ChatOpenAI instance when available."""
    log = log or logger
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log.info("ExplanationAgent initialized without LLM (OPENAI_API_KEY not found)")
        return None

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, api_key=api_key)
        log.info("ExplanationAgent initialized with OpenAI gpt-3.5-turbo")
        return llm
    except Exception as exc:
        log.warning("Failed to initialize OpenAI LLM: %s. Using template-based generation.", exc)
        return None


def generate_template_explanation(
    ticker: str,
    prediction: Dict[str, Any],
    sentiment: Dict[str, Any],
    smart_money: Dict[str, Any],
    user_tier: str,
    confidence_level: str,
) -> str:
    """Generate a deterministic, template-based explanation."""
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


def generate_llm_explanation(
    llm: Any,
    ticker: str,
    prediction: Dict[str, Any],
    sentiment: Dict[str, Any],
    smart_money: Dict[str, Any],
    user_tier: str,
    confidence_level: str,
    similar_news: Optional[List[Dict[str, Any]]] = None,
    historical_analyses: Optional[List[Dict[str, Any]]] = None,
    news_statistics: Optional[Dict[str, Any]] = None,
    fallback_template: Optional[Callable[..., str]] = None,
    log: Optional[logging.Logger] = None,
) -> str:
    """Generate an LLM-based explanation with graceful fallback."""
    log = log or logger

    if llm is None:
        raise ValueError("LLM instance is required for LLM-based explanation generation")

    try:
        context_parts: List[str] = []

        if historical_analyses:
            recent_analysis = historical_analyses[0]
            context_parts.append(
                f"Previous analysis for {ticker} predicted {recent_analysis.get('prediction_direction', 'unknown')} "
                f"with {recent_analysis.get('prediction_confidence', 0):.1%} confidence. "
                f"Sentiment was {recent_analysis.get('sentiment_label', 'unknown')}."
            )

        if news_statistics:
            total_articles = news_statistics.get("total_articles", 0)
            avg_sentiment = news_statistics.get("avg_sentiment_score", 0.5)
            dist = news_statistics.get("sentiment_distribution", {})
            context_parts.append(
                f"News coverage: {total_articles} articles tracked. "
                f"Overall sentiment distribution: {dist.get('positive', 0)} positive, "
                f"{dist.get('neutral', 0)} neutral, {dist.get('negative', 0)} negative. "
                f"Average sentiment score: {avg_sentiment:.2f}."
            )

        if similar_news:
            top_similar = similar_news[:3]
            context_parts.append("Recent similar news headlines:")
            for i, article in enumerate(top_similar, 1):
                title = article.get("title", "Unknown")
                article_sentiment = article.get("sentiment", "neutral")
                context_parts.append(f"{i}. {title} (sentiment: {article_sentiment})")

        historical_context = "\n".join(context_parts) if context_parts else "No historical data available."

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

        response = llm.invoke(prompt)
        explanation = response.content if hasattr(response, "content") else str(response)

        if "disclaimer" not in explanation.lower() and "not investment advice" not in explanation.lower():
            explanation += (
                "\n\nDisclaimer: This analysis is for informational purposes only and does not constitute investment "
                "advice. Always conduct your own research before making investment decisions."
            )

        return explanation.strip()

    except Exception as exc:
        log.error("LLM explanation generation failed: %s", exc)
        if fallback_template:
            log.info("Falling back to template-based explanation")
            return fallback_template(
                ticker,
                prediction,
                sentiment,
                smart_money,
                user_tier,
                confidence_level,
            )
        raise

