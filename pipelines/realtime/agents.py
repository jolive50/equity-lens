from pydantic import SecretStr  # For secure API key handling

import logging  # For recording events and debugging
import re  # For pattern matching in text parsing
from dataclasses import dataclass  # For creating simple data classes
from typing import Dict, List, Literal, Optional, Any  # Type hints

# LangGraph runtime utilities (provided via langchain-core components)
# Why: Provides tools to chain together prompts, LLMs, and parsers
from langchain_core.output_parsers import StrOutputParser  # Converts LLM output to string
from langchain_core.prompts import ChatPromptTemplate  # Creates formatted prompts for LLMs
from langchain_core.runnables import Runnable, RunnableLambda  # Base classes for chainable components


from .api_keys import get_api_key, get_available_api_keys  # Centralised API key helpers
from .reflection import ReflectionAgent  # Quality assurance agent for output validation
from .storage.vector_store import VectorStore  # Import VectorStore for type hints

logger = logging.getLogger(__name__)

"""LangGraph agent definitions for the StockSense workflow.

This file defines specialized AI "agents" that each perform specific analysis tasks.
Think of agents as expert team members: one does predictions, one analyzes news,
one tracks big investors, one validates quality, etc.

What this file does:
- Defines AI agent classes (PredictionAgent, SentimentAgent, ReflectionAgent, etc.)
- Each agent has specialized logic for its task using deterministic ML pipelines
- Provides helper functions to parse structured outputs
- Imports ReflectionAgent for quality assurance and output validation

Why we need agents:
- Separates concerns (each agent has one job - Single Responsibility)
- Reusable across different workflows
- Easy to test independently
- ReflectionAgent ensures quality control before user sees results
"""

import logging  # For recording events and debugging
import re  # For pattern matching in text parsing
from dataclasses import dataclass  # For creating simple data classes
from typing import Dict, List, Literal, Optional, Any  # Type hints

# LangGraph runtime utilities (provided via langchain-core components)
# Why: Provides tools to chain together prompts, LLMs, and parsers
from langchain_core.output_parsers import StrOutputParser  # Converts LLM output to string
from langchain_core.prompts import ChatPromptTemplate  # Creates formatted prompts for LLMs
from langchain_core.runnables import Runnable, RunnableLambda  # Base classes for chainable components

from .api_keys import get_api_key, get_available_api_keys  # Centralised API key helpers
from .reflection import ReflectionAgent  # Quality assurance agent for output validation

logger = logging.getLogger(__name__)


# ===== DATA STRUCTURES =====
# These dataclasses define the shape of data returned by agents
# Why dataclasses: Clean, typed, auto-generates __init__, __repr__, etc.

@dataclass
class PredictionResult:
    """Structured output from the prediction agent.

    What: Container for stock direction prediction
    Why: Ensures consistent format across workflow
    Data: direction (up/down/neutral), confidence (0-1), explanation text,
          plus optional probability curve and horizon metadata when the ML model is available
    """
    direction: Literal["up", "down", "neutral"]  # Must be one of these three
    confidence: float  # 0.0 (not confident) to 1.0 (very confident)
    narrative: str  # English explanation of why prediction was made
    daily_probs: Optional[List[Dict[str, Any]]] = None  # Full probability curve (if provided)
    horizon_95: Optional[Dict[str, Any]] = None  # 95% confidence horizon metadata
    feature_importance: Optional[Dict[str, float]] = None  # Optional model feature impact scores


@dataclass
class SentimentResult:
    """Structured output from the sentiment agent.

    What: Container for news sentiment analysis
    Why: Standardizes sentiment data format
    Data: current sentiment, score, trend direction, top headlines
    """
    current: Literal["positive", "neutral", "negative"]  # Overall sentiment
    score: float  # Strength of sentiment (0.0 to 1.0)
    trend: Literal["improving", "stable", "declining"]  # Is sentiment getting better or worse?
    headlines: List[str]  # Top 3 relevant news headlines


@dataclass
class SmartMoneyResult:
    """Structured output from the smart money agent.

    What: Container for institutional investor activity
    Why: Tracks what "smart money" (big investors) are doing
    Data: Three categories - institutions, insiders, congressional trades
    """
    institutions: Dict[str, Any]  # Hedge funds, mutual funds activity
    insiders: Dict[str, Any]  # Company executives buying/selling
    congress: Dict[str, Any]  # Congressional stock trades (public disclosures)


# ===== CORE AGENT CLASSES =====

class PredictionAgent:
    """Generates probabilistic directional forecasts for stock symbols using the gradient boosting model.

    Data Flow:
    Input: ticker (str), market_data (price/volume), fundamentals (P/E, revenue, etc.)
    Output: PredictionResult (direction, confidence, narrative)

    Example:
    Input: ticker="AAPL", market_data=[{close: 150, volume: 1_000_000}, ...], fundamentals={pe_ratio: 28}
    Output: PredictionResult(direction="up", confidence=0.85, narrative="Gradient boosting model indicates...")
    """

    def __init__(
        self,
        llm: Optional[Runnable] = None,  # Can now be used for explanations
        *,
        use_ml_model: bool = True,  # Parameter retained for signature compatibility
        use_gpt_explanations: bool = False  # Enable GPT-based explanations with SHAP
    ) -> None:
        """Initialize the prediction agent.

        WHAT: Sets up forecaster and optional GPT explanation generator
        WHY: ML for predictions, GPT for human-readable explanations
        HOW: Load LSTM forecaster, optionally configure OpenAI LLM
        DATA: Creates forecaster instance, stores LLM reference

        Args:
            llm: Optional LLM for generating explanations (OpenAI recommended)
            use_ml_model: Always True, kept for compatibility
            use_gpt_explanations: If True, use GPT to explain predictions with SHAP
        """
        if not use_ml_model:
            logger.warning("PredictionAgent enforces ML-only mode; ignoring use_ml_model=False request.")

        try:
            from .models.forecaster import create_forecaster
        except ImportError as exc:
            raise RuntimeError(
                "Probabilistic forecaster dependencies are missing. "
                "Install the required model packages before running StockSense."
            ) from exc

        # WHAT: Initialize LSTM forecaster
        # WHY: ML model for actual predictions
        self.forecaster = create_forecaster("lstm")
        if self.forecaster is None:
            raise RuntimeError("create_forecaster returned None; ensure the forecaster is configured correctly.")

        # WHAT: Store LLM for explanations
        # WHY: GPT can generate high school level explanations
        # HOW: Store reference, use only if enabled
        self.llm = llm
        self.use_gpt_explanations = use_gpt_explanations

        # WHAT: Log configuration
        if use_gpt_explanations and llm is not None:
            logger.info("PredictionAgent: GPT explanations enabled with SHAP feature importance")
        else:
            logger.info("PredictionAgent: Using basic narrative (GPT disabled)")

    def _build_narrative(self, ticker: str, ml_result) -> str:
        """Compose narrative explaining the ML prediction.

        WHAT: Generates human-readable explanation of forecast
        WHY: Users need to understand why model made its prediction
        HOW: Use GPT with SHAP insights if enabled, else basic summary
        DATA: ML result + SHAP -> narrative string

        Args:
            ticker: Stock symbol
            ml_result: ForecastResult from LSTM model

        Returns:
            Narrative string explaining the prediction
        """
        # WHAT: Check if GPT explanations enabled
        # WHY: Use GPT for better explanations if available
        if self.use_gpt_explanations and self.llm is not None:
            return self._build_gpt_narrative(ticker, ml_result)
        else:
            return self._build_basic_narrative(ticker, ml_result)

    def _build_basic_narrative(self, ticker: str, ml_result) -> str:
        """Build basic narrative without GPT.

        WHAT: Creates simple string-based explanation
        WHY: Fallback when GPT not available
        HOW: String concatenation with key metrics
        DATA: ML result -> formatted string
        """
        probabilities = ml_result.model_metadata.get("probabilities", {}) if ml_result.model_metadata else {}
        prob_up = probabilities.get("up", 0.0)
        prob_down = probabilities.get("down", 0.0)
        prob_neutral = probabilities.get("neutral", 0.0)

        feature_importance = ml_result.feature_importance or {}
        top_features = sorted(
            feature_importance.items(),
            key=lambda item: abs(item[1]),
            reverse=True
        )[:3]
        feature_summary = ", ".join(f"{name} ({weight:.2f})" for name, weight in top_features) if top_features else "insufficient feature data"

        days = ml_result.horizon_95.get("days", 0) if ml_result.horizon_95 else 0

        return (
            f"{ticker.upper()} forecast: {ml_result.direction.upper()} with "
            f"{ml_result.confidence:.1%} confidence using the LSTM forecaster. "
            f"Probability distribution - up: {prob_up:.1%}, down: {prob_down:.1%}, neutral: {prob_neutral:.1%}. "
            f"95% confidence horizon: {days} day(s). "
            f"Top contributing features: {feature_summary}."
        )

    def _build_gpt_narrative(self, ticker: str, ml_result) -> str:
        """Build GPT-powered narrative with SHAP explanations.

        WHAT: Generates high school level explanation using GPT
        WHY: GPT can create more natural, educational explanations
        HOW: Prompt GPT with prediction + SHAP feature importance
        DATA: ML result + SHAP -> GPT prompt -> narrative

        Args:
            ticker: Stock symbol
            ml_result: ForecastResult with SHAP feature importance

        Returns:
            Paragraph-long explanation from GPT
        """
        # WHAT: Extract prediction details
        probabilities = ml_result.model_metadata.get("probabilities", {}) if ml_result.model_metadata else {}
        prob_up = probabilities.get("up", 0.0)
        prob_down = probabilities.get("down", 0.0)
        prob_neutral = probabilities.get("neutral", 0.0)

        # WHAT: Get SHAP feature importance
        # WHY: Explain which features drove the prediction
        feature_importance = ml_result.feature_importance or {}
        top_features = sorted(
            feature_importance.items(),
            key=lambda item: abs(item[1]),
            reverse=True
        )[:5]  # Top 5 features

        # WHAT: Format features for prompt
        features_text = "\n".join(
            f"- {name}: {weight:.1%} importance"
            for name, weight in top_features
        ) if top_features else "No specific features identified"

        # WHAT: Build GPT prompt
        # WHY: Need clear instructions for high school level explanation
        # HOW: Provide prediction, probabilities, and SHAP insights
        prompt = f"""You are explaining a stock price prediction to a high school student. Write a clear, educational paragraph explaining the prediction.

Stock: {ticker.upper()}
Prediction: {ml_result.direction.upper()}
Confidence: {ml_result.confidence:.1%}

Probabilities:
- Up: {prob_up:.1%}
- Down: {prob_down:.1%}
- Steady: {prob_neutral:.1%}

Most Important Features (from SHAP analysis):
{features_text}

Write a single paragraph (4-6 sentences) that:
1. States the prediction and confidence clearly
2. Explains what the top features mean in simple terms
3. Describes why these features suggest this direction
4. Uses analogies or examples a high school student would understand

Do not use jargon like "LSTM", "SHAP", or "gradient boosting". Instead say "our AI model" or "the prediction system".
Be conversational but accurate. Focus on helping the student understand WHY the model made this prediction."""

        try:
            # Invoke GPT to generate explanation
            response = self.llm.invoke(prompt)

            # Extract text from response
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)

        except Exception as e:
            # Fall back to basic narrative on error
            logger.warning(f"GPT explanation failed: {e}, using basic narrative")
            return self._build_basic_narrative(ticker, ml_result)

    def run(self, *, ticker: str, market_data: Dict, fundamentals: Dict[str, float]) -> PredictionResult:
        """Make a prediction for a stock using the gradient boosting forecaster."""

        if not isinstance(market_data, list):
            raise ValueError("market_data must be a list of daily price records.")

        try:
            ml_result = self.forecaster.predict(market_data, fundamentals)
        except Exception as exc:
            raise RuntimeError(f"Forecasting ML model failed for {ticker}: {exc}") from exc

        from typing import cast
        allowed_directions = {"up", "down", "neutral"}
        direction = ml_result.direction if ml_result.direction in allowed_directions else "neutral"
        direction_literal = cast(Literal["up", "down", "neutral"], direction)
        return PredictionResult(
            direction=direction_literal,
            confidence=ml_result.confidence,
            narrative=self._build_narrative(ticker, ml_result),
            daily_probs=ml_result.daily_probs,
            horizon_95=ml_result.horizon_95,
            feature_importance=ml_result.feature_importance
        )





class SentimentAgent:
    """Analyzes news sentiment and social media for stock symbols.

    What: Determines if news about a stock is positive, negative, or neutral
    Why: News sentiment often drives short-term price movements
    How: Uses FinBERT (specialized financial sentiment AI)

    Data Flow:
    Input: ticker (str), news_data (list of article dictionaries)
    Output: SentimentResult (current sentiment, score, trend, headlines, similar_articles)

    Example:
    Input: ticker="AAPL", news_data=[{title: "Apple beats earnings", content: "..."}]
    Output: SentimentResult(current="positive", score=0.85, trend="improving", headlines=[...])
    """

    def __init__(
        self,
        llm: Optional[Runnable] = None,  # Deprecated parameter retained for compatibility
        *,
        use_finbert: bool = True,
        vector_store: Optional[any] = None,  # VectorStore integration point
    ) -> None:
        """Initialize the sentiment agent in FinBERT-only mode with optional VectorStore.

        What: Sets up FinBERT sentiment analyzer with optional VectorStore integration
        Why: VectorStore enables semantic search and historical context for better analysis
        How: Creates sentiment analyzer and passes VectorStore reference if provided

        Args:
            llm: (Unused) Language model, retained for compatibility
            use_finbert: Whether to use FinBERT (must be True)
            vector_store: Optional VectorStore instance for semantic search and storage

        VectorStore integration points:
            - Store processed news articles for semantic search
            - Retrieve similar news for context augmentation
            - Enhance sentiment analysis with historical context
        """
        if llm is not None:
            logger.warning("SentimentAgent ignores provided LLM; operating with FinBERT only.")

        if not use_finbert:
            logger.warning("SentimentAgent requires FinBERT; ignoring use_finbert=False request.")

        try:
            from .sentiment.finbert import create_sentiment_analyzer
        except ImportError as exc:
            raise RuntimeError(
                "FinBERT dependencies are missing. Install tensorflow/transformers and download the model "
                "before running StockSense."
            ) from exc

        # WHAT: Create sentiment analyzer with optional VectorStore
        # WHY: VectorStore enables storing articles and searching similar historical patterns
        # HOW: Pass vector_store to factory function
        # DATA: vector_store (optional) → sentiment_analyzer with enhanced capabilities
        self.sentiment_analyzer = create_sentiment_analyzer(
            vector_store=vector_store,
            enable_similarity_search=True
        )
        if self.sentiment_analyzer is None:
            raise RuntimeError("create_sentiment_analyzer returned None; ensure FinBERT assets are available.")

        # WHAT: Store VectorStore reference for find_related_news method
        # WHY: Allows direct semantic search on stored articles
        # HOW: Store reference as instance variable
        # DATA: vector_store → self.vector_store
        self.vector_store = vector_store

    def run(self, *, ticker: str, news_data: List[Dict]) -> SentimentResult:
        """
        Analyze sentiment for a stock's news using FinBERT.
        Stores news articles in VectorStore if available.
        Augments sentiment analysis with historical context from VectorStore.
        """
        # WHAT: Validate news_data is not empty
        # WHY: FinBERT requires at least one article to analyze
        # HOW: Check if news_data list has items
        # DATA: news_data (List[Dict]) must be non-empty
        if not news_data:
            raise ValueError("news_data must contain at least one article for sentiment analysis.")

        # WHAT: Process news articles with FinBERT and VectorStore integration
        # WHY: Get sentiment analysis with optional historical context from VectorStore
        # HOW: Pass ticker to enable VectorStore storage and similarity search
        # DATA: news_data + ticker → FinBERT analysis + VectorStore storage → sentiment result with similar_articles
        try:
            finbert_result = self.sentiment_analyzer.process_news_articles(
                news_data,
                ticker=ticker
            )
        except Exception as exc:
            raise RuntimeError(f"FinBERT sentiment analysis failed for {ticker}: {exc}") from exc

        # WHAT: Validate finbert_result has required keys
        # WHY: Ensure FinBERT returned valid sentiment analysis
        # HOW: Check for required keys in result dictionary
        # DATA: finbert_result must have {current, score, trend, headlines}
        required_keys = {"current", "score", "trend", "headlines"}
        if not isinstance(finbert_result, dict) or not required_keys.issubset(finbert_result.keys()):
            raise RuntimeError("FinBERT returned an unexpected response structure.")

        # WHAT: Extract and validate sentiment values from FinBERT result
        # WHY: Type safety and ensure values are in expected range
        # HOW: Cast to expected types and validate against allowed values
        # DATA: finbert_result dict → typed sentiment values
        allowed_sentiments = {"positive", "neutral", "negative"}
        allowed_trends = {"improving", "stable", "declining"}
        current = str(finbert_result["current"]).lower()
        trend = str(finbert_result["trend"]).lower()
        current = current if current in allowed_sentiments else "neutral"
        trend = trend if trend in allowed_trends else "stable"
        score = float(finbert_result["score"])
        headlines = list(finbert_result.get("headlines", []))[:3]

        # WHAT: Cast to Literal types for type safety
        # WHY: mypy and type checkers require exact literal types
        # HOW: Use typing.cast with Literal types
        # DATA: str → Literal["positive" | "neutral" | "negative"]
        from typing import cast
        current_literal = cast(Literal["positive", "neutral", "negative"], current)
        trend_literal = cast(Literal["improving", "stable", "declining"], trend)

        # WHAT: Return SentimentResult with all analysis data
        # WHY: Provides structured sentiment information to coordination agent
        # HOW: Create SentimentResult with validated fields
        # DATA: sentiment fields → SentimentResult dataclass
        return SentimentResult(
            current=current_literal,
            score=score,
            trend=trend_literal,
            headlines=headlines
        )


class SmartMoneyAgent:
    """Tracks institutional, insider, and congressional activity via configured data services."""

    def __init__(
        self,
        llm: Optional[Runnable] = None,  # Deprecated parameter retained for compatibility
        *,
        use_data_service: bool = True,
    ) -> None:
        if llm is not None:
            logger.warning("SmartMoneyAgent ignores provided LLM; operating with data services only.")

        if not use_data_service:
            raise ValueError("SmartMoneyAgent requires the data service to be enabled.")

        try:
            from .smart_money import create_smart_money_service
        except ImportError as exc:
            raise RuntimeError("Smart money service dependencies are missing. Install required packages before running StockSense.") from exc

        api_keys = get_available_api_keys("alpha_vantage", "finnhub")
        self.smart_money_service = create_smart_money_service(api_keys)
        if self.smart_money_service is None:
            raise RuntimeError("Smart money service could not be initialised; verify API keys and SEC data configuration.")

    def run(self, *, ticker: str) -> SmartMoneyResult:
        """Analyze smart money activity for a stock using deterministic data sources."""

        try:
            institutional = self.smart_money_service.get_institutional_summary(ticker)
            insider = self.smart_money_service.get_insider_summary(ticker)
            congressional = self.smart_money_service.get_congressional_summary(ticker)
        except Exception as exc:
            raise RuntimeError(f"Smart money service failed for {ticker}: {exc}") from exc

        return SmartMoneyResult(
            institutions=institutional,
            insiders=insider,
            congress=congressional
        )


class ExplanationAgent:
    """Generates plain English explanations for retail investors.

    What: Translates technical analysis into simple language anyone can understand
    Why: Users want to know WHY a prediction was made, not just the result
    How: Uses LLM to synthesize all analysis into coherent explanation

    Data Flow:
    Input: ticker, prediction result, sentiment result, smart money result, user tier
    Output: Plain English explanation string (3-4 paragraphs)

    Example:
    Input: All analysis results for AAPL
    Output: "Based on our analysis, AAPL shows strong upward momentum with 85% confidence.
             Recent news sentiment is very positive, driven by strong earnings reports..."
    """

    def __init__(
        self,
        llm: Runnable,  # Language model for generating explanations
        *,
        prompt: Optional[ChatPromptTemplate] = None,  # Custom prompt
    ) -> None:
        """Initialize the explanation agent.

        What: Sets up LLM chain for generating explanations
        How: Creates prompt template, connects to LLM
        Why: Explanations require natural language generation (LLM's strength)
        """

        # Create prompt template for explanations
        # This tells the LLM HOW to explain analysis to users
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

        # Create LLM chain: prompt → LLM → parser
        self._chain = base_prompt | llm | StrOutputParser()

    def run(
        self,
        *,
        ticker: str,  # Stock symbol
        prediction: Dict,  # Prediction results
        sentiment: Dict,  # Sentiment analysis results
        smart_money: Dict,  # Smart money data
        user_tier: str,  # User subscription level (basic/registered/premium)
        confidence_level: str  # Overall confidence (high/medium/low)
    ) -> str:
        """Generate plain English explanation.

        What: Creates human-readable explanation of all analysis
        How: Sends all results to LLM with instruction prompt, returns explanation
        Why: Users need to understand the "why" behind predictions

        Data Flow:
        Receives: All analysis results (prediction, sentiment, smart money)
        Returns: String with 3-4 paragraph explanation

        Example:
        Input: All analysis results for AAPL showing upward direction, positive sentiment
        Output: "Based on our analysis, AAPL is likely to move upward with 85% confidence.
                Recent news sentiment is strongly positive, driven by earnings beats and
                new product announcements. Institutional investors have increased their
                positions by 3% this quarter, signaling confidence in the company's direction.
                However, please note that this is informational analysis, not investment advice..."
        """

        # Invoke LLM chain with all data
        # LLM will synthesize everything into coherent explanation
        return self._chain.invoke({
            "ticker": ticker,
            "prediction": str(prediction),  # Convert to string for LLM
            "sentiment": str(sentiment),
            "smart_money": str(smart_money),
            "user_tier": user_tier,  # Affects what info to include (basic vs premium)
            "confidence_level": confidence_level  # Affects how cautious explanation should be
        }).strip()  # Remove leading/trailing whitespace


# ===== LLM BUILDER FUNCTIONS =====
# These create LLM instances for agents to use

def build_openai_llm(model: str = "gpt-4o-mini") -> Runnable:
    """Build OpenAI LLM using the LangGraph runtime.

    What: Creates connection to OpenAI's API
    Why: Agents need an LLM for text generation
    How: Uses the ChatOpenAI wrapper exposed through langchain-openai

    Data Flow:
    Input: model name (default "gpt-4o-mini" - cost-effective GPT-4 variant)
    Output: Runnable LLM instance

    Raises:
        ValueError: If OPENAI_API_KEY not set in environment
    """
    from langchain_openai import ChatOpenAI  # OpenAI integration

    # What: Pull the OpenAI API key from the central registry (raises if missing when required)
    # Why: Guarantees we never attempt to hit OpenAI without credentials, keeping usage compliant
    # How: Delegate to api_keys.get_api_key with required=True so a clear exception is raised when absent
    # Data: Returns the API key string trimmed of whitespace

    api_key_str = get_api_key("openai", required=True)
    if api_key_str is None:
        raise ValueError("OPENAI_API_KEY is required but not found.")
    api_key = SecretStr(api_key_str)

    # Create and return ChatOpenAI instance
    return ChatOpenAI(
        model=model,  # Which GPT model to use
        api_key=api_key,  # Authentication as SecretStr
        temperature=0.1  # Low temperature for consistent, conservative responses
                         # Why 0.1: Financial analysis should be consistent, not creative
    )


def build_real_llm_agent(tag: str, llm_model: Runnable) -> Runnable:
    """Utility to build a real LLM agent that processes actual prompts.

    What: Wraps an LLM model to handle different prompt formats
    Why: LangGraph prompts can have different formats, need unified handler
    How: Creates a function that extracts content and invokes LLM

    Data Flow:
    Input: tag (identifier for logging), llm_model (LLM instance)
    Output: Runnable that can process various prompt formats

    Note: This is a utility function, not used directly by main workflow
    """

    def _process(prompt) -> str:
        """Internal function that processes prompts.

        What: Handles different prompt formats and invokes LLM
        How: Checks prompt type, extracts content, calls LLM, returns text
        Why: Different parts of code might pass prompts in different formats
        """
        try:
            # Case 1: Prompt has messages attribute (ChatPromptTemplate format)
            if hasattr(prompt, 'messages') and prompt.messages:
                # Extract actual content and process with LLM
                result = llm_model.invoke(prompt)

                # LLM might return object with .content attribute or just string
                if hasattr(result, 'content'):
                    return result.content  # Return text content
                else:
                    return str(result)  # Convert to string

            # Case 2: Prompt is already a string
            elif isinstance(prompt, str):
                # Create simple chat template from string
                chat_prompt = ChatPromptTemplate.from_template("{input}")
                formatted_prompt = chat_prompt.format(input=prompt)
                result = llm_model.invoke(formatted_prompt)
                return result.content if hasattr(result, 'content') else str(result)

            # Case 3: Unknown format, try converting to string
            else:
                result = llm_model.invoke(str(prompt))
                return result.content if hasattr(result, 'content') else str(result)

        except Exception as e:
            # If anything goes wrong, log error and return error message
            logger.error(f"[{tag}] Error processing prompt: {e}")
            return f"[{tag}] Processing error: {str(e)[:100]}..."  # Truncate to 100 chars

    # Wrap function in RunnableLambda to make it chainable
    # Why: LangGraph components need to be Runnable to work with pipes (|)
    return RunnableLambda(_process)


# ===== ENHANCED WORKFLOW AGENTS =====
# These are specialized agents for multi-company analysis

class CoordinationAgent:
    """Coordinates multiple analysis agents and synthesizes insights.

    What: Manager agent that combines results from other agents
    Why: When analyzing multiple stocks, need to synthesize overall picture
    How: Receives results from working agents, calculates overall confidence, makes recommendations

    Data Flow:
    Input: Results from HistoricalAnalysisAgent and SentimentAnalysisAgent
    Output: Synthesized analysis with overall confidence and recommendation

    Key Feature: Only recommends action if confidence >= 95% (per requirements)
    """

    def __init__(
        self,
        llm: Runnable,  # Language model for synthesis
        *,
        confidence_threshold: float = 0.95,  # Minimum confidence to recommend action
    ) -> None:
        """Initialize coordination agent.

        What: Sets up agent with confidence threshold
        Why: Requirements specify 95% confidence threshold for recommendations
        How: Creates prompt template and stores threshold
        """
        # Store confidence threshold (default 95% per requirements)
        self.confidence_threshold = confidence_threshold

        # Create prompt for coordination and synthesis
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

        # Create LLM chain
        self._chain = base_prompt | llm | StrOutputParser()

    def coordinate(
        self,
        *,
        tickers: List[str],  # List of stock symbols being analyzed
        working_agent_results: Dict[str, Any]  # Results from other agents
    ) -> Dict[str, Any]:
        """Coordinate analysis across multiple agents and tickers.

        What: Synthesizes results from multiple working agents
        How:
        1. Extract confidence scores from working agents
        2. Calculate weighted overall confidence (60% historical, 40% sentiment)
        3. Check if confidence meets 95% threshold
        4. If yes: generate full analysis, If no: return insufficient confidence
        5. Return structured result

        Data Flow:
        Receives: working_agent_results = {
            "historical_confidence": 0.85,
            "sentiment_confidence": 0.78,
            "historical_analysis": {...},
            "sentiment_analysis": {...}
        }
        Returns: {
            "confidence": 0.82,
            "threshold_met": False,
            "recommendation": "INSUFFICIENT_CONFIDENCE",
            ...
        }
        """

        # === STEP 1: Calculate Overall Confidence ===
        # Weighted combination: 60% historical, 40% sentiment
        # Why: Historical fundamentals are more reliable than sentiment
        overall_confidence = 0.0

        # Add historical component (if available)
        if 'historical_confidence' in working_agent_results:
            overall_confidence += working_agent_results['historical_confidence'] * 0.6  # 60% weight

        # Add sentiment component (if available)
        if 'sentiment_confidence' in working_agent_results:
            overall_confidence += working_agent_results['sentiment_confidence'] * 0.4  # 40% weight

        # === STEP 2: Check Confidence Threshold ===
        # Only recommend action if >= 95% confident (per requirements)
        if overall_confidence < self.confidence_threshold:
            # Confidence too low, return early without recommendation
            return {
                "confidence": overall_confidence,  # Actual confidence level
                "recommendation": "INSUFFICIENT_CONFIDENCE",  # Don't recommend action
                "threshold_met": False,  # Flag that threshold not met
                "message": f"Confidence {overall_confidence:.3f} below threshold {self.confidence_threshold}"
            }

        # === STEP 3: Generate Full Analysis ===
        # If we get here, confidence is high enough
        # Ask LLM to synthesize all working agent results
        raw_output = self._chain.invoke({
            "working_agent_results": str(working_agent_results),
            "confidence_threshold": self.confidence_threshold
        })

        # Return comprehensive analysis
        return {
            "confidence": overall_confidence,  # Overall confidence score
            "threshold_met": True,  # Flag that threshold was met
            "raw_analysis": raw_output,  # Full LLM synthesis
            "_extract_coordination_data": raw_output  # Raw text for further parsing
        }


class HistoricalAnalysisAgent:
    """Analyzes historical data and fundamental metrics for S&P 500 companies.

    What: Working agent that analyzes historical trends and fundamentals
    Why: Need to understand company performance over time, not just current snapshot
    How: Combines LLM analysis with quantitative metric calculations

    Data Flow:
    Input: comprehensive_data (market data + fundamentals for multiple companies)
    Output: Analysis with confidence score, top performers, concerns

    Key Metrics:
    - Revenue Growth Rate
    - EBITDA Margin
    - P/E Ratio
    - ROE (Return on Equity)
    - Price Momentum
    """

    def __init__(
        self,
        llm: Runnable,  # Language model for qualitative analysis
        *,
        use_fundamental_analysis: bool = True,  # Whether to include fundamental metrics
    ) -> None:
        """Initialize historical analysis agent.

        What: Sets up agent for fundamental and technical analysis
        How: Creates prompt template for multi-company analysis
        Why: Need specialized prompt for analyzing multiple stocks simultaneously
        """
        self.use_fundamental_analysis = use_fundamental_analysis

        # Create prompt template for historical analysis
        # This guides LLM on what metrics to focus on
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

        # Create LLM chain
        self._chain = base_prompt | llm | StrOutputParser()

    def run(self, *, comprehensive_data: Dict[str, Dict], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run historical analysis across multiple companies.

        What: Analyzes historical trends and fundamentals for multiple stocks
        How:
        1. Send data to LLM for qualitative analysis
        2. Run quantitative analysis on metrics
        3. Combine LLM and quantitative confidence scores
        4. Return comprehensive analysis

        Data Flow:
        Receives: comprehensive_data = {
            "AAPL": {
                "fundamentals": {revenue_growth: 0.08, pe_ratio: 28, ...},
                "market_data": [{close: 150, volume: 1000000}, ...]
            },
            "MSFT": {...},
            ...
        }
        Returns: {
            "overall_trend": "positive",
            "technical_score": 75,
            "fundamental_score": 82,
            "confidence": 0.78,
            "top_performers": ["AAPL", "MSFT"],
            "concern_companies": ["XYZ"],
            ...
        }
        """

        # === STEP 1: LLM Qualitative Analysis ===
        # Ask LLM to analyze the data
        raw_output = self._chain.invoke({
            "comprehensive_data": str(comprehensive_data),
            "market_data": str(market_data)
        })

        # === STEP 2: Parse LLM Output ===
        # Extract structured fields from LLM's text response
        overall_trend = _extract_historical_trend(raw_output)  # "positive", "neutral", "negative"
        technical_score = _extract_score(raw_output, "TECHNICAL_SCORE")  # 0-100
        fundamental_score = _extract_score(raw_output, "FUNDAMENTAL_SCORE")  # 0-100
        confidence = _extract_confidence(raw_output, default=0.5)  # 0.0-1.0
        narrative = _extract_narrative(raw_output)  # Explanation text
        top_performers = _extract_ticker_list(raw_output, "TOP_PERFORMERS")  # ["AAPL", ...]
        concern_companies = _extract_ticker_list(raw_output, "CONCERN_COMPANIES")  # ["XYZ", ...]

        # === STEP 3: Quantitative Analysis ===
        # Run our own calculations on the fundamental metrics
        # Why: Don't rely solely on LLM, verify with real math
        market_trend_analysis = self._analyze_market_trends(comprehensive_data)

        # === STEP 4: Combine Confidence Scores ===
        # Blend LLM confidence (40%) with quantitative confidence (60%)
        # Why: Quantitative metrics are more reliable than LLM opinions
        combined_confidence = self._combine_confidence_scores(
            confidence,  # LLM's confidence
            market_trend_analysis["confidence"]  # Quantitative confidence
        )

        # === STEP 5: Return Comprehensive Analysis ===
        return {
            "overall_trend": overall_trend,
            "technical_score": technical_score,
            "fundamental_score": fundamental_score,
            "confidence": combined_confidence,  # Combined score
            "narrative": narrative,
            "top_performers": top_performers,
            "concern_companies": concern_companies,
            "historical_confidence": combined_confidence,  # For coordination agent
            "market_trends": market_trend_analysis  # Include quantitative trends
        }

    def _analyze_market_trends(self, comprehensive_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze market trends using quantitative fundamental metrics.

        What: Calculates sector-wide averages and trends from real metrics
        Why: Math doesn't lie - want to verify LLM analysis with calculations
        How:
        1. Collect metrics from all companies
        2. Calculate sector averages
        3. Determine trend directions
        4. Calculate confidence based on trend strength

        Data Flow:
        Receives: comprehensive_data with fundamentals for each ticker
        Returns: {
            "avg_revenue_growth": 0.06,
            "revenue_trend": "positive",
            "profitability_trend": "strong",
            "confidence": 0.75,
            ...
        }
        """
        import statistics  # For calculating means

        # Initialize metric collectors
        # These will hold all companies' metrics for aggregation
        sector_metrics = {
            "revenue_growth": [],  # List of all revenue growth rates
            "ebitda_margin": [],  # List of all EBITDA margins
            "pe_ratios": [],  # List of all P/E ratios
            "debt_to_equity": [],  # List of all debt/equity ratios
            "roe": [],  # List of all ROE values
            "price_momentum": []  # List of all price momentum values
        }

        ticker_fundamentals = {}  # Store fundamentals by ticker

        # === STEP 1: Aggregate Metrics Across All Companies ===
        # Loop through each company's data
        for ticker, data in comprehensive_data.items():
            fundamentals = data["fundamentals"]
            ticker_fundamentals[ticker] = fundamentals

            # Collect each metric (if available)
            # Why check existence: Not all companies have all metrics

            if "revenue_growth" in fundamentals:
                sector_metrics["revenue_growth"].append(fundamentals["revenue_growth"])

            if "ebitda_margin" in fundamentals:
                sector_metrics["ebitda_margin"].append(fundamentals["ebitda_margin"])

            if "pe_ratio" in fundamentals and fundamentals["pe_ratio"] > 0:
                # Only include positive P/E ratios (negative means company losing money)
                sector_metrics["pe_ratios"].append(fundamentals["pe_ratio"])

            if "debt_to_equity" in fundamentals:
                sector_metrics["debt_to_equity"].append(fundamentals["debt_to_equity"])

            if "roe" in fundamentals:
                sector_metrics["roe"].append(fundamentals["roe"])

            if "price_momentum_12m" in fundamentals:
                sector_metrics["price_momentum"].append(fundamentals["price_momentum_12m"])

        # === STEP 2: Calculate Sector-Wide Averages ===
        # Why: Individual companies vary, but sector average shows overall health
        trends = {
            # Average revenue growth across all companies
            "avg_revenue_growth": statistics.mean(sector_metrics["revenue_growth"]) if sector_metrics["revenue_growth"] else 0.0,

            # Average EBITDA margin
            "avg_ebitda_margin": statistics.mean(sector_metrics["ebitda_margin"]) if sector_metrics["ebitda_margin"] else 0.0,

            # Average P/E ratio (only positive values)
            "avg_pe_ratio": statistics.mean([p for p in sector_metrics["pe_ratios"] if p > 0]) if sector_metrics["pe_ratios"] else 0.0,

            # Average ROE (Return on Equity)
            "avg_roe": statistics.mean(sector_metrics["roe"]) if sector_metrics["roe"] else 0.0,

            # Average price momentum over 12 months
            "avg_price_momentum": statistics.mean(sector_metrics["price_momentum"]) if sector_metrics["price_momentum"] else 0.0
        }

        # === STEP 3: Determine Trend Directions ===
        # Use thresholds to classify trends

        # Revenue trend: positive if > 5% growth, negative if < -2%, else neutral
        trends["revenue_trend"] = "positive" if trends["avg_revenue_growth"] > 0.05 else \
                                 "negative" if trends["avg_revenue_growth"] < -0.02 else "neutral"

        # Profitability trend: strong if ROE > 15%, weak if < 8%, else moderate
        trends["profitability_trend"] = "strong" if trends["avg_roe"] > 0.15 else \
                                       "weak" if trends["avg_roe"] < 0.08 else "moderate"

        # Valuation trend: expensive if P/E > 25, cheap if < 15, else fair
        trends["valuation_trend"] = "expensive" if trends["avg_pe_ratio"] > 25 else \
                                   "cheap" if trends["avg_pe_ratio"] < 15 else "fair"

        # Momentum trend: positive if > 10% price increase, negative if < -5%, else neutral
        trends["momentum_trend"] = "positive" if trends["avg_price_momentum"] > 0.1 else \
                                  "negative" if trends["avg_price_momentum"] < -0.05 else "neutral"

        # === STEP 4: Calculate Confidence Based on Trend Consistency ===
        # More positive trends = higher confidence
        confidence_factors = []

        # Add confidence points for positive indicators
        if trends["revenue_trend"] == "positive":
            confidence_factors.append(0.8)  # 80% confidence from revenue growth

        if trends["profitability_trend"] == "strong":
            confidence_factors.append(0.7)  # 70% confidence from strong profitability

        if trends["valuation_trend"] in ["fair", "cheap"]:
            confidence_factors.append(0.6)  # 60% confidence from reasonable valuation

        if trends["momentum_trend"] == "positive":
            confidence_factors.append(0.7)  # 70% confidence from positive momentum

        # Average all confidence factors to get overall confidence
        # If no positive factors, default to 0.5 (neutral)
        trends["confidence"] = statistics.mean(confidence_factors) if confidence_factors else 0.5

        return trends

    def _combine_confidence_scores(self, llm_confidence: float, trend_confidence: float) -> float:
        """Combine LLM and quantitative trend confidence scores.

        What: Blends two confidence scores into one
        Why: Want both LLM's qualitative insights and quantitative verification
        How: Weighted average favoring quantitative (more reliable)

        Data Flow:
        Receives: llm_confidence=0.7, trend_confidence=0.8
        Returns: 0.76 (0.7 * 0.4 + 0.8 * 0.6)
        """
        # Weighted combination favoring quantitative analysis (60%)
        # Why 60/40: Quantitative metrics are more objective than LLM opinions
        return (llm_confidence * 0.4 + trend_confidence * 0.6)


class SentimentAnalysisAgent:
    """Analyzes sentiment from news and social media for S&P 500 companies.

    What: Working agent that analyzes news sentiment across multiple stocks
    Why: Sentiment drives short-term price movements, need to track it
    How: Uses FinBERT (if available) or LLM to analyze news articles

    Data Flow:
    Input: comprehensive_news_data (news for multiple companies)
    Output: Sector-wide sentiment analysis with confidence

    Key Features:
    - Analyzes sentiment across multiple companies simultaneously
    - Calculates sentiment convergence (how much companies agree)
    - Identifies positive drivers and negative concerns
    """

    def __init__(
        self,
        llm: Runnable,  # Language model for sentiment analysis
        *,
        use_finbert: bool = True,  # Whether to use FinBERT
    ) -> None:
        """Initialize sentiment analysis agent.

        What: Sets up FinBERT (if available) for multi-company sentiment
        How: Loads FinBERT assets and validates availability
        Why: FinBERT is specialized for financial sentiment
        """
        self.use_finbert = use_finbert

        # Try to initialize FinBERT sentiment analyzer if available
        self.sentiment_analyzer = None
        if use_finbert:
            try:
                from .sentiment.finbert import create_sentiment_analyzer
                self.sentiment_analyzer = create_sentiment_analyzer()
            except ImportError:
                logger.warning("FinBERT dependencies missing; SentimentAnalysisAgent will use LLM fallback.")

        # Create prompt template for multi-company sentiment analysis
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

        # Create LLM chain
        self._chain = base_prompt | llm | StrOutputParser()

    def run(self, *, comprehensive_news_data: Dict[str, List[Dict]], market_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run sentiment analysis across multiple companies.

        What: Analyzes news sentiment for multiple stocks simultaneously
        How:
        1. Try FinBERT first (if available) for sector-wide analysis
        2. Fall back to LLM if FinBERT unavailable
        3. Extract themes from headlines
        4. Return comprehensive sentiment analysis

        Data Flow:
        Receives: comprehensive_news_data = {
            "AAPL": [
                {title: "Apple beats earnings", content: "..."},
                {title: "iPhone sales strong", content: "..."}
            ],
            "MSFT": [...]
        }
        Returns: {
            "overall_sentiment": "positive",
            "sentiment_score": 75,
            "confidence": 0.82,
            "headlines": [...],
            "positive_drivers": "Strong earnings, product launches",
            "negative_concerns": "Regulatory concerns",
            ...
        }
        """

        # === STRATEGY 1: Try FinBERT Sector Analysis ===
        # Why: FinBERT can analyze multiple companies' sentiment simultaneously
        if self.use_finbert and self.sentiment_analyzer and comprehensive_news_data:
            try:
                # Use FinBERT's sector-wide sentiment analysis
                # This analyzes all companies' news and calculates convergence
                finbert_result = self.sentiment_analyzer.analyze_sector_sentiment(comprehensive_news_data)

                # Extract top headlines from all companies
                # Take 1 headline per company to get diverse coverage
                headlines = []
                for ticker, result in finbert_result["sector_sentiment"].items():
                    headlines.extend(result.get("headlines", [])[:1])  # Top headline per ticker

                # Return structured result from FinBERT analysis
                return {
                    "overall_sentiment": finbert_result["overall_sentiment"],  # "positive", "negative", "neutral"
                    "sentiment_score": int(finbert_result["overall_score"] * 100),  # Convert to 0-100 scale
                    "trend_direction": "improving" if finbert_result["overall_score"] > 0.6 else \
                                     "declining" if finbert_result["overall_score"] < 0.4 else "stable",
                    "confidence": finbert_result["convergence_score"],  # How much companies agree
                    "headlines": headlines[:3],  # Top 3 headlines
                    "positive_drivers": self._extract_positives_from_headlines(headlines),  # Extract themes
                    "negative_concerns": self._extract_negatives_from_headlines(headlines),  # Extract concerns
                    "sentiment_confidence": finbert_result["convergence_score"],  # For coordination agent
                    "sector_analysis": finbert_result["sector_sentiment"],  # Per-ticker details
                    "total_articles": finbert_result["total_articles"]  # How many articles analyzed
                }
            except Exception as e:
                # If FinBERT fails, log and fall back to LLM
                logger.warning(f"FinBERT sector analysis failed: {e}, falling back to LLM")

        # === STRATEGY 2: LLM Fallback ===
        # Use GPT to analyze sentiment if FinBERT unavailable

        # Invoke LLM chain with news data
        raw_output = self._chain.invoke({
            "comprehensive_news_data": str(comprehensive_news_data),
            "market_context": str(market_context or {})  # Optional market context
        })

        # Parse LLM response
        overall_sentiment = _extract_sentiment_current(raw_output)
        sentiment_score = _extract_score(raw_output, "SENTIMENT_SCORE")
        trend_direction = _extract_sentiment_trend(raw_output)
        confidence = _extract_confidence(raw_output, default=0.5)
        headlines = _extract_headlines(raw_output)
        positive_drivers = _extract_text_section(raw_output, "POSITIVE_DRIVERS")
        negative_concerns = _extract_text_section(raw_output, "NEGATIVE_CONCERNS")

        # Return structured result
        return {
            "overall_sentiment": overall_sentiment,
            "sentiment_score": sentiment_score,
            "trend_direction": trend_direction,
            "confidence": confidence,
            "headlines": headlines,
            "positive_drivers": positive_drivers,
            "negative_concerns": negative_concerns,
            "sentiment_confidence": confidence  # For coordination agent
        }

    def _extract_positives_from_headlines(self, headlines: List[str]) -> str:
        """Extract positive themes from headlines.

        What: Identifies positive keywords in headlines
        Why: Want to know what's driving positive sentiment
        How: Checks for positive keywords, extracts themes

        Data Flow:
        Receives: ["Apple beats earnings", "Microsoft shows strong growth"]
        Returns: "'beat' trend, 'strong' trend"
        """
        # Positive keywords to look for
        positive_keywords = ["strong", "growth", "beat", "exceed", "positive", "bullish", "upgrade"]
        positive_themes = []

        # Check each headline for positive keywords
        for headline in headlines:
            headline_lower = headline.lower()
            for keyword in positive_keywords:
                if keyword in headline_lower:
                    positive_themes.append(f"'{keyword}' trend")
                    break  # Only count each headline once

        # Return comma-separated unique themes
        return ", ".join(set(positive_themes)) if positive_themes else "Mixed positive indicators"

    def _extract_negatives_from_headlines(self, headlines: List[str]) -> str:
        """Extract negative themes from headlines.

        What: Identifies negative keywords in headlines
        Why: Want to know what concerns exist
        How: Checks for negative keywords, extracts themes

        Data Flow:
        Receives: ["Company misses earnings", "Concerns about volatility"]
        Returns: "'miss' trend, 'concern' trend"
        """
        # Negative keywords to look for
        negative_keywords = ["weak", "miss", "down", "negative", "bearish", "downgrade", "concern", "volatility"]
        negative_themes = []

        # Check each headline for negative keywords
        for headline in headlines:
            headline_lower = headline.lower()
            for keyword in negative_keywords:
                if keyword in headline_lower:
                    negative_themes.append(f"'{keyword}' trend")
                    break  # Only count each headline once

        # Return comma-separated unique themes
        return ", ".join(set(negative_themes)) if negative_themes else "No major concerns identified"


# ===== HELPER FUNCTIONS FOR PARSING AGENT OUTPUTS =====
# These functions extract structured data from LLM text responses
# Why: LLMs return unstructured text, we need structured data for our workflow

# Regular expression patterns for parsing
# Why regex: Efficient way to find patterns in text
CONFIDENCE_PATTERN = re.compile(r"CONFIDENCE:\s*([0-9.]+)")  # Finds "CONFIDENCE: 0.85"
DIRECTION_PATTERN = re.compile(r"DIRECTION:\s*(up|down|neutral)", re.IGNORECASE)  # Finds "DIRECTION: up"
SENTIMENT_PATTERN = re.compile(r"CURRENT:\s*(positive|neutral|negative)", re.IGNORECASE)  # Finds "CURRENT: positive"
TREND_PATTERN = re.compile(r"TREND:\s*(improving|stable|declining)", re.IGNORECASE)  # Finds "TREND: improving"


def _extract_confidence(raw_text: str, *, default: float) -> float:
    """Extract confidence score from agent output.

    What: Finds "CONFIDENCE: 0.85" in text and returns 0.85
    How: Uses regex to match pattern, converts to float
    Why: Need numerical confidence value for calculations

    Data Flow:
    Input: "...CONFIDENCE: 0.85..."
    Output: 0.85
    """
    # Search for CONFIDENCE pattern in text
    match = CONFIDENCE_PATTERN.search(raw_text)
    if not match:
        return default  # Return default if not found

    # Extract matched value and convert to float
    value = float(match.group(1))  # group(1) is first captured group

    # Clamp to valid range [0.0, 1.0]
    # Why: Prevent invalid values like -0.5 or 1.5
    return max(0.0, min(1.0, value))


def _extract_direction(raw_text: str) -> Literal["up", "down", "neutral"]:
    """Extract direction from agent output.

    What: Finds "DIRECTION: up" in text and returns "up"
    How: Uses regex pattern matching
    Why: Need direction as enum for type safety

    Data Flow:
    Input: "...DIRECTION: up..."
    Output: "up"
    """
    match = DIRECTION_PATTERN.search(raw_text)
    if not match:
        return "neutral"  # Default to neutral if not found

    # Return matched direction in lowercase, cast to Literal
    from typing import cast
    val = match.group(1).lower() if match else "neutral"
    allowed = {"up", "down", "neutral"}
    result = val if val in allowed else "neutral"
    return cast(Literal["up", "down", "neutral"], result)


def _extract_narrative(raw_text: str) -> str:
    """Extract narrative explanation from agent output.

    What: Extracts multi-line narrative text
    How: Finds "NARRATIVE:" line, collects following lines until blank line
    Why: Narratives can span multiple lines

    Data Flow:
    Input: "...NARRATIVE: The stock shows strong momentum.
           Technical indicators are positive.

           OTHER_FIELD: ..."
    Output: "The stock shows strong momentum. Technical indicators are positive."
    """
    lines = raw_text.split('\n')  # Split into lines
    narrative_lines = []
    in_narrative = False

    # Process each line
    for line in lines:
        if line.strip().startswith('NARRATIVE:'):
            # Found start of narrative
            in_narrative = True
            # Extract text after "NARRATIVE:"
            narrative_lines.append(line.replace('NARRATIVE:', '').strip())
        elif in_narrative and line.strip():
            # In narrative section, line has content
            narrative_lines.append(line.strip())
        elif in_narrative and not line.strip():
            # In narrative section, hit blank line - end of narrative
            break

    # Join all narrative lines with spaces
    return ' '.join(narrative_lines) if narrative_lines else "No narrative provided"


def _extract_sentiment_current(raw_text: str) -> Literal["positive", "neutral", "negative"]:
    """Extract current sentiment from agent output.

    What: Finds "CURRENT: positive" and returns "positive"
    How: Uses regex pattern matching
    Why: Need sentiment as enum
    """
    match = SENTIMENT_PATTERN.search(raw_text)
    from typing import cast
    if not match:
        return cast(Literal["positive", "neutral", "negative"], "neutral")
    val = match.group(1).lower()
    allowed = {"positive", "neutral", "negative"}
    result = val if val in allowed else "neutral"
    return cast(Literal["positive", "neutral", "negative"], result)


def _extract_sentiment_trend(raw_text: str) -> Literal["improving", "stable", "declining"]:
    """Extract sentiment trend from agent output.

    What: Finds "TREND: improving" and returns "improving"
    How: Uses regex pattern matching
    Why: Need trend direction for analysis
    """
    match = TREND_PATTERN.search(raw_text)
    from typing import cast
    if not match:
        return cast(Literal["improving", "stable", "declining"], "stable")
    val = match.group(1).lower()
    allowed = {"improving", "stable", "declining"}
    result = val if val in allowed else "stable"
    return cast(Literal["improving", "stable", "declining"], result)


def _extract_headlines(raw_text: str) -> List[str]:
    """Extract headlines from agent output.

    What: Extracts list of headlines from text
    How: Finds "HEADLINES:" section, collects following lines
    Why: Headlines can be multi-line list

    Data Flow:
    Input: "...HEADLINES: Apple beats earnings
           iPhone sales strong
           Market reacts positively

           OTHER_FIELD: ..."
    Output: ["Apple beats earnings", "iPhone sales strong", "Market reacts positively"]
    """
    lines = raw_text.split('\n')
    headlines = []
    in_headlines = False

    for line in lines:
        if line.strip().startswith('HEADLINES:'):
            # Found start of headlines section
            in_headlines = True
            # Extract any headline on same line
            headline = line.replace('HEADLINES:', '').strip()
            if headline:
                headlines.append(headline)
        elif in_headlines and line.strip() and not line.strip().startswith(('CURRENT:', 'SCORE:', 'TREND:')):
            # In headlines section, line has content, not a new field
            headlines.append(line.strip())
        elif in_headlines and not line.strip():
            # Hit blank line, end of headlines
            break

    # Return max 3 headlines
    return headlines[:3]


def _extract_smart_money_section(raw_text: str, section: str) -> Dict[str, Any]:
    """Extract a specific section from smart money agent output.

    What: Extracts one of three sections: INSTITUTIONS, INSIDERS, or CONGRESS
    How: Finds section header, collects following lines
    Why: Smart money data has three categories

    Data Flow:
    Input: raw_text="...INSTITUTIONS: Ownership increased 3%
           Major funds buying...

           INSIDERS: ...", section="INSTITUTIONS"
    Output: {"summary": "Ownership increased 3% Major funds buying..."}
    """
    lines = raw_text.split('\n')
    section_data = []
    in_section = False

    for line in lines:
        if line.strip().startswith(f'{section}:'):
            # Found our section
            in_section = True
            # Extract content on same line
            content = line.replace(f'{section}:', '').strip()
            if content:
                section_data.append(content)
        elif in_section and line.strip() and not line.strip().startswith(('INSTITUTIONS:', 'INSIDERS:', 'CONGRESS:')):
            # In our section, line has content, not a new section
            section_data.append(line.strip())
        elif in_section and not line.strip():
            # Hit blank line, end of section
            break

    # Parse section data into structured format
    if not section_data:
        return {"summary": f"No recent {section.lower()} activity"}

    # Join all lines into summary
    summary = ' '.join(section_data)
    return {"summary": summary}


# Additional helper functions for enhanced agents

HISTORICAL_TREND_PATTERN = re.compile(r"OVERALL_TREND:\s*(positive|neutral|negative)", re.IGNORECASE)
SCORE_PATTERN = re.compile(r"(\w+_SCORE):\s*(\d+)", re.IGNORECASE)
TICKER_LIST_PATTERN = re.compile(r"(\w+):\s*\[([^\]]+)\]", re.IGNORECASE)


def _extract_historical_trend(raw_text: str) -> Literal["positive", "neutral", "negative"]:
    """Extract historical trend from agent output.

    What: Finds "OVERALL_TREND: positive" and returns "positive"
    How: Uses regex pattern matching
    Why: Need trend classification for historical analysis
    """
    match = HISTORICAL_TREND_PATTERN.search(raw_text)
    from typing import cast
    if not match:
        return cast(Literal["positive", "neutral", "negative"], "neutral")
    val = match.group(1).lower()
    allowed = {"positive", "neutral", "negative"}
    result = val if val in allowed else "neutral"
    return cast(Literal["positive", "neutral", "negative"], result)


def _extract_score(raw_text: str, score_type: str) -> int:
    """Extract score from agent output.

    What: Finds "TECHNICAL_SCORE: 75" and returns 75
    How: Uses dynamic regex based on score_type
    Why: Different agents return different score types

    Data Flow:
    Input: raw_text="...TECHNICAL_SCORE: 75...", score_type="TECHNICAL_SCORE"
    Output: 75
    """
    # Create pattern for specific score type
    pattern = re.compile(f"{score_type}:\\s*(\\d+)", re.IGNORECASE)
    match = pattern.search(raw_text)
    if not match:
        return 50  # Default neutral score (50 out of 100)
    return int(match.group(1))


def _extract_ticker_list(raw_text: str, section: str) -> List[str]:
    """Extract list of tickers from agent output.

    What: Finds "TOP_PERFORMERS: [AAPL, MSFT, GOOGL]" and returns ["AAPL", "MSFT", "GOOGL"]
    How: Uses regex to find bracketed list, splits by comma
    Why: Ticker lists need to be parsed into arrays

    Data Flow:
    Input: "...TOP_PERFORMERS: [AAPL, MSFT, GOOGL]...", section="TOP_PERFORMERS"
    Output: ["AAPL", "MSFT", "GOOGL"]
    """
    # Create pattern for specific section
    pattern = re.compile(f"{section}:\\s*\\[([^\\]]+)\\]", re.IGNORECASE)
    match = pattern.search(raw_text)
    if not match:
        return []

    # Split by comma and clean up
    # Convert to uppercase (ticker symbols are uppercase by convention)
    tickers = [ticker.strip().upper() for ticker in match.group(1).split(',')]

    # Filter out empty strings
    return [ticker for ticker in tickers if ticker]


def _extract_text_section(raw_text: str, section: str) -> str:
    """Extract a text section from agent output.

    What: Extracts multi-line text section (like POSITIVE_DRIVERS)
    How: Finds section header, collects lines until blank or new field
    Why: Some sections are paragraphs, not single values

    Data Flow:
    Input: "...POSITIVE_DRIVERS: Strong earnings
           New product launches
           Market leadership

           NEGATIVE_CONCERNS: ...", section="POSITIVE_DRIVERS"
    Output: "Strong earnings New product launches Market leadership"
    """
    lines = raw_text.split('\n')
    section_data = []
    in_section = False

    for line in lines:
        if line.strip().startswith(f'{section}:'):
            # Found our section
            in_section = True
            # Extract content on same line
            content = line.replace(f'{section}:', '').strip()
            if content:
                section_data.append(content)
        elif in_section and line.strip() and not line.strip().startswith(('OVERALL_SENTIMENT:', 'SENTIMENT_SCORE:', 'TREND_DIRECTION:', 'CONFIDENCE:')):
            # In our section, line has content, not a new field
            section_data.append(line.strip())
        elif in_section and not line.strip():
            # Hit blank line, end of section
            break

    # Join all lines with spaces
    return ' '.join(section_data) if section_data else "No data available"
