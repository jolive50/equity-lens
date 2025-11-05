"""Prediction agent for stock price forecasting."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Any
from langchain_core.runnables import Runnable


logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Structured output from prediction agent.

    Attributes:
        direction: Predicted price direction (up/down/neutral)
        confidence: Confidence score (0.0-1.0)
        narrative: Human-readable explanation
        daily_probs: Optional probability curve for multiple days
        horizon_95: Optional 95% confidence horizon metadata
        feature_importance: Optional SHAP feature importance scores
    """
    direction: Literal["up", "down", "neutral"]
    confidence: float
    narrative: str
    daily_probs: Optional[List[Dict[str, Any]]] = None
    horizon_95: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None


class PredictionAgent:
    """Generates probabilistic directional forecasts using ML models.

    Uses LSTM forecaster from Pam's models/prediction/ to generate predictions.
    Optionally uses GPT to explain predictions with SHAP feature importance.
    """

    def __init__(
        self,
        llm: Optional[Runnable] = None,
        *,
        use_ml_model: bool = True,
        use_gpt_explanations: bool = False
    ) -> None:
        """Initialize prediction agent.

        Args:
            llm: Optional LLM for generating explanations
            use_ml_model: Always True (kept for compatibility)
            use_gpt_explanations: If True, use GPT to explain predictions

        Raises:
            RuntimeError: If forecaster dependencies missing or initialization fails
        """
        if not use_ml_model:
            logger.warning("PredictionAgent enforces ML-only mode")

        try:
            from models.prediction.forecaster import create_forecaster
        except ImportError as exc:
            raise RuntimeError(
                "Forecaster dependencies missing. Install required packages."
            ) from exc

        self.forecaster = create_forecaster("lstm")
        if self.forecaster is None:
            raise RuntimeError("create_forecaster returned None")

        self.llm = llm
        self.use_gpt_explanations = use_gpt_explanations

        if use_gpt_explanations and llm is not None:
            logger.info("PredictionAgent: GPT explanations enabled")
        else:
            logger.info("PredictionAgent: Using basic narrative")

    def _build_narrative(self, ticker: str, ml_result) -> str:
        """Generate explanation of prediction.

        Args:
            ticker: Stock symbol
            ml_result: ForecastResult from LSTM model

        Returns:
            Narrative string explaining the prediction
        """
        if self.use_gpt_explanations and self.llm is not None:
            return self._build_gpt_narrative(ticker, ml_result)
        else:
            return self._build_basic_narrative(ticker, ml_result)

    def _build_basic_narrative(self, ticker: str, ml_result) -> str:
        """Build basic narrative without GPT."""
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
        feature_summary = ", ".join(
            f"{name} ({weight:.2f})" for name, weight in top_features
        ) if top_features else "insufficient feature data"

        days = ml_result.horizon_95.get("days", 0) if ml_result.horizon_95 else 0

        return (
            f"{ticker.upper()} forecast: {ml_result.direction.upper()} with "
            f"{ml_result.confidence:.1%} confidence using LSTM. "
            f"Probabilities - up: {prob_up:.1%}, down: {prob_down:.1%}, neutral: {prob_neutral:.1%}. "
            f"95% confidence horizon: {days} day(s). "
            f"Top features: {feature_summary}."
        )

    def _build_gpt_narrative(self, ticker: str, ml_result) -> str:
        """Build GPT-powered narrative with SHAP explanations.

        Args:
            ticker: Stock symbol
            ml_result: ForecastResult with SHAP feature importance

        Returns:
            Paragraph-long explanation from GPT
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
        )[:5]

        features_text = "\n".join(
            f"- {name}: {weight:.1%} importance"
            for name, weight in top_features
        ) if top_features else "No specific features identified"

        prompt = f"""You are explaining a stock price prediction to a high school student.

Stock: {ticker.upper()}
Prediction: {ml_result.direction.upper()}
Confidence: {ml_result.confidence:.1%}

Probabilities:
- Up: {prob_up:.1%}
- Down: {prob_down:.1%}
- Steady: {prob_neutral:.1%}

Most Important Features:
{features_text}

Write a single paragraph (4-6 sentences) that:
1. States the prediction and confidence clearly
2. Explains what the top features mean in simple terms
3. Describes why these features suggest this direction
4. Uses analogies a high school student would understand

Do not use jargon like "LSTM" or "SHAP". Say "our AI model" instead.
Be conversational but accurate."""

        try:
            response = self.llm.invoke(prompt)
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        except Exception as e:
            logger.warning(f"GPT explanation failed: {e}, using basic narrative")
            return self._build_basic_narrative(ticker, ml_result)

    def run(
        self,
        *,
        ticker: str,
        market_data: Dict,
        fundamentals: Dict[str, float]
    ) -> PredictionResult:
        """Make prediction for a stock.

        Args:
            ticker: Stock symbol
            market_data: Historical price/volume data
            fundamentals: Company financial metrics

        Returns:
            PredictionResult with direction, confidence, and narrative

        Raises:
            ValueError: If market_data format invalid
            RuntimeError: If forecasting fails
        """
        if not isinstance(market_data, list):
            raise ValueError("market_data must be a list of daily price records")

        try:
            ml_result = self.forecaster.predict(market_data, fundamentals)
        except Exception as exc:
            raise RuntimeError(f"Forecasting failed for {ticker}: {exc}") from exc

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
