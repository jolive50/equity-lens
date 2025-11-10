"""Prediction ensemble combining multiple models.

PAM's Component - Ensemble System
Combines 2+ prediction models using weighted averaging or voting.
"""
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from .base_predictor import BasePredictionModel, PredictionResult

logger = logging.getLogger(__name__)


class PredictionEnsemble(BasePredictionModel):
    """Ensemble of prediction models with configurable strategy."""

    def __init__(
        self,
        models: List[BasePredictionModel],
        strategy: str = "weighted_average",
        weights: Optional[Dict[str, float]] = None
    ):
        """Initialize prediction ensemble.

        Args:
            models: List of 2+ BasePredictionModel instances
            strategy: "weighted_average", "simple_average", or "voting"
            weights: Optional dict mapping model names to weights
        """
        if len(models) < 2:
            raise ValueError("Ensemble requires at least 2 models")

        self.models = models
        self.strategy = strategy
        self.weights = weights or self._equal_weights()

        logger.info(f"Created ensemble with {len(models)} models using {strategy} strategy")

    def _equal_weights(self) -> Dict[str, float]:
        """Generate equal weights for all models."""
        weight = 1.0 / len(self.models)
        return {model.get_model_info()["name"]: weight for model in self.models}

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Make ensemble prediction.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            PredictionResult aggregating all model predictions
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            try:
                pred = model.predict(data)
                predictions.append((model.get_model_info()["name"], pred))
            except Exception as e:
                logger.warning(f"Model {model.get_model_info()['name']} failed: {e}")
                continue

        if not predictions:
            raise RuntimeError("All models failed to predict")

        # Combine predictions based on strategy
        if self.strategy == "weighted_average":
            return self._weighted_average(predictions)
        elif self.strategy == "simple_average":
            return self._simple_average(predictions)
        elif self.strategy == "voting":
            return self._voting(predictions)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _weighted_average(self, predictions: List[tuple]) -> PredictionResult:
        """Combine using weighted average of probabilities."""
        total_up, total_down, total_neutral = 0.0, 0.0, 0.0
        total_weight = 0.0

        for model_name, pred in predictions:
            weight = self.weights.get(model_name, 1.0 / len(predictions))
            total_up += pred.probabilities["up"] * weight
            total_down += pred.probabilities["down"] * weight
            total_neutral += pred.probabilities["neutral"] * weight
            total_weight += weight

        # Normalize
        total_up /= total_weight
        total_down /= total_weight
        total_neutral /= total_weight

        # Determine direction
        if total_up > max(total_down, total_neutral):
            direction, confidence = "up", total_up
        elif total_down > total_neutral:
            direction, confidence = "down", total_down
        else:
            direction, confidence = "neutral", total_neutral

        return PredictionResult(
            direction=direction,
            confidence=float(confidence),
            probabilities={
                "up": float(total_up),
                "down": float(total_down),
                "neutral": float(total_neutral)
            },
            metadata={
                "model": "Ensemble",
                "strategy": "weighted_average",
                "num_models": len(predictions),
                "models": [name for name, _ in predictions]
            }
        )

    def _simple_average(self, predictions: List[tuple]) -> PredictionResult:
        """Simple average of all model probabilities."""
        equal_weights = {name: 1.0/len(predictions) for name, _ in predictions}
        old_weights = self.weights
        self.weights = equal_weights
        result = self._weighted_average(predictions)
        self.weights = old_weights
        result.metadata["strategy"] = "simple_average"
        return result

    def _voting(self, predictions: List[tuple]) -> PredictionResult:
        """Majority vote among models."""
        votes = {"up": 0, "down": 0, "neutral": 0}

        for _, pred in predictions:
            votes[pred.direction] += 1

        direction = max(votes, key=votes.get)
        confidence = votes[direction] / len(predictions)

        return PredictionResult(
            direction=direction,
            confidence=confidence,
            probabilities={
                "up": votes["up"] / len(predictions),
                "down": votes["down"] / len(predictions),
                "neutral": votes["neutral"] / len(predictions)
            },
            metadata={
                "model": "Ensemble",
                "strategy": "voting",
                "num_models": len(predictions),
                "votes": votes
            }
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Return ensemble metadata."""
        return {
            "name": "PredictionEnsemble",
            "type": "ensemble",
            "strategy": self.strategy,
            "num_models": len(self.models),
            "models": [m.get_model_info()["name"] for m in self.models]
        }
