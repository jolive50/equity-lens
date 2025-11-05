"""Prediction ensemble for combining multiple models."""

import logging
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from models.prediction.base_predictor import BasePredictionModel, PredictionResult

logger = logging.getLogger(__name__)


class PredictionEnsemble(BasePredictionModel):
    """Ensemble combining 2+ prediction models.

    Flexible architecture supporting:
    - Any combination of models (LSTM, GRU, XGBoost)
    - Multiple ensemble strategies (weighted_average, simple_average, voting)
    - Runtime model selection via configuration
    """

    def __init__(
        self,
        models: List[BasePredictionModel],
        strategy: str = "weighted_average",
        weights: Dict[str, float] = None
    ):
        """Initialize prediction ensemble.

        Args:
            models: List of 2+ prediction models
            strategy: Combination strategy ('weighted_average', 'simple_average', 'voting')
            weights: Optional dict mapping model names to weights

        Raises:
            ValueError: If less than 2 models provided or invalid strategy
        """
        super().__init__(model_name="PredictionEnsemble")

        if len(models) < 2:
            raise ValueError("Ensemble requires at least 2 models")

        valid_strategies = ['weighted_average', 'simple_average', 'voting']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Use one of {valid_strategies}")

        self.models = models
        self.strategy = strategy

        # Set weights
        if weights:
            self.weights = weights
        else:
            # Default: use model weights if available, else equal weights
            total_weight = sum(m.weight if hasattr(m, 'weight') else 1.0 for m in models)
            self.weights = {
                m.model_name: (m.weight if hasattr(m, 'weight') else 1.0) / total_weight
                for m in models
            }

        self.is_trained = all(m.is_trained for m in models)
        logger.info(
            f"Initialized ensemble with {len(models)} models: "
            f"{[m.model_name for m in models]} using {strategy}"
        )

    def load_model(self, model_path: str) -> None:
        """Ensemble doesn't load from single path.

        Individual models should be loaded before creating ensemble.

        Raises:
            NotImplementedError: Ensemble models load individually
        """
        raise NotImplementedError(
            "Ensemble models load individually. "
            "Load each model separately before creating ensemble."
        )

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Predict using ensemble of models.

        Args:
            data: DataFrame with OHLCV columns

        Returns:
            PredictionResult with combined prediction

        Raises:
            RuntimeError: If any model fails
        """
        if not self.is_trained:
            raise RuntimeError("Not all ensemble models are trained")

        # Get predictions from all models
        predictions = []
        for model in self.models:
            try:
                pred = model.predict(data)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"{model.model_name} prediction failed: {e}")
                raise RuntimeError(f"Ensemble failed: {model.model_name} error: {e}")

        # Combine predictions based on strategy
        if self.strategy == 'weighted_average':
            result = self._weighted_average(predictions)
        elif self.strategy == 'simple_average':
            result = self._simple_average(predictions)
        elif self.strategy == 'voting':
            result = self._voting(predictions)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return result

    def _weighted_average(self, predictions: List[PredictionResult]) -> PredictionResult:
        """Combine predictions using weighted average of probabilities.

        Args:
            predictions: List of PredictionResult from each model

        Returns:
            Combined PredictionResult
        """
        # Initialize probability accumulators
        prob_up = 0.0
        prob_down = 0.0
        prob_neutral = 0.0

        # Weighted sum of probabilities
        for pred in predictions:
            model_name = pred.metadata['model']
            weight = self.weights.get(model_name, 1.0 / len(predictions))

            prob_up += pred.probabilities['up'] * weight
            prob_down += pred.probabilities['down'] * weight
            prob_neutral += pred.probabilities['neutral'] * weight

        # Determine direction
        if prob_up > prob_down and prob_up > prob_neutral:
            direction = "up"
            confidence = prob_up
        elif prob_down > prob_up and prob_down > prob_neutral:
            direction = "down"
            confidence = prob_down
        else:
            direction = "neutral"
            confidence = prob_neutral

        return PredictionResult(
            direction=direction,
            confidence=confidence,
            probabilities={
                'up': prob_up,
                'down': prob_down,
                'neutral': prob_neutral
            },
            metadata={
                'model': 'Ensemble',
                'strategy': 'weighted_average',
                'num_models': len(predictions),
                'models': [p.metadata['model'] for p in predictions],
                'individual_predictions': [
                    {
                        'model': p.metadata['model'],
                        'direction': p.direction,
                        'confidence': p.confidence
                    } for p in predictions
                ]
            }
        )

    def _simple_average(self, predictions: List[PredictionResult]) -> PredictionResult:
        """Combine predictions using simple average (equal weights).

        Args:
            predictions: List of PredictionResult from each model

        Returns:
            Combined PredictionResult
        """
        # Calculate simple average
        prob_up = np.mean([p.probabilities['up'] for p in predictions])
        prob_down = np.mean([p.probabilities['down'] for p in predictions])
        prob_neutral = np.mean([p.probabilities['neutral'] for p in predictions])

        # Determine direction
        if prob_up > prob_down and prob_up > prob_neutral:
            direction = "up"
            confidence = prob_up
        elif prob_down > prob_up and prob_down > prob_neutral:
            direction = "down"
            confidence = prob_down
        else:
            direction = "neutral"
            confidence = prob_neutral

        return PredictionResult(
            direction=direction,
            confidence=confidence,
            probabilities={
                'up': float(prob_up),
                'down': float(prob_down),
                'neutral': float(prob_neutral)
            },
            metadata={
                'model': 'Ensemble',
                'strategy': 'simple_average',
                'num_models': len(predictions),
                'models': [p.metadata['model'] for p in predictions],
                'individual_predictions': [
                    {
                        'model': p.metadata['model'],
                        'direction': p.direction,
                        'confidence': p.confidence
                    } for p in predictions
                ]
            }
        )

    def _voting(self, predictions: List[PredictionResult]) -> PredictionResult:
        """Combine predictions using majority voting.

        Args:
            predictions: List of PredictionResult from each model

        Returns:
            Combined PredictionResult
        """
        # Count votes for each direction
        votes = {'up': 0, 'down': 0, 'neutral': 0}
        for pred in predictions:
            votes[pred.direction] += 1

        # Find majority direction
        direction = max(votes, key=votes.get)
        confidence = votes[direction] / len(predictions)

        # Calculate average probabilities for metadata
        prob_up = np.mean([p.probabilities['up'] for p in predictions])
        prob_down = np.mean([p.probabilities['down'] for p in predictions])
        prob_neutral = np.mean([p.probabilities['neutral'] for p in predictions])

        return PredictionResult(
            direction=direction,
            confidence=confidence,
            probabilities={
                'up': float(prob_up),
                'down': float(prob_down),
                'neutral': float(prob_neutral)
            },
            metadata={
                'model': 'Ensemble',
                'strategy': 'voting',
                'num_models': len(predictions),
                'votes': votes,
                'models': [p.metadata['model'] for p in predictions],
                'individual_predictions': [
                    {
                        'model': p.metadata['model'],
                        'direction': p.direction,
                        'confidence': p.confidence
                    } for p in predictions
                ]
            }
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Return ensemble metadata."""
        info = super().get_model_info()
        info.update({
            'num_models': len(self.models),
            'models': [m.get_model_info() for m in self.models],
            'strategy': self.strategy,
            'weights': self.weights
        })
        return info
