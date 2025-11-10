from typing import Dict, Any, List, Optional
import numpy as np
from .base_sentiment import BaseSentimentModel, SentimentResult


class SentimentEnsemble(BaseSentimentModel):
    """Flexible ensemble of 2+ sentiment models.

    Supports multiple combination strategies:
    - weighted_average: Weighted average of probabilities
    - simple_average: Equal weight to all models
    - voting: Majority vote on labels
    """

    VALID_STRATEGIES = ["weighted_average", "simple_average", "voting"]

    def __init__(
        self,
        models: List[BaseSentimentModel],
        strategy: str = "weighted_average",
        weights: Optional[Dict[str, float]] = None
    ):
        """Initialize sentiment ensemble.

        Args:
            models: List of 2+ sentiment models
            strategy: Combination strategy (weighted_average, simple_average, voting)
            weights: Optional weight for each model (defaults to equal weights)

        Raises:
            ValueError: If fewer than 2 models or invalid strategy
        """
        if len(models) < 2:
            raise ValueError("Ensemble requires at least 2 models")

        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Invalid strategy. Must be one of: {self.VALID_STRATEGIES}")

        self.models = models
        self.strategy = strategy

        # Set weights
        if weights:
            self.weights = weights
        else:
            # Equal weights for all models
            equal_weight = 1.0 / len(models)
            self.weights = {
                model.get_model_info()["name"]: equal_weight
                for model in models
            }

        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment using ensemble of models.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult with ensemble prediction and per-model breakdown

        Raises:
            RuntimeError: If ensemble analysis fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            # Get predictions from all models
            results = []
            for model in self.models:
                try:
                    result = model.analyze(text)
                    results.append(result)
                except Exception as e:
                    print(f"Warning: Model {model.get_model_info()['name']} failed: {e}")
                    continue

            if not results:
                raise RuntimeError("All models failed to analyze text")

            # Combine results based on strategy
            if self.strategy == "weighted_average":
                return self._weighted_average(results)
            elif self.strategy == "simple_average":
                return self._simple_average(results)
            elif self.strategy == "voting":
                return self._voting(results)
            else:
                raise RuntimeError(f"Unknown strategy: {self.strategy}")

        except Exception as e:
            raise RuntimeError(f"Ensemble analysis failed: {e}") from e

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment of multiple texts using ensemble.

        Args:
            texts: List of texts to analyze

        Returns:
            List of SentimentResult objects

        Raises:
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        return [self.analyze(text) for text in texts]

    def _weighted_average(self, results: List[SentimentResult]) -> SentimentResult:
        """Combine results using weighted average of probabilities."""
        # Initialize probability accumulators
        weighted_probs = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

        # Accumulate weighted probabilities
        per_model_results = {}
        for result in results:
            model_name = result.metadata.get("model", "unknown")
            weight = self.weights.get(model_name, 1.0 / len(results))

            for label in ["positive", "negative", "neutral"]:
                weighted_probs[label] += result.probabilities[label] * weight

            per_model_results[model_name] = {
                "label": result.label,
                "confidence": result.confidence,
                "probabilities": result.probabilities
            }

        # Determine final label and confidence
        label = max(weighted_probs, key=weighted_probs.get)
        confidence = weighted_probs[label]

        return SentimentResult(
            label=label,
            confidence=confidence,
            probabilities=weighted_probs,
            metadata={
                "model": "SentimentEnsemble",
                "strategy": self.strategy,
                "num_models": len(results),
                "weights": self.weights,
                "per_model_results": per_model_results
            }
        )

    def _simple_average(self, results: List[SentimentResult]) -> SentimentResult:
        """Combine results using simple average (equal weights)."""
        # Initialize probability accumulators
        avg_probs = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

        # Accumulate probabilities
        per_model_results = {}
        for result in results:
            model_name = result.metadata.get("model", "unknown")

            for label in ["positive", "negative", "neutral"]:
                avg_probs[label] += result.probabilities[label]

            per_model_results[model_name] = {
                "label": result.label,
                "confidence": result.confidence,
                "probabilities": result.probabilities
            }

        # Average the probabilities
        n = len(results)
        avg_probs = {k: v / n for k, v in avg_probs.items()}

        # Determine final label and confidence
        label = max(avg_probs, key=avg_probs.get)
        confidence = avg_probs[label]

        return SentimentResult(
            label=label,
            confidence=confidence,
            probabilities=avg_probs,
            metadata={
                "model": "SentimentEnsemble",
                "strategy": self.strategy,
                "num_models": len(results),
                "per_model_results": per_model_results
            }
        )

    def _voting(self, results: List[SentimentResult]) -> SentimentResult:
        """Combine results using majority voting."""
        # Count votes for each label
        votes = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        per_model_results = {}

        for result in results:
            model_name = result.metadata.get("model", "unknown")
            weight = self.weights.get(model_name, 1.0 / len(results))

            votes[result.label] += weight

            per_model_results[model_name] = {
                "label": result.label,
                "confidence": result.confidence,
                "probabilities": result.probabilities
            }

        # Determine winner
        label = max(votes, key=votes.get)
        confidence = votes[label] / sum(votes.values())

        # Create probability distribution from votes
        total_votes = sum(votes.values())
        probabilities = {k: v / total_votes for k, v in votes.items()}

        return SentimentResult(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            metadata={
                "model": "SentimentEnsemble",
                "strategy": self.strategy,
                "num_models": len(results),
                "weights": self.weights,
                "votes": votes,
                "per_model_results": per_model_results
            }
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "name": "SentimentEnsemble",
            "version": "1.0",
            "type": "ensemble",
            "strategy": self.strategy,
            "num_models": len(self.models),
            "models": [model.get_model_info()["name"] for model in self.models],
            "weights": self.weights,
            "capabilities": ["analyze", "analyze_batch"]
        }
