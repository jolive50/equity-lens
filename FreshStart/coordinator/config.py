"""Model configuration system for runtime model selection."""

import yaml
import logging
from typing import Dict, List, Any
from pathlib import Path


logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration for model selection and ensemble strategies.

    Loads configuration from YAML files to control which prediction and
    sentiment models to use, ensemble strategies, and model weights.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize model configuration.

        Args:
            config_path: Path to YAML configuration file

        Raises:
            FileNotFoundError: If config file not found
            yaml.YAMLError: If config file invalid
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        logger.info(f"ModelConfig loaded from {config_path}")

    def _load_config(self, path: str) -> Dict:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If file not found
            yaml.YAMLError: If YAML invalid
        """
        config_file = Path(path)
        if not config_file.exists():
            logger.warning(f"Config file {path} not found, using defaults")
            return self._get_default_config()

        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse config file: {e}")
            raise

    def _get_default_config(self) -> Dict:
        """Get default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "prediction": {
                "mode": "single",
                "models": ["lstm"],
                "strategy": "weighted_average",
                "weights": {
                    "lstm": 1.0
                }
            },
            "sentiment": {
                "mode": "single",
                "models": ["finbert"],
                "strategy": "weighted_average",
                "weights": {
                    "finbert": 1.0
                }
            }
        }

    def get_prediction_models(self) -> List[str]:
        """Get list of prediction models to use.

        Returns:
            List of model names (e.g., ["lstm", "gru"])
        """
        return self.config.get("prediction", {}).get("models", ["lstm"])

    def get_sentiment_models(self) -> List[str]:
        """Get list of sentiment models to use.

        Returns:
            List of model names (e.g., ["finbert", "roberta"])
        """
        return self.config.get("sentiment", {}).get("models", ["finbert"])

    def get_prediction_mode(self) -> str:
        """Get prediction mode (single or ensemble).

        Returns:
            Mode string ("single" or "ensemble")
        """
        return self.config.get("prediction", {}).get("mode", "single")

    def get_sentiment_mode(self) -> str:
        """Get sentiment mode (single or ensemble).

        Returns:
            Mode string ("single" or "ensemble")
        """
        return self.config.get("sentiment", {}).get("mode", "single")

    def get_ensemble_strategy(self, model_type: str) -> str:
        """Get ensemble strategy for model type.

        Args:
            model_type: Type of model ("prediction" or "sentiment")

        Returns:
            Strategy string (e.g., "weighted_average", "voting")
        """
        return self.config.get(model_type, {}).get("strategy", "weighted_average")

    def get_model_weights(self, model_type: str) -> Dict[str, float]:
        """Get model weights for ensemble.

        Args:
            model_type: Type of model ("prediction" or "sentiment")

        Returns:
            Dictionary mapping model names to weights
        """
        return self.config.get(model_type, {}).get("weights", {})

    def validate(self) -> bool:
        """Validate configuration.

        Returns:
            True if configuration valid, False otherwise
        """
        # Check required sections
        if "prediction" not in self.config or "sentiment" not in self.config:
            logger.error("Config missing required sections")
            return False

        # Validate prediction config
        pred_models = self.get_prediction_models()
        if not pred_models:
            logger.error("No prediction models specified")
            return False

        # Validate sentiment config
        sent_models = self.get_sentiment_models()
        if not sent_models:
            logger.error("No sentiment models specified")
            return False

        # Validate weights sum to 1.0 for ensemble mode
        if self.get_prediction_mode() == "ensemble":
            pred_weights = self.get_model_weights("prediction")
            weight_sum = sum(pred_weights.values())
            if abs(weight_sum - 1.0) > 0.01:
                logger.warning(
                    f"Prediction weights sum to {weight_sum:.2f}, expected 1.0"
                )

        if self.get_sentiment_mode() == "ensemble":
            sent_weights = self.get_model_weights("sentiment")
            weight_sum = sum(sent_weights.values())
            if abs(weight_sum - 1.0) > 0.01:
                logger.warning(
                    f"Sentiment weights sum to {weight_sum:.2f}, expected 1.0"
                )

        return True
