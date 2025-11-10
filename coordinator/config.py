"""Configuration system for model selection and ensemble strategies.

JOSH's Component - Model Configuration System
Allows runtime selection of prediction/sentiment models and ensemble strategies.
"""
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class WorkflowConfig:
    """Configuration for FreshStart workflow."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize workflow configuration.

        Args:
            config_dict: Optional configuration dictionary
        """
        if config_dict:
            self._load_from_dict(config_dict)
        else:
            self._load_defaults()

    def _load_defaults(self):
        """Load default configuration."""
        self.prediction_models = ["LSTM"]  # Default to LSTM only
        self.sentiment_models = []  # TAE's models (placeholder)
        self.prediction_ensemble_strategy = "weighted_average"
        self.prediction_model_weights = {"LSTM": 1.0}
        self.use_ensemble = False
        self.reflection_enabled = True
        self.confidence_threshold = 0.95

        logger.info("Loaded default configuration")

    def _load_from_dict(self, config_dict: Dict[str, Any]):
        """Load configuration from dictionary."""
        self.prediction_models = config_dict.get("prediction_models", ["LSTM"])
        self.sentiment_models = config_dict.get("sentiment_models", [])
        self.prediction_ensemble_strategy = config_dict.get("ensemble_strategy", "weighted_average")
        self.prediction_model_weights = config_dict.get("model_weights", {})
        self.use_ensemble = config_dict.get("use_ensemble", False)
        self.reflection_enabled = config_dict.get("reflection_enabled", True)
        self.confidence_threshold = config_dict.get("confidence_threshold", 0.95)

        logger.info(f"Loaded configuration with {len(self.prediction_models)} prediction models")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "WorkflowConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML config file

        Returns:
            WorkflowConfig instance
        """
        path = Path(yaml_path)
        if not path.exists():
            logger.warning(f"Config file not found: {yaml_path}, using defaults")
            return cls()

        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(config_dict)

    def get_prediction_model(self):
        """Get configured prediction model instance.

        Returns:
            BasePredictionModel instance or None
        """
        from models.prediction.lstm_model import LSTMModel
        from models.prediction.gru_model import GRUModel
        from models.prediction.gradient_boost_model import GradientBoostModel
        from models.prediction.ensemble import PredictionEnsemble

        if self.use_ensemble and len(self.prediction_models) >= 2:
            # Create ensemble
            models = []
            for model_name in self.prediction_models:
                if model_name == "LSTM":
                    models.append(LSTMModel())
                elif model_name == "GRU":
                    models.append(GRUModel())
                elif model_name == "GradientBoost":
                    models.append(GradientBoostModel())

            if len(models) >= 2:
                return PredictionEnsemble(
                    models=models,
                    strategy=self.prediction_ensemble_strategy,
                    weights=self.prediction_model_weights
                )
            else:
                logger.warning("Not enough models for ensemble, using single model")
                return models[0] if models else LSTMModel()

        # Single model
        model_name = self.prediction_models[0] if self.prediction_models else "LSTM"

        if model_name == "LSTM":
            return LSTMModel()
        elif model_name == "GRU":
            return GRUModel()
        elif model_name == "GradientBoost":
            return GradientBoostModel()
        else:
            logger.warning(f"Unknown model {model_name}, using LSTM")
            return LSTMModel()


# Example config.yaml structure
DEFAULT_CONFIG_YAML = """
# FreshStart Workflow Configuration

# Prediction models to use
prediction_models:
  - LSTM
  # - GRU
  # - GradientBoost

# Whether to use ensemble (requires 2+ models)
use_ensemble: false

# Ensemble strategy: weighted_average, simple_average, voting
ensemble_strategy: weighted_average

# Model weights for weighted_average strategy
model_weights:
  LSTM: 0.5
  GRU: 0.3
  GradientBoost: 0.2

# Enable reflection agent for quality validation
reflection_enabled: true

# Confidence threshold for recommendations
confidence_threshold: 0.95

# Sentiment models (TAE's responsibility - placeholder)
sentiment_models: []
"""
