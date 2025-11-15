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
        self.prediction_models = ["LSTM"]
        self.sentiment_models = ["FinBERT"]
        self.prediction_ensemble_strategy = "weighted_average"
        self.sentiment_ensemble_strategy = "weighted_average"
        self.prediction_model_weights = {"LSTM": 1.0}
        self.sentiment_model_weights = {"FinBERT": 1.0}
        self.use_ensemble = False
        self.use_sentiment_ensemble = False
        self.reflection_enabled = True
        self.confidence_threshold = 0.75

        logger.info("Loaded default configuration")

    def _load_from_dict(self, config_dict: Dict[str, Any]):
        """Load configuration from dictionary."""
        self.prediction_models = config_dict.get("prediction_models", ["LSTM"])
        self.sentiment_models = config_dict.get("sentiment_models", ["FinBERT"])
        self.prediction_ensemble_strategy = config_dict.get("prediction_ensemble_strategy", "weighted_average")
        self.sentiment_ensemble_strategy = config_dict.get("sentiment_ensemble_strategy", "weighted_average")
        self.prediction_model_weights = config_dict.get("prediction_model_weights", {})
        self.sentiment_model_weights = config_dict.get("sentiment_model_weights", {})
        self.use_ensemble = config_dict.get("use_ensemble", False)
        self.use_sentiment_ensemble = config_dict.get("use_sentiment_ensemble", False)
        self.reflection_enabled = config_dict.get("reflection_enabled", True)
        self.confidence_threshold = config_dict.get("confidence_threshold", 0.75)

        logger.info(f"Loaded configuration with {len(self.prediction_models)} prediction models, {len(self.sentiment_models)} sentiment models")

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
                elif model_name in ["GradientBoost", "XGBoost"]:
                    # XGBoost is the preferred name (GradientBoostModel uses XGBoost internally)
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
        elif model_name in ["GradientBoost", "XGBoost"]:
            # XGBoost is the preferred name (GradientBoostModel uses XGBoost internally)
            return GradientBoostModel()
        else:
            logger.warning(f"Unknown model {model_name}, using LSTM")
            return LSTMModel()

    def get_sentiment_model(self):
        """Get configured sentiment model instance using new API.

        Returns:
            BaseSentiment instance (new API) or None
        """
        from models.sentiment import FinBertSentiment, RobertaSentiment, SentimentEnsemble

        if self.use_sentiment_ensemble and len(self.sentiment_models) >= 2:
            # Create ensemble - only supports FinBERT, RoBERTa (new API models)
            finbert_model = None
            roberta_model = None

            for model_name in self.sentiment_models:
                try:
                    if model_name == "FinBERT":
                        finbert_model = FinBertSentiment()
                        logger.info("Created FinBERT for ensemble")
                    elif model_name == "RoBERTa":
                        roberta_model = RobertaSentiment()
                        logger.info("Created RoBERTa for ensemble")
                    else:
                        logger.warning(f"{model_name} not supported - only FinBERT and RoBERTa available")
                except Exception as e:
                    logger.warning(f"Failed to create sentiment model {model_name}: {e}")
                    continue

            # Create ensemble if we have at least one model
            if finbert_model or roberta_model:
                return SentimentEnsemble(
                    finbert=finbert_model,
                    roberta=roberta_model,
                    weights=self.sentiment_model_weights
                )
            else:
                logger.warning("No models loaded for ensemble, falling back to FinBERT")
                return FinBertSentiment()

        # Single model
        model_name = self.sentiment_models[0] if self.sentiment_models else "FinBERT"

        try:
            if model_name == "FinBERT":
                return FinBertSentiment()
            elif model_name == "RoBERTa":
                return RobertaSentiment()
            else:
                logger.warning(f"Model {model_name} not supported (only FinBERT/RoBERTa), using FinBERT")
                return FinBertSentiment()
        except Exception as e:
            logger.error(f"Failed to load sentiment model {model_name}: {e}")
            return None


# Example config.yaml structure
DEFAULT_CONFIG_YAML = """
# FreshStart Workflow Configuration
# Enable all models in ensemble mode for maximum accuracy

# ============================================================
# PREDICTION MODELS (Pam's Models)
# ============================================================
prediction_models:
  - LSTM
  - GRU
  - XGBoost

# Enable ensemble mode (combines all prediction models)
use_ensemble: true

# Ensemble strategy: weighted_average, simple_average, voting
prediction_ensemble_strategy: weighted_average

# Model weights for weighted_average strategy
prediction_model_weights:
  LSTM: 0.4
  GRU: 0.35
  XGBoost: 0.25

# ============================================================
# SENTIMENT MODELS (Tae's Models)
# ============================================================
sentiment_models:
  - FinBERT
  - RoBERTa
  - VADER
  - TextBlob

# Enable sentiment ensemble mode
use_sentiment_ensemble: true

# Sentiment ensemble strategy: weighted_average, simple_average, voting
sentiment_ensemble_strategy: weighted_average

# Sentiment model weights
sentiment_model_weights:
  FinBERT: 0.4
  RoBERTa: 0.3
  VADER: 0.2
  TextBlob: 0.1

# ============================================================
# WORKFLOW SETTINGS
# ============================================================

# Enable reflection agent for quality validation
reflection_enabled: true

# Confidence threshold for high-confidence recommendations
confidence_threshold: 0.75
"""
