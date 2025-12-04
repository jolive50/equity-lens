"""Configuration system for model selection and ensemble strategies.

JOSH's Component - Model Configuration System
Allows runtime selection of prediction/sentiment models and ensemble strategies.
"""
import logging
import os
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
        self.sentiment_models = ["FinBERT", "RoBERTa", "DeBERTa"]
        self.prediction_ensemble_strategy = "weighted_average"
        self.sentiment_ensemble_strategy = "weighted_average"
        self.prediction_model_weights = {"LSTM": 1.0}
        self.sentiment_model_weights = {"finbert": 0.5, "roberta": 0.3, "deberta": 0.2}
        self.use_ensemble = False
        self.use_sentiment_ensemble = False
        self.reflection_enabled = True
        self.confidence_threshold = 0.75
        self.enable_alpha_vantage_news = False
        self.alpha_vantage_max_items = 10
        self.alpha_vantage_api_key = None
        self.sentiment_av_weight = 0.3

        logger.info("Loaded default configuration")

    def _load_from_dict(self, config_dict: Dict[str, Any]):
        """Load configuration from dictionary."""
        self.prediction_models = config_dict.get("prediction_models", ["LSTM"])
        raw_sentiment_models = config_dict.get("sentiment_models", ["FinBERT", "RoBERTa", "DeBERTa"])
        self.prediction_ensemble_strategy = config_dict.get("prediction_ensemble_strategy", "weighted_average")
        self.sentiment_ensemble_strategy = config_dict.get("sentiment_ensemble_strategy", "weighted_average")
        self.prediction_model_weights = config_dict.get("prediction_model_weights", {})
        self.sentiment_model_weights = config_dict.get("sentiment_model_weights", {})
        self.use_ensemble = config_dict.get("use_ensemble", False)
        self.use_sentiment_ensemble = config_dict.get("use_sentiment_ensemble", False)
        self.reflection_enabled = config_dict.get("reflection_enabled", True)
        self.confidence_threshold = config_dict.get("confidence_threshold", 0.75)
        self.enable_alpha_vantage_news = config_dict.get("enable_alpha_vantage_news", False)
        self.alpha_vantage_max_items = config_dict.get("alpha_vantage_max_items", 10)
        self.alpha_vantage_api_key = config_dict.get("alpha_vantage_api_key")
        self.sentiment_av_weight = config_dict.get("sentiment_av_weight", 0.3)

        self.sentiment_models = self._normalize_sentiment_models(raw_sentiment_models)
        self.sentiment_model_weights = self._normalize_sentiment_weights(self.sentiment_model_weights)

        logger.info(
            "Loaded configuration with %d prediction models, %d sentiment models (alpha_vantage_news=%s)",
            len(self.prediction_models),
            len(self.sentiment_models),
            self.enable_alpha_vantage_news,
        )

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

    def _normalize_sentiment_models(self, models: List[str]) -> List[str]:
        """Keep only supported sentiment models and track Alpha Vantage usage."""
        supported = {
            "finbert": "FinBERT",
            "roberta": "RoBERTa",
            "deberta": "DeBERTa",
        }
        normalized: List[str] = []
        alpha_requested = bool(self.enable_alpha_vantage_news)

        for name in models:
            key = (name or "").strip().lower()
            if key in supported:
                normalized.append(supported[key])
            elif key in ("alpha_vantage", "alphavantage"):
                alpha_requested = True
            elif key:
                logger.warning(
                    "Sentiment model %s is not supported. Use FinBERT, RoBERTa, DeBERTa, or AlphaVantage for news.",
                    name,
                )

        if not normalized:
            normalized = ["FinBERT"]

        self.enable_alpha_vantage_news = alpha_requested
        return normalized

    def _normalize_sentiment_weights(self, weights: Dict[str, Any]) -> Dict[str, float]:
        """Normalize sentiment weights to match ensemble expectations (lowercase keys)."""
        norm: Dict[str, float] = {}
        for name, weight in (weights or {}).items():
            key = (name or "").strip().lower()
            if key in ("finbert", "roberta", "deberta"):
                try:
                    norm[key] = float(weight)
                except Exception:
                    logger.warning("Invalid weight for %s: %s", name, weight)

        if not norm:
            if self.sentiment_models:
                base = 1.0 / len(self.sentiment_models)
                norm = {m.lower(): base for m in self.sentiment_models}
            else:
                norm = {"finbert": 1.0}
        return norm

    def resolve_alpha_vantage_key(self) -> Optional[str]:
        """Return Alpha Vantage API key from config or environment."""
        key = self.alpha_vantage_api_key or os.getenv("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHAVANTAGE_KEY")
        return key.strip() if key else None

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
        from models.sentiment import (
            FinBertSentiment,
            DeBertaSentiment,
            RobertaSentiment,
            SentimentEnsemble,
        )
        logger.info("Building sentiment models: %s (ensemble=%s)", self.sentiment_models, self.use_sentiment_ensemble)

        def _build_model(name: str):
            key = (name or "").lower()
            logger.info("Attempting to create sentiment model: %s", name)
            if key == "finbert":
                model = FinBertSentiment()
                logger.info("Created FinBERT sentiment model")
                return model
            if key == "roberta":
                model = RobertaSentiment()
                logger.info("Created RoBERTa sentiment model")
                return model
            if key == "deberta":
                # Guard against missing deps for DeBERTa
                try:
                    import sentencepiece  # noqa: F401
                    import torch  # noqa: F401
                except Exception as dep_exc:
                    raise RuntimeError(f"DeBERTa dependencies missing: {dep_exc}") from dep_exc
                model = DeBertaSentiment()
                logger.info("Created DeBERTa sentiment model")
                return model
            raise ValueError(f"Unsupported sentiment model: {name}")

        if self.use_sentiment_ensemble and len(self.sentiment_models) >= 2:
            finbert_model = None
            deberta_model = None
            roberta_model = None

            logger.info("Configured sentiment models: %s", self.sentiment_models)
            for model_name in self.sentiment_models:
                try:
                    model = _build_model(model_name)
                    key = model_name.lower()
                    if key == "finbert":
                        finbert_model = model
                    elif key == "deberta":
                        deberta_model = model
                    elif key == "roberta":
                        roberta_model = model
                    logger.info("Created %s for sentiment ensemble", model_name)
                except Exception as e:
                    logger.warning("Failed to create sentiment model %s: %s", model_name, e)
                    continue

            if finbert_model or deberta_model or roberta_model:
                weights = self._normalize_sentiment_weights(self.sentiment_model_weights)
                av_mode = "blend" if getattr(self, "enable_alpha_vantage_news", False) else "ignore"
                av_weight = float(getattr(self, "sentiment_av_weight", 0.3) or 0.0)
                logger.info(
                    "Sentiment ensemble members loaded: finbert=%s, deberta=%s, roberta=%s; av_mode=%s av_weight=%.2f",
                    bool(finbert_model),
                    bool(deberta_model),
                    bool(roberta_model),
                    av_mode,
                    av_weight,
                )
                return SentimentEnsemble(
                    finbert=finbert_model,
                    deberta=deberta_model,
                    roberta=roberta_model,
                    weights=weights,
                    av_trust_mode=av_mode,
                    av_weight=av_weight,
                )

            logger.warning("No models loaded for ensemble, falling back to FinBERT")
            return FinBertSentiment()

        # Single model
        model_name = self.sentiment_models[0] if self.sentiment_models else "FinBERT"
        try:
            return _build_model(model_name)
        except Exception as e:
            logger.error("Failed to load sentiment model %s: %s", model_name, e)
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
  - DeBERTa
  - AlphaVantage

# Enable sentiment ensemble mode
use_sentiment_ensemble: true

# Sentiment ensemble strategy: weighted_average, simple_average, voting
sentiment_ensemble_strategy: weighted_average

# Sentiment model weights (lowercase keys required by ensemble code)
sentiment_model_weights:
  finbert: 0.4
  roberta: 0.3
  deberta: 0.3

# Alpha Vantage news feed for sentiment (requires ALPHA_VANTAGE_API_KEY env var)
enable_alpha_vantage_news: true
alpha_vantage_max_items: 10
# alpha_vantage_api_key: "YOUR_KEY"

# ============================================================
# WORKFLOW SETTINGS
# ============================================================

# Enable reflection agent for quality validation
reflection_enabled: true

# Confidence threshold for high-confidence recommendations
confidence_threshold: 0.75
"""
