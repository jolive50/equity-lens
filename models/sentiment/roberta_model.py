import os
from typing import Dict, Any, Optional, List
import numpy as np
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from .base_sentiment import BaseSentimentModel, SentimentResult


class RoBERTaModel(BaseSentimentModel):
    """RoBERTa sentiment model with fine-tuning capability.

    Uses cardiffnlp/twitter-roberta-base-sentiment by default.
    Supports fine-tuning on custom financial datasets.
    """

    LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

    def __init__(
        self,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
        weights_path: Optional[str] = None
    ):
        """Initialize RoBERTa model.

        Args:
            model_name: HuggingFace model identifier
            weights_path: Path to fine-tuned weights (optional)

        Raises:
            RuntimeError: If model loading fails
        """
        self.model_name = model_name
        self.weights_path = weights_path

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = TFAutoModelForSequenceClassification.from_pretrained(
                model_name,
                from_pt=True
            )

            if weights_path and os.path.exists(weights_path):
                self.model.load_weights(weights_path)

        except Exception as e:
            raise RuntimeError(f"Failed to load RoBERTa: {e}") from e

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of single text."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="tf",
                truncation=True,
                padding=True,
                max_length=512
            )

            outputs = self.model(**inputs)
            probabilities = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]

            label_idx = int(np.argmax(probabilities))
            label = self.LABEL_MAP[label_idx]
            confidence = float(probabilities[label_idx])

            return SentimentResult(
                label=label,
                confidence=confidence,
                probabilities={
                    "negative": float(probabilities[0]),
                    "neutral": float(probabilities[1]),
                    "positive": float(probabilities[2])
                },
                metadata={
                    "model": "RoBERTa",
                    "model_name": self.model_name,
                    "fine_tuned": self.weights_path is not None
                }
            )

        except Exception as e:
            raise RuntimeError(f"RoBERTa analysis failed: {e}") from e

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment of multiple texts efficiently."""
        if not texts:
            raise ValueError("Texts list cannot be empty")

        try:
            inputs = self.tokenizer(
                texts,
                return_tensors="tf",
                truncation=True,
                padding=True,
                max_length=512
            )

            outputs = self.model(**inputs)
            probabilities = tf.nn.softmax(outputs.logits, axis=-1).numpy()

            results = []
            for probs in probabilities:
                label_idx = int(np.argmax(probs))
                label = self.LABEL_MAP[label_idx]
                confidence = float(probs[label_idx])

                results.append(SentimentResult(
                    label=label,
                    confidence=confidence,
                    probabilities={
                        "negative": float(probs[0]),
                        "neutral": float(probs[1]),
                        "positive": float(probs[2])
                    },
                    metadata={
                        "model": "RoBERTa",
                        "model_name": self.model_name,
                        "fine_tuned": self.weights_path is not None
                    }
                ))

            return results

        except Exception as e:
            raise RuntimeError(f"RoBERTa batch analysis failed: {e}") from e

    def fine_tune(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        output_dir: str = "./roberta_finetuned"
    ) -> Dict[str, Any]:
        """Fine-tune RoBERTa on custom dataset.

        Args:
            train_texts: Training texts
            train_labels: Training labels (0=negative, 1=neutral, 2=positive)
            val_texts: Validation texts (optional)
            val_labels: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            output_dir: Directory to save fine-tuned weights

        Returns:
            Training history dictionary

        Raises:
            ValueError: If input validation fails
        """
        if len(train_texts) != len(train_labels):
            raise ValueError("Texts and labels must have same length")

        # Prepare training data
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="tf"
        )

        train_dataset = tf.data.Dataset.from_tensor_slices((
            dict(train_encodings),
            train_labels
        )).batch(batch_size)

        # Prepare validation data if provided
        val_dataset = None
        if val_texts and val_labels:
            if len(val_texts) != len(val_labels):
                raise ValueError("Validation texts and labels must have same length")

            val_encodings = self.tokenizer(
                val_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="tf"
            )

            val_dataset = tf.data.Dataset.from_tensor_slices((
                dict(val_encodings),
                val_labels
            )).batch(batch_size)

        # Configure optimizer and loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Train model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            verbose=1
        )

        # Save fine-tuned weights
        os.makedirs(output_dir, exist_ok=True)
        weights_file = os.path.join(output_dir, "roberta_weights.h5")
        self.model.save_weights(weights_file)
        self.weights_path = weights_file

        return {
            "epochs": epochs,
            "final_train_loss": float(history.history["loss"][-1]),
            "final_train_accuracy": float(history.history["accuracy"][-1]),
            "final_val_loss": float(history.history["val_loss"][-1]) if val_dataset else None,
            "final_val_accuracy": float(history.history["val_accuracy"][-1]) if val_dataset else None,
            "weights_saved_to": weights_file
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "name": "RoBERTa",
            "version": self.model_name,
            "type": "transformer",
            "capabilities": ["analyze", "analyze_batch", "fine_tune"],
            "fine_tuned": self.weights_path is not None,
            "weights_path": self.weights_path
        }
