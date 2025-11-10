import os
from typing import Dict, Any, Optional, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .base_sentiment import BaseSentimentModel, SentimentResult


class _FinBERTDataset(Dataset):
    """Simple dataset for FinBERT fine-tuning."""

    def __init__(self, tokenizer: AutoTokenizer, texts: List[str], labels: List[int]):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class FinBERTModel(BaseSentimentModel):
    """FinBERT sentiment model powered by PyTorch transformers."""

    LABEL_MAP = {0: "positive", 1: "negative", 2: "neutral"}

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        weights_path: Optional[str] = None,
    ):
        """Initialize FinBERT model."""
        self.model_name = model_name
        self.weights_path = weights_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            tokenizer_source = (
                weights_path
                if weights_path and os.path.isdir(weights_path)
                else model_name
            )
            model_source = tokenizer_source

            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_source)

            if weights_path and os.path.isfile(weights_path):
                state_dict = torch.load(weights_path, map_location="cpu")
                self.model.load_state_dict(state_dict)

            self.model.to(self.device)
            self.model.eval()

        except Exception as e:
            raise RuntimeError(f"Failed to load FinBERT: {e}") from e

    def analyze(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text snippet."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(
                    outputs.logits, dim=-1
                ).cpu().numpy()[0]

            label_idx = int(np.argmax(probabilities))
            label = self.LABEL_MAP[label_idx]
            confidence = float(probabilities[label_idx])

            return SentimentResult(
                label=label,
                confidence=confidence,
                probabilities={
                    "positive": float(probabilities[0]),
                    "negative": float(probabilities[1]),
                    "neutral": float(probabilities[2]),
                },
                metadata={
                    "model": "FinBERT",
                    "model_name": self.model_name,
                    "fine_tuned": self.weights_path is not None,
                },
            )

        except Exception as e:
            raise RuntimeError(f"FinBERT analysis failed: {e}") from e

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment of multiple texts."""
        if not texts:
            raise ValueError("Texts list cannot be empty")

        try:
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(
                    outputs.logits, dim=-1
                ).cpu().numpy()

            results = []
            for probs in probabilities:
                label_idx = int(np.argmax(probs))
                label = self.LABEL_MAP[label_idx]
                confidence = float(probs[label_idx])

                results.append(
                    SentimentResult(
                        label=label,
                        confidence=confidence,
                        probabilities={
                            "positive": float(probs[0]),
                            "negative": float(probs[1]),
                            "neutral": float(probs[2]),
                        },
                        metadata={
                            "model": "FinBERT",
                            "model_name": self.model_name,
                            "fine_tuned": self.weights_path is not None,
                        },
                    )
                )

            return results

        except Exception as e:
            raise RuntimeError(f"FinBERT batch analysis failed: {e}") from e

    def fine_tune(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        output_dir: str = "./finbert_finetuned",
    ) -> Dict[str, Any]:
        """Fine-tune FinBERT on a labeled dataset."""
        if len(train_texts) != len(train_labels):
            raise ValueError("Texts and labels must have same length")
        if val_texts and val_labels and len(val_texts) != len(val_labels):
            raise ValueError("Validation texts and labels must have same length")

        train_dataset = _FinBERTDataset(self.tokenizer, train_texts, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_texts and val_labels:
            val_dataset = _FinBERTDataset(self.tokenizer, val_texts, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        history = {
            "epochs": epochs,
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        for _ in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()

                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch["labels"].size(0)
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)

            train_loss = running_loss / max(total, 1)
            train_acc = correct / max(total, 1)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)

            if val_loader:
                self.model.eval()
                val_loss_total = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch in val_loader:
                        batch = {k: v.to(self.device) for k, v in batch.items()}
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        val_loss_total += loss.item() * batch["labels"].size(0)
                        preds = outputs.logits.argmax(dim=-1)
                        val_correct += (preds == batch["labels"]).sum().item()
                        val_total += batch["labels"].size(0)

                history["val_loss"].append(
                    val_loss_total / max(val_total, 1)
                )
                history["val_accuracy"].append(
                    val_correct / max(val_total, 1)
                )

        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.weights_path = output_dir

        return {
            "epochs": epochs,
            "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
            "final_train_accuracy": history["train_accuracy"][-1]
            if history["train_accuracy"]
            else None,
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "final_val_accuracy": history["val_accuracy"][-1]
            if history["val_accuracy"]
            else None,
            "weights_saved_to": output_dir,
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "name": "FinBERT",
            "version": self.model_name,
            "type": "transformer",
            "backend": "pytorch",
            "capabilities": ["analyze", "analyze_batch", "fine_tune"],
            "fine_tuned": self.weights_path is not None,
            "weights_path": self.weights_path,
        }
