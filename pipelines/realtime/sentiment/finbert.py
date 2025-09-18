"""FinBERT sentiment scoring scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional


@dataclass(slots=True)
class FinBERTConfig:
    """Configuration for running FinBERT inference over financial text."""

    model_name: str = "ProsusAI/finbert"
    batch_size: int = 16
    device: Literal["cpu", "cuda"] = "cpu"
    destination: str = "data/realtime/feature_store/sentiment/finbert/"
    confidence_threshold: float = 0.0
    max_length: int = 512
    prefetch_batches: int = 2
    tokenizer_kwargs: Optional[dict] = None


def score_with_finbert(text_batches: Iterable[list[str]], config: FinBERTConfig) -> None:
    """Score incoming text payloads with FinBERT and write sentiment features."""

    raise NotImplementedError(
        "Implement transformers pipeline loading, GPU/CPU selection, and Parquet feature emission."
    )
