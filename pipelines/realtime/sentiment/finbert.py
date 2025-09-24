"""FinBERT sentiment scoring using Hugging Face transformers pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Literal, Optional, Sequence, Tuple
from datetime import datetime

import pandas as pd

try:  # pragma: no cover - optional at runtime until user installs transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
    import torch
except Exception:  # pragma: no cover - allow module import without deps
    AutoModelForSequenceClassification = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    TextClassificationPipeline = None  # type: ignore[assignment]
    torch = None  # type: ignore[assignment]

_LOGGER = logging.getLogger(__name__)


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


def _batched(iterable: Sequence[str], batch_size: int) -> Iterator[List[str]]:
    current: List[str] = []
    for item in iterable:
        current.append(item)
        if len(current) >= batch_size:
            yield current
            current = []
    if current:
        yield current


def score_with_finbert(texts: Sequence[str], config: FinBERTConfig) -> Path:
    """Score texts with FinBERT and persist a Parquet with probabilities.

    Returns:
        Path to the written Parquet file.
    """

    if AutoModelForSequenceClassification is None or AutoTokenizer is None or TextClassificationPipeline is None:
        raise RuntimeError(
            "transformers is not installed. Please `pip install transformers torch` to run FinBERT."
        )

    destination = Path(config.destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    device_index = -1
    if config.device == "cuda" and torch is not None and torch.cuda.is_available():
        device_index = 0

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name)
    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device_index,
        top_k=None,
        truncation=True,
        max_length=config.max_length,
        function_to_apply="softmax",
    )

    labels = list(model.config.id2label.values())

    rows: List[dict] = []
    scored_at_iso = datetime.utcnow().isoformat() + "Z"
    for batch in _batched(list(texts), config.batch_size):
        outputs = pipe(batch)  # List[List[{'label': str, 'score': float}]]
        for text, dist in zip(batch, outputs):
            scores = {item["label"].lower(): float(item["score"]) for item in dist}
            pred_label = max(scores, key=scores.get)
            pred_score = scores[pred_label]
            if pred_score < config.confidence_threshold:
                pred_label = "neutral"
            rows.append(
                {
                    "text": text,
                    "positive": scores.get("positive", 0.0),
                    "neutral": scores.get("neutral", 0.0),
                    "negative": scores.get("negative", 0.0),
                    "label": pred_label,
                    "score": pred_score,
                    "scored_at": scored_at_iso,
                }
            )

    frame = pd.DataFrame(rows)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_parquet = destination / f"finbert_scores_{ts}.parquet"
    try:
        frame.to_parquet(out_parquet, engine="pyarrow")
    except ModuleNotFoundError:  # pragma: no cover
        frame.to_parquet(out_parquet)

    # Also emit JSONL for lightweight readers
    out_jsonl = destination / f"finbert_scores_{ts}.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fp:
        for rec in rows:
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

    _LOGGER.info("Wrote %d FinBERT rows to %s and %s", len(frame), out_parquet, out_jsonl)
    return out_parquet
