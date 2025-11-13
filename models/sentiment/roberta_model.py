# FreshStart/models/sentiment/roberta_model.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# --- fetcher import (YF only, per design) ---
from data.fetchers.news_data import fetch_news_yf_only

from .base_sentiment import BaseSentiment, normalize_probs


DEFAULT_ROBERTA = os.environ.get(
    "ROBERTA_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)
# Other viable 3-class checkpoints:
#   - "cardiffnlp/twitter-roberta-base-sentiment"
#   - "distilbert-base-uncased-finetuned-sst-2-english" (binary; not 3-class)


class RobertaSentiment(BaseSentiment):
    """
    Generic 3-class sentiment wrapper (TensorFlow/HF).
    Maps model labels to {"negative","neutral","positive"} when possible.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ROBERTA,
        *,
        headline_first: float = 0.65,
        min_body_words: int = 60,
        neutral_cap: Optional[float] = 0.8,
        neg_gate: Optional[float] = None,
        max_length: int = 514,
        use_mixed_precision: bool = False,
        temperature: float = 1.0,
        prefer_positive: float = 0.03,
        class_weights: Optional[Sequence[float]] = (1.0, 0.95, 1.05),  # NEG, NEU, POS
    ) -> None:
        super().__init__(
            provider="roberta",
            headline_first=headline_first,
            min_body_words=min_body_words,
            neutral_cap=neutral_cap,
            neg_gate=neg_gate,
        )
        self.model_name = model_name
        self.max_length = max_length
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[TFAutoModelForSequenceClassification] = None
        self._label_ix2key: Optional[Dict[int, str]] = None
        self._memo: Dict[str, Dict[str, float]] = {}

        self.temperature: float = float(temperature) if temperature and temperature > 0 else 1.0
        self.prefer_positive: float = float(prefer_positive or 0.0)
        self._class_weights: Optional[np.ndarray] = None
        if class_weights is not None:
            arr = np.asarray(list(class_weights), dtype="float32")
            if arr.shape[0] == 3:
                self._class_weights = arr

        if use_mixed_precision:
            try:
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
            except Exception:
                pass

    # ---------------- lifecycle ----------------
    def load(self) -> None:
        if self._tokenizer and self._model and self._label_ix2key:
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            # Some RoBERTa checkpoints are PT-first → from_pt=True ensures TF weights are built
            self._model = TFAutoModelForSequenceClassification.from_pretrained(
                self.model_name, from_pt=True
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load TF model '{self.model_name}'. "
                "Install: pip install tensorflow tf-keras transformers"
            ) from e

        # Label mapping
        cfg = self._model.config
        id2label = getattr(cfg, "id2label", None) or {}
        ix2name: Dict[int, str] = {}
        for k, v in id2label.items():
            try:
                ix2name[int(k)] = str(v)
            except Exception:
                pass

        def norm_key(name: str) -> str:
            n = name.strip().lower()
            if "neg" in n or n == "bearish":
                return "negative"
            if "pos" in n or n == "bullish":
                return "positive"
            if "neu" in n:
                return "neutral"
            # common twitter-roberta labels: negative, neutral, positive
            if n in ("negative", "neutral", "positive"):
                return n
            return "neutral"

        self._label_ix2key = (
            {ix: norm_key(name) for ix, name in ix2name.items()}
            if ix2name
            else {0: "negative", 1: "neutral", 2: "positive"}
        )

    # ---------------- internal helpers ----------------
    def _forward_logits(self, texts: Sequence[str]) -> np.ndarray:
        assert self._tokenizer is not None and self._model is not None
        enc = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="tf",
        )
        out = self._model(**enc, training=False)
        logits = out.logits
        if self.temperature != 1.0:
            logits = logits / self.temperature
        return logits.numpy()

    def _logits_to_probs_dicts(self, logits: np.ndarray) -> List[Dict[str, float]]:
        assert self._label_ix2key is not None
        probs = tf.nn.softmax(logits, axis=-1).numpy().tolist()
        out: List[Dict[str, float]] = []

        for row in probs:
            # map raw probs to NEG/NEU/POS buckets
            p: Dict[str, float] = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
            for ix, val in enumerate(row):
                key = self._label_ix2key.get(ix, "neutral")
                p[key] += float(val)

            # apply class weights (NEG, NEU, POS) → downweight neutral, upweight positive
            if self._class_weights is not None:
                w_neg, w_neu, w_pos = [float(x) for x in self._class_weights]
                p["negative"] *= w_neg
                p["neutral"]  *= w_neu
                p["positive"] *= w_pos

            p = normalize_probs(p)

            # gentle positive bias (prefer_positive) – take a little mass from neg+neu
            if self.prefer_positive > 0.0:
                eps = float(self.prefer_positive)
                neg_neu = p["negative"] + p["neutral"]
                if neg_neu > 0:
                    p["positive"] += eps
                    if p["negative"] > 0 or p["neutral"] > 0:
                        p["negative"] -= eps * (p["negative"] / neg_neu)
                        p["neutral"]  -= eps * (p["neutral"] / neg_neu)
                    p["negative"] = max(p["negative"], 0.0)
                    p["neutral"]  = max(p["neutral"], 0.0)

            p = normalize_probs(p)
            out.append(p)

        return out

    # ---------------- BaseSentiment hook ----------------
    def _predict_text(self, text: str) -> Dict[str, float]:
        t = (text or "").strip()
        if not t:
            return {"negative": 0.0, "neutral": 1.0, "positive": 0.0}
        memo = self._memo.get(t)
        if memo is not None:
            return memo
        self.load()
        logits = self._forward_logits([t])
        p = self._logits_to_probs_dicts(logits)[0]
        self._memo[t] = p
        return p


# ---------------- CLI with YF-only fetcher ----------------
def _cli() -> None:
    """
    Run RoBERTa (TF) on freshly fetched Yahoo Finance news.

    Example:
      python -m FreshStart.models.sentiment.roberta_model AAPL --max-items 15
    """
    import argparse
    from textwrap import shorten

    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", help="Stock ticker, e.g., AAPL, MSFT, NVDA")
    ap.add_argument("--model", default=DEFAULT_ROBERTA, help="HF repo or local path")
    ap.add_argument("--neutral-cap", type=float, default=0.8)
    ap.add_argument("--neg-gate", type=float, default=None)
    ap.add_argument("--max-items", type=int, default=25, help="Max YF items to fetch (default 25)")
    ap.add_argument("--mixed-precision", action="store_true", help="Enable TF mixed precision")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--prefer-positive", type=float, default=0.03)
    ap.add_argument(
        "--class-weights",
        type=float,
        nargs=3,
        metavar=("W_NEG", "W_NEU", "W_POS"),
        default=(1.0, 0.95, 1.05),
        help="Per-class weights for (negative, neutral, positive)",
    )
    args = ap.parse_args()

    # Init model
    model = RobertaSentiment(
        model_name=args.model,
        neutral_cap=args.neutral_cap,
        neg_gate=args.neg_gate,
        use_mixed_precision=args.mixed_precision,
        temperature=args.temperature,
        prefer_positive=args.prefer_positive,
        class_weights=args.class_weights,
    )
    model.load()

    # Fetch YF-only articles
    print(f"\nFetching Yahoo Finance news for {args.ticker} (max {args.max_items})\n")
    try:
        items = fetch_news_yf_only(args.ticker, max_items=args.max_items)
    except Exception as e:
        print(f"Failed to fetch YF news: {e}")
        items = []

    print(f"Returned: {len(items)} articles\n")

    # Score using the fetcher schema
    results = model.predict_batch_from_fetcher(items)
    out = [r.to_dict() for r in results]

    # Compact YF summary
    for i, r in enumerate(results, 1):
        src = (r.meta.get("source") or "yahoo").strip().lower()
        if src in ("finance.yahoo.com", "yahoo/unknown"):
            src = "yahoo"
        tpub = r.meta.get("time_published") or "-"
        title = shorten((items[i - 1].get("title") or "").strip(), width=90, placeholder=" [...]")
        lab = r.label.name.capitalize()
        print(f"YF {i:02d}. [{lab:7}] [{src:8}] {tpub}  {title}")

    # Full JSON
    print("\n--- JSON OUTPUT ---")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
