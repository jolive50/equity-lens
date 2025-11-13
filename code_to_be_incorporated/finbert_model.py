# FreshStart/models/sentiment/finbert_model.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# --- fetcher import (YF only, per design) ---
try:
    from FreshStart.data.fetchers.news_data import fetch_news_yf_only
except ModuleNotFoundError:
    from ...data.fetchers.news_data import fetch_news_yf_only

from .base_sentiment import BaseSentiment, normalize_probs

DEFAULT_FINBERT = os.environ.get("FINBERT_MODEL", "ProsusAI/finbert")


class FinBertSentiment(BaseSentiment):
    """
    FinBERT wrapper (TensorFlow) that returns 3-class probabilities:
      {"negative": p0, "neutral": p1, "positive": p2}

    Tuning knobs (from your fine-tune):
      - temperature (default 0.85): softmax over logits/temperature (sharper when <1)
      - prefer_positive (default 0.03): small additive bias to positive before renorm
      - neutral_cap (default 0.75): clamp neutral mass after title/body mixing
    """

    def __init__(
        self,
        model_name: str = DEFAULT_FINBERT,
        *,
        headline_first: float = 0.65,
        min_body_words: int = 60,
        neutral_cap: Optional[float] = 0.75,   # tuned default
        neg_gate: Optional[float] = None,
        max_length: int = 514,
        use_mixed_precision: bool = False,
        temperature: float = 0.85,             # tuned default
        prefer_positive: float = 0.03,         # tuned default
    ) -> None:
        super().__init__(
            provider="finbert",
            headline_first=headline_first,
            min_body_words=min_body_words,
            neutral_cap=neutral_cap,
            neg_gate=neg_gate,
        )
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = float(max(1e-6, temperature))
        self.prefer_positive = float(max(0.0, prefer_positive))
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[TFAutoModelForSequenceClassification] = None
        self._label_ix2key: Optional[Dict[int, str]] = None
        self._memo: Dict[str, Dict[str, float]] = {}

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
            # Many checkpoints (e.g., ProsusAI/finbert) are PT-only → load with from_pt=True
            self._model = TFAutoModelForSequenceClassification.from_pretrained(
                self.model_name, from_pt=True
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FinBERT TF model '{self.model_name}'. "
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
        return out.logits.numpy()  # [B, 3]

    def _softmax_with_temperature(self, logits: np.ndarray) -> np.ndarray:
        # logits shape: [B, C]
        z = logits / self.temperature
        z = z - np.max(z, axis=-1, keepdims=True)
        e = np.exp(z)
        return e / np.clip(np.sum(e, axis=-1, keepdims=True), 1e-9, None)

    def _logits_to_probs_dicts(self, logits: np.ndarray) -> List[Dict[str, float]]:
        assert self._label_ix2key is not None
        # temperature-scaled softmax
        probs = self._softmax_with_temperature(logits).tolist()  # [B, 3]
        out: List[Dict[str, float]] = []
        for row in probs:
            p: Dict[str, float] = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
            for ix, val in enumerate(row):
                key = self._label_ix2key.get(ix, "neutral")
                p[key] += float(val)

            # small positive preference, then renormalize
            if self.prefer_positive > 0:
                p["positive"] += self.prefer_positive
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
    Run FinBERT (TF) on freshly fetched Yahoo Finance news.

    Example:
      python -m FreshStart.models.sentiment.finbert_model AAPL --max-items 15
    """
    import argparse
    from textwrap import shorten

    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", help="Stock ticker, e.g., AAPL, MSFT, NVDA")
    ap.add_argument("--neutral-cap", type=float, default=0.75)
    ap.add_argument("--neg-gate", type=float, default=None)
    ap.add_argument("--max-items", type=int, default=25, help="Max YF items to fetch (default 25)")
    ap.add_argument("--mixed-precision", action="store_true", help="Enable TF mixed precision")
    ap.add_argument("--temperature", type=float, default=0.85, help="Softmax temperature (<1 sharper)")
    ap.add_argument("--prefer-positive", type=float, default=0.03, help="Additive bias to positive prob")
    args = ap.parse_args()

    # Init model with tuned params
    model = FinBertSentiment(
        neutral_cap=args.neutral_cap,
        neg_gate=args.neg_gate,
        use_mixed_precision=args.mixed_precision,
        temperature=args.temperature,
        prefer_positive=args.prefer_positive,
    )
    model.load()

    # Fetch YF-only articles (already cleaned/trimmed by fetcher)
    print(f"\nFetching Yahoo Finance news for {args.ticker} (max {args.max_items})\n")
    try:
        items = fetch_news_yf_only(args.ticker, max_items=args.max_items)
    except Exception as e:
        print(f"Failed to fetch YF news: {e}")
        items = []

    print(f"Returned: {len(items)} articles\n")

    # Score with FinBERT using the fetcher schema
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
