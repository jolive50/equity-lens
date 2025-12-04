# FreshStart/models/sentiment/deberta_model.py
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline as hf_pipeline,
)

# --- fetcher (YF-only per your latest design) ---
from data.fetchers.news_data import fetch_news_yf_only

from .base_sentiment import (
    BaseSentiment,
    normalize_probs,
    SentimentResult,
    probs_to_score,
    entropy,
)

# Public, available model (binary SST-2 head)
DEFAULT_DEBERTA = os.environ.get(
    "DEBERTA_MODEL",
    "mrm8488/deberta-v3-small-finetuned-sst2",
)

# Optional zero-shot DeBERTa for native 3-way (slower)
ZS_DEFAULT = os.environ.get(
    "DEBERTA_ZS_MODEL",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
)


def _binary_to_three_way(p_neg: float, p_pos: float) -> Dict[str, float]:
    """Map 2-class (neg,pos) into (neg,neu,pos) via uncertainty band."""
    p_neg = float(max(0.0, min(1.0, p_neg)))
    p_pos = float(max(0.0, min(1.0, p_pos)))
    # high uncertainty (p_neg ≈ p_pos) → more neutral
    p_neu = max(0.0, min(1.0, 1.0 - abs(p_pos - p_neg)))
    return normalize_probs({"negative": p_neg, "neutral": p_neu, "positive": p_pos})


class DeBertaSentiment(BaseSentiment):
    """
    DeBERTa sentiment wrapper. Supports:
      - 2-class heads (SST-2): mapped to 3-class via uncertainty band
      - 3-class heads: used directly
    """

    def __init__(
        self,
        model_name: str = DEFAULT_DEBERTA,
        *,
        headline_first: float = 0.65,
        min_body_words: int = 60,
        neutral_cap: Optional[float] = 0.8,          # tuned: neutral_cap = 0.8
        neg_gate: Optional[float] = None,
        max_length: int = 514,
        device: Optional[str] = None,
        use_amp: bool = True,
        temperature: float = 1.0,                    # tuned: temperature = 1.0
    ) -> None:
        super().__init__(
            provider="deberta",
            headline_first=headline_first,
            min_body_words=min_body_words,
            neutral_cap=neutral_cap,
            neg_gate=neg_gate,
        )
        self.model_name = model_name
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device
        self._use_amp = bool(use_amp and (self._device == "cuda"))

        self._tokenizer = None
        self._model = None
        self._num_labels = 3  # detect on load

        self.temperature: float = float(temperature) if temperature and temperature > 0 else 1.0
        self._memo: Dict[str, Dict[str, float]] = {}

    # ---------- lifecycle ----------
    def load(self) -> None:
        if self._tokenizer and self._model:
            return
        from transformers import AutoConfig

        try:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            except Exception:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            cfg = AutoConfig.from_pretrained(self.model_name)
            self._num_labels = int(getattr(cfg, "num_labels", 3) or 3)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load DeBERTa model '{self.model_name}'. "
                "Install: pip install 'transformers>=4.44' torch"
            ) from e
        self._model.to(self._device)
        self._model.eval()

    def _forward_logits(self, texts: Sequence[str]) -> torch.Tensor:
        assert self._tokenizer and self._model
        enc = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        if self._use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self._model(**enc)
        else:
            out = self._model(**enc)
        logits = out.logits
        if self.temperature != 1.0:
            logits = logits / self.temperature
        return logits

    def _predict_text(self, text: str) -> Dict[str, float]:
        t = (text or "").strip()
        if not t:
            return {"negative": 0.0, "neutral": 1.0, "positive": 0.0}
        if t in self._memo:
            return self._memo[t]
        self.load()
        logits = self._forward_logits([t])[0]
        probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        if self._num_labels == 2:
            p = _binary_to_three_way(probs[0], probs[1])
        else:
            p = normalize_probs({"negative": probs[0], "neutral": probs[1], "positive": probs[2]})
        self._memo[t] = p
        return p


# -------- Optional: zero-shot DeBERTa (native 3-way) --------
def zero_shot_scores(
    texts: List[str],
    *,
    model_id: str = ZS_DEFAULT,
    device: Optional[str] = None,
) -> List[Dict[str, float]]:
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    zsp = hf_pipeline("zero-shot-classification", model=model_id, device=device)
    labels = ["negative", "neutral", "positive"]
    out: List[Dict[str, float]] = []
    for t in texts:
        if not t.strip():
            out.append({"negative": 0.0, "neutral": 1.0, "positive": 0.0})
            continue
        r = zsp(t, candidate_labels=labels, multi_label=False)
        p = {lab: float(score) for lab, score in zip(r["labels"], r["scores"])}
        out.append(
            normalize_probs(
                {
                    "negative": p.get("negative", 0.0),
                    "neutral": p.get("neutral", 0.0),
                    "positive": p.get("positive", 0.0),
                }
            )
        )
    return out


# ---------------- CLI (YF-only) ----------------
def _cli() -> None:
    """
    Run DeBERTa sentiment on freshly fetched Yahoo Finance news (no saved JSON).

    Examples:
      python -m FreshStart.models.sentiment.deberta_model AAPL
      python -m FreshStart.models.sentiment.deberta_model AAPL --max-items 15 --zs
    """
    import argparse
    from textwrap import shorten

    ap = argparse.ArgumentParser()
    ap.add_argument("ticker", help="Stock ticker, e.g., AAPL, MSFT, NVDA")
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--no-amp", action="store_true", help="Disable AMP")
    ap.add_argument("--neutral-cap", type=float, default=0.8)
    ap.add_argument("--neg-gate", type=float, default=None)
    ap.add_argument("--max-items", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--zs", action="store_true", help="Use zero-shot DeBERTa (native 3-way, slower)")
    args = ap.parse_args()

    # Fetch YF news (new signature: no keep_meta)
    print(f"\nFetching Yahoo Finance news for {args.ticker} (max {args.max_items})\n")
    try:
        items = fetch_news_yf_only(args.ticker, max_items=args.max_items)
    except Exception as e:
        print(f"Failed to fetch YF news: {e}")
        items = []
    print(f"Returned: {len(items)} articles\n")

    if args.zs:
        titles = [(it.get("title") or "").strip() for it in items]
        zs_probs = zero_shot_scores(titles)
        # Build results manually to keep formatting consistent
        results: List[SentimentResult] = []
        from .base_sentiment import cap_neutral, negative_gate

        for it, p in zip(items, zs_probs):
            mixed = normalize_probs(p)
            if args.neutral_cap is not None:
                mixed = cap_neutral(mixed, args.neutral_cap)
            if args.neg_gate is not None:
                mixed = negative_gate(mixed, threshold=args.neg_gate)

            conf = 1.0  # could swap to entropy-based confidence if you want later
            results.append(
                SentimentResult(
                    label=BaseSentiment._probs_to_label(mixed),
                    probs=mixed,
                    score=probs_to_score(mixed),
                    confidence=conf,
                    provider="deberta_zs",
                    text_id=it.get("id"),
                    title_used=bool(it.get("title")),
                    body_used=bool(it.get("body")),
                    tokens=None,
                    entropy=entropy(
                        [mixed["negative"], mixed["neutral"], mixed["positive"]]
                    ),
                    meta={
                        "source": it.get("source"),
                        "url": it.get("url"),
                        "provider_article": it.get("provider"),
                        "ticker": it.get("ticker"),
                        "time_published": it.get("time_published"),
                    },
                )
            )
    else:
        model = DeBertaSentiment(
            neutral_cap=args.neutral_cap,
            neg_gate=args.neg_gate,
            device=args.device,
            use_amp=not args.no_amp,
            temperature=args.temperature,
        )
        model.load()
        results = model.predict_batch_from_fetcher(items)

    # Compact summary
    for i, r in enumerate(results, 1):
        src = (r.meta.get("source") or "yahoo").strip().lower()
        src = src.replace("finance.yahoo.com", "yahoo").replace("yahoo/unknown", "yahoo")
        lab = r.label.name.capitalize()
        tpub = r.meta.get("time_published") or "-"
        title = shorten((items[i - 1].get("title") or "").strip(), width=90, placeholder=" [...]")
        print(f"YF {i:02d}. [{lab:7}] [{src:8}] {tpub}  {title}")

    # Full JSON
    out = [r.to_dict() for r in results]
    print("\n--- JSON OUTPUT ---")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
