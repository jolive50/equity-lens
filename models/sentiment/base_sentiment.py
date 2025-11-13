# FreshStart/models/sentiment/base_sentiment.py
from __future__ import annotations

import abc
import json
import logging
import math
import re
import time
from dataclasses import dataclass, asdict
from enum import IntEnum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# Get logger for sentiment models
logger = logging.getLogger("freshstart.sentiment")

# =========================
# Types & core structures
# =========================

class SentimentLabel5(IntEnum):
    STRONGLY_NEGATIVE = -2   # "Bearish"
    NEGATIVE          = -1   # "Slightly Bearish"
    NEUTRAL           =  0   # "Neutral"
    POSITIVE          =  1   # "Slightly Bullish"
    STRONGLY_POSITIVE =  2   # "Bullish"

LabelLike = Union[int, str, SentimentLabel5]

# 5-class canonical strings (lowercase for JSON)
_LABEL5_TO_STR = {
    SentimentLabel5.STRONGLY_NEGATIVE: "bearish",
    SentimentLabel5.NEGATIVE:          "slightly_bearish",
    SentimentLabel5.NEUTRAL:           "neutral",
    SentimentLabel5.POSITIVE:          "slightly_bullish",
    SentimentLabel5.STRONGLY_POSITIVE: "bullish",
}

def label5_to_str(lbl: SentimentLabel5) -> str:
    return _LABEL5_TO_STR[SentimentLabel5(lbl)]

def str_to_label5(s: str) -> SentimentLabel5:
    s = (s or "").strip().lower()
    return {
        "strongly_negative": SentimentLabel5.STRONGLY_NEGATIVE,
        "very_negative":     SentimentLabel5.STRONGLY_NEGATIVE,
        "bearish":           SentimentLabel5.STRONGLY_NEGATIVE,

        "negative":          SentimentLabel5.NEGATIVE,
        "slightly_negative": SentimentLabel5.NEGATIVE,
        "mildly_negative":   SentimentLabel5.NEGATIVE,
        "slightly_bearish":  SentimentLabel5.NEGATIVE,

        "neutral":           SentimentLabel5.NEUTRAL,

        "positive":          SentimentLabel5.POSITIVE,
        "slightly_positive": SentimentLabel5.POSITIVE,
        "mildly_positive":   SentimentLabel5.POSITIVE,
        "slightly_bullish":  SentimentLabel5.POSITIVE,

        "strongly_positive": SentimentLabel5.STRONGLY_POSITIVE,
        "very_positive":     SentimentLabel5.STRONGLY_POSITIVE,
        "bullish":           SentimentLabel5.STRONGLY_POSITIVE,
    }.get(s, SentimentLabel5.NEUTRAL)

@dataclass
class SentimentResult:
    """
    Unified result from any sentiment backend.

    Notes:
      - `probs` remain 3-class ("negative","neutral","positive") for math/ensembles.
      - `label` is now 5-class (see SentimentLabel5).
    """
    label: SentimentLabel5
    probs: Dict[str, float]           # keys: "negative", "neutral", "positive"
    score: float                      # convenient scalar, e.g., p_pos - p_neg in [-1,1]
    confidence: float                 # 0..1 (1 - normalized entropy over 3 classes)
    provider: str                     # "finbert", "deberta", etc.
    text_id: Optional[str] = None
    title_used: Optional[bool] = None
    body_used: Optional[bool] = None
    tokens: Optional[int] = None
    entropy: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # emit both numeric and human-readable label
        d["label"] = int(self.label)
        d["label_text"] = label5_to_str(self.label)
        return d

# =========================
# Normalization utilities
# =========================

def softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps) or 1.0
    return [e / s for e in exps]

def entropy(p: Sequence[float]) -> float:
    eps = 1e-12
    return -sum(max(pi, eps) * math.log(max(pi, eps)) for pi in p)

def normalized_confidence(p: Sequence[float]) -> float:
    """1 - (H / H_max) over 3 classes."""
    H = entropy(p)
    Hmax = math.log(3.0)
    return max(0.0, min(1.0, 1.0 - (H / Hmax)))

def normalize_probs(
    p_like: Union[Sequence[float], Mapping[str, float]],
    keys: Tuple[str, str, str] = ("negative", "neutral", "positive"),
) -> Dict[str, float]:
    if isinstance(p_like, Mapping):
        s = sum(float(p_like.get(k, 0.0)) for k in keys)
        s = s if s > 0 else 1.0
        return {k: max(0.0, float(p_like.get(k, 0.0)) / s) for k in keys}
    # assume sequence in NEG, NEU, POS order
    arr = [float(x) for x in p_like]
    if len(arr) != 3:
        raise ValueError("Expected 3-class probabilities or logits.")
    probs = softmax(arr)
    return {k: probs[i] for i, k in enumerate(keys)}

def probs_to_score(p: Mapping[str, float]) -> float:
    """Convenient scalar in [-1,1]: p_pos - p_neg."""
    return float(p.get("positive", 0.0) - p.get("negative", 0.0))

def cap_neutral(p: Dict[str, float], neutral_cap: Optional[float]) -> Dict[str, float]:
    """Clamp neutral prob to <= cap, renormalize remainder proportionally."""
    if neutral_cap is None:
        return p
    ncap = float(neutral_cap)
    if p["neutral"] <= ncap:
        return p
    overflow = p["neutral"] - ncap
    pos_neg = p["positive"] + p["negative"]
    if pos_neg <= 0:
        return {"negative": 0.5, "neutral": ncap, "positive": 0.5}
    scale = overflow / pos_neg
    q = {
        "negative": max(0.0, p["negative"] + p["negative"] * scale),
        "neutral":  ncap,
        "positive": max(0.0, p["positive"] + p["positive"] * scale),
    }
    return normalize_probs(q)

def negative_gate(p: Dict[str, float], threshold: float = 0.80) -> Dict[str, float]:
    """
    If the model is VERY confident it's negative, collapse neutral mass to negative
    (useful for downside-risk gating).
    """
    if p["negative"] >= threshold and p["negative"] >= (p["positive"] + 0.1):
        mass = p["neutral"]
        q = {"negative": p["negative"] + mass, "neutral": 0.0, "positive": p["positive"]}
        return normalize_probs(q)
    return p

# =========================
# Title/Body mixing
# =========================
def default_mix(
    title_probs: Dict[str, float],
    body_probs: Optional[Dict[str, float]],
    *,
    headline_first: float = 0.65,
    min_body_words: int = 60,
    body_word_count: Optional[int] = None,
) -> Dict[str, float]:
    """
    Headline-first mixing:
      - If no body or too short, return title_probs.
      - Else convex combo: w * title + (1-w) * body.
    """
    if not body_probs or (body_word_count is not None and body_word_count < min_body_words):
        return normalize_probs(title_probs)
    w = max(0.0, min(1.0, float(headline_first)))
    q = {
        "negative": w * title_probs["negative"] + (1 - w) * body_probs["negative"],
        "neutral":  w * title_probs["neutral"]  + (1 - w) * body_probs["neutral"],
        "positive": w * title_probs["positive"] + (1 - w) * body_probs["positive"],
    }
    return normalize_probs(q)

# =========================
# Base interface (5-class)
# =========================
class BaseSentiment(abc.ABC):
    """
    Subclass this with your concrete backends (FinBERT, DeBERTa, etc.).
    Implement `_predict_text` to return 3-class probs for a single string.
    5-class label is derived from score via thresholds.
    """

    def __init__(
        self,
        *,
        provider: str,
        headline_first: float = 0.65,
        min_body_words: int = 60,
        neutral_cap: Optional[float] = None,
        neg_gate: Optional[float] = None,
        # 5-class mapping thresholds on score = p_pos - p_neg
        mild_thresh: float = 0.15,
        strong_thresh: float = 0.50,
    ) -> None:
        self.provider = provider
        self.headline_first = headline_first
        self.min_body_words = min_body_words
        self.neutral_cap = neutral_cap
        self.neg_gate = neg_gate
        self.mild_thresh = float(mild_thresh)
        self.strong_thresh = float(strong_thresh)

    # ---- lifecycle
    def load(self) -> None:
        """Override if the model requires lazy loading."""
        return

    def close(self) -> None:
        """Override to free model resources."""
        return

    # ---- abstract core
    @abc.abstractmethod
    def _predict_text(self, text: str) -> Dict[str, float]:
        """
        Return normalized probs dict for a single text:
        {"negative": p0, "neutral": p1, "positive": p2}
        """
        raise NotImplementedError

    # ---- public API
    def predict_one(
        self,
        *,
        title: Optional[str] = None,
        body: Optional[str] = None,
        text_id: Optional[str] = None,
        body_word_count: Optional[int] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> SentimentResult:
        start_time = time.time()
        title = (title or "").strip()
        body = (body or "").strip()

        # Log prediction start
        logger.debug(f"🔮 [{self.provider}] Predicting sentiment for text_id={text_id}")
        logger.debug(f"   Title: {title[:80]}{'...' if len(title) > 80 else ''}")
        if body:
            logger.debug(f"   Body: {len(body)} chars, {body_word_count or 0} words")

        # Predict title
        title_start = time.time()
        title_probs = self._predict_text(title) if title else {"negative": 0.0, "neutral": 1.0, "positive": 0.0}
        title_time = time.time() - title_start
        logger.debug(f"   Title prediction: {title_probs} ({title_time:.3f}s)")

        # Predict body
        body_probs = None
        if body:
            body_start = time.time()
            body_probs = self._predict_text(body)
            body_time = time.time() - body_start
            logger.debug(f"   Body prediction: {body_probs} ({body_time:.3f}s)")

        # Mix title and body
        mixed = default_mix(
            title_probs,
            body_probs,
            headline_first=self.headline_first,
            min_body_words=self.min_body_words,
            body_word_count=body_word_count,
        )
        logger.debug(f"   Mixed (title+body): {mixed}")

        # post-process (neutral cap, negative gate, normalization)
        if self.neutral_cap is not None:
            before_cap = mixed.copy()
            mixed = cap_neutral(mixed, self.neutral_cap)
            if mixed != before_cap:
                logger.debug(f"   After neutral_cap={self.neutral_cap}: {mixed}")
        if self.neg_gate is not None:
            before_gate = mixed.copy()
            mixed = negative_gate(mixed, threshold=self.neg_gate)
            if mixed != before_gate:
                logger.debug(f"   After neg_gate={self.neg_gate}: {mixed}")
        mixed = normalize_probs(mixed)

        conf = normalized_confidence([mixed["negative"], mixed["neutral"], mixed["positive"]])
        score = probs_to_score(mixed)
        label5 = self._score_to_label5(score)
        ent = entropy([mixed["negative"], mixed["neutral"], mixed["positive"]])

        elapsed = time.time() - start_time
        logger.info(
            f"✅ [{self.provider}] Sentiment: {label5_to_str(label5)} | "
            f"Score: {score:+.3f} | Conf: {conf:.3f} | "
            f"Probs: neg={mixed['negative']:.3f} neu={mixed['neutral']:.3f} pos={mixed['positive']:.3f} | "
            f"Time: {elapsed:.3f}s"
        )

        result = SentimentResult(
            label=label5,
            probs=mixed,
            score=score,
            confidence=conf,
            provider=self.provider,
            text_id=text_id,
            title_used=bool(title),
            body_used=bool(body),
            tokens=None,
            entropy=ent,
            meta=meta or {},
        )
        return result

    def predict_batch(
        self,
        items: Iterable[Mapping[str, Any]],
        *,
        title_key: str = "title",
        body_key: str = "body",
        id_key: Optional[str] = "id",
        body_wc_key: Optional[str] = "word_count",
    ) -> List[SentimentResult]:
        out: List[SentimentResult] = []
        for it in items:
            r = self.predict_one(
                title=(it.get(title_key) or ""),
                body=(it.get(body_key) or ""),
                text_id=(it.get(id_key) if id_key else None),
                body_word_count=(it.get(body_wc_key) if body_wc_key else None),
                meta={
                    "source": it.get("source"),
                    "url": it.get("url"),
                    "provider_article": it.get("provider"),
                    "ticker": it.get("ticker"),
                    "time_published": it.get("time_published"),
                },
            )
            out.append(r)
        return out

    # Convenience: exact schema from FreshStart.data.fetchers.news_data
    def predict_batch_from_fetcher(self, news_items: Iterable[Mapping[str, Any]]) -> List[SentimentResult]:
        """
        Accepts items as returned by fetch_news_yf_only(...):
        {id, ticker, time_published, source, title, url, provider, body, word_count}
        """
        return self.predict_batch(
            news_items,
            title_key="title",
            body_key="body",
            id_key="id",
            body_wc_key="word_count",
        )

    # ---- helpers
    def _score_to_label5(self, score: float) -> SentimentLabel5:
        """
        Map score (p_pos - p_neg) to 5-class:
          <= -strong  -> STRONGLY_NEGATIVE ("bearish")
          <  -mild    -> NEGATIVE ("slightly_bearish")
          between     -> NEUTRAL
          >=  strong  -> STRONGLY_POSITIVE ("bullish")
          >   mild    -> POSITIVE ("slightly_bullish")
        """
        s = float(score)
        if s <= -self.strong_thresh:
            return SentimentLabel5.STRONGLY_NEGATIVE
        if s < -self.mild_thresh:
            return SentimentLabel5.NEGATIVE
        if s >= self.strong_thresh:
            return SentimentLabel5.STRONGLY_POSITIVE
        if s > self.mild_thresh:
            return SentimentLabel5.POSITIVE
        return SentimentLabel5.NEUTRAL

# =========================
# Minimal reference impl
# =========================
class DummySentiment(BaseSentiment):
    """
    Heuristic 3-prob backend that outputs 5-class label via score thresholds.
    Good for smoke tests and CI.
    """
    _NEG_RX = re.compile(r"\b(down|plunge|fall|drop|lawsuit|ban|shortfall|miss|fraud|recall|bearish)\b", re.I)
    _POS_RX = re.compile(r"\b(up|surge|rise|beat|record|profit|win|approval|bullish|upgrade)\b", re.I)

    def __init__(self) -> None:
        super().__init__(provider="dummy", headline_first=0.65, min_body_words=60, neutral_cap=None, neg_gate=None)

    def _predict_text(self, text: str) -> Dict[str, float]:
        if not text or not text.strip():
            return {"negative": 0.0, "neutral": 1.0, "positive": 0.0}
        neg_hits = len(self._NEG_RX.findall(text))
        pos_hits = len(self._POS_RX.findall(text))
        if neg_hits == 0 and pos_hits == 0:
            return {"negative": 0.2, "neutral": 0.6, "positive": 0.2}
        if neg_hits > pos_hits:
            return normalize_probs([0.70, 0.20, 0.10])
        if pos_hits > neg_hits:
            return normalize_probs([0.10, 0.20, 0.70])
        return {"negative": 0.33, "neutral": 0.34, "positive": 0.33}

# =========================
# CLI for quick tests (Windows-friendly)
# =========================
def _read_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_json_file(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _cli() -> None:
    """
    Examples (Windows-friendly):

      # file I/O (recommended)
      python -m FreshStart.models.sentiment.base_sentiment --provider dummy --input news.json --output preds.json

      # still supports stdin (PowerShell)
      Get-Content -Raw news.json | python -m FreshStart.models.sentiment.base_sentiment --provider dummy --stdin-json
    """
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="dummy", help="dummy (reference impl)")
    ap.add_argument("--stdin-json", action="store_true", help="Read JSON list from stdin and emit results.")
    ap.add_argument("--input", type=str, help="Path to input JSON file (list of items).")
    ap.add_argument("--output", type=str, help="Path to write results JSON.")
    ap.add_argument("--neutral-cap", type=float, default=None)
    ap.add_argument("--neg-gate", type=float, default=None)
    ap.add_argument("--mild-thresh", type=float, default=0.15)
    ap.add_argument("--strong-thresh", type=float, default=0.50)
    args = ap.parse_args()

    if args.provider != "dummy":
        print("Only 'dummy' provider is wired here. Use your model subclass module (e.g., finbert_model.py).", file=sys.stderr)

    model: BaseSentiment = DummySentiment()
    if args.neutral_cap is not None:
        model.neutral_cap = args.neutral_cap
    if args.neg_gate is not None:
        model.neg_gate = args.neg_gate
    model.mild_thresh = float(args.mild_thresh)
    model.strong_thresh = float(args.strong_thresh)

    model.load()

    if args.input:
        data = _read_json_file(args.input)
    elif args.stdin_json:
        data = json.load(sys.stdin)
    else:
        # quick demo
        data = [
            {"id": "1", "title": "Apple stock surges on record iPhone upgrades", "body": "Investors cheered...", "word_count": 120, "source":"demo","url":"https://x","ticker":"AAPL","time_published":"2025-01-01T00:00:00Z","provider":"demo"},
            {"id": "2", "title": "Company faces antitrust lawsuit and shares plunge", "body": "Regulators filed...", "word_count": 90,  "source":"demo","url":"https://y","ticker":"ACME","time_published":"2025-01-01T00:00:00Z","provider":"demo"},
            {"id": "3", "title": "Mixed outlook for earnings next quarter", "body": "", "word_count": 0, "source":"demo","url":"https://z","ticker":"ACME","time_published":"2025-01-01T00:00:00Z","provider":"demo"},
        ]

    results = model.predict_batch_from_fetcher(data)
    out = [r.to_dict() for r in results]

    if args.output:
        _write_json_file(args.output, out)
    else:
        print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    _cli()
