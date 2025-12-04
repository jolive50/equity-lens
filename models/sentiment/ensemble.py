# FreshStart/models/sentiment/ensemble.py
from __future__ import annotations

import logging
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math

# Model wrappers (already in your tree)
from .base_sentiment import (
    BaseSentiment,
    SentimentResult,
    normalize_probs,
    normalized_confidence,
    entropy as _entropy,
    label5_to_str,
)

# Fetchers
from data.fetchers.news_data import fetch_news_yf_only
from .alpha_vantage_sentiment import fetch_alpha_vantage_news

logger = logging.getLogger("freshstart.sentiment.ensemble")


def _soft_vote(model_probs: Dict[str, Dict[str, float]], weights: Dict[str, float]) -> Dict[str, float]:
    wsum = 0.0
    acc = {"negative": 0.0, "neutral": 0.0, "positive": 0.0}
    for name, p in model_probs.items():
        w = float(weights.get(name, 0.0))
        if w <= 0:
            continue
        acc["negative"] += w * p.get("negative", 0.0)
        acc["neutral"]  += w * p.get("neutral",  0.0)
        acc["positive"] += w * p.get("positive", 0.0)
        wsum += w
    if wsum <= 0:
        return {"negative": 0.0, "neutral": 1.0, "positive": 0.0}
    acc = {k: v / wsum for k, v in acc.items()}
    return normalize_probs(acc)


def _score_from_probs(p: Mapping[str, float]) -> float:
    return float(p.get("positive", 0.0) - p.get("negative", 0.0))


def _apply_neutral_cap(p: Dict[str, float], cap: Optional[float]) -> Dict[str, float]:
    if cap is None or p["neutral"] <= cap:
        return p
    overflow = p["neutral"] - cap
    pos_neg = p["positive"] + p["negative"]
    if pos_neg <= 0:
        return {"negative": 0.5, "neutral": cap, "positive": 0.5}
    scale = overflow / pos_neg
    q = {
        "negative": p["negative"] + p["negative"] * scale,
        "neutral": cap,
        "positive": p["positive"] + p["positive"] * scale,
    }
    return normalize_probs(q)


def _apply_neg_gate(p: Dict[str, float], thr: Optional[float]) -> Dict[str, float]:
    if thr is None:
        return p
    if p["negative"] >= thr and p["negative"] >= (p["positive"] + 0.1):
        q = {"negative": p["negative"] + p["neutral"], "neutral": 0.0, "positive": p["positive"]}
        return normalize_probs(q)
    return p


def _conf_from_probs(p: Mapping[str, float]) -> float:
    return normalized_confidence([p["negative"], p["neutral"], p["positive"]])


def _map_five_band(score: float, t_strong: float, t_mild: float) -> str:
    # Strong Negative, Mild Negative, Neutral, Mild Positive, Strong Positive
    if score <= -t_strong:
        return "strong_negative"
    if score < -t_mild:
        return "mild_negative"
    if -t_mild <= score <= t_mild:
        return "neutral"
    if score < t_strong:
        return "mild_positive"
    return "strong_positive"


def _av_probs_from_label(label: str, score: Optional[float]) -> Dict[str, float]:
    lab = (label or "").strip().lower()
    base = 0.80  # anchor mass on AV label
    rest = 0.20
    if lab.startswith("pos"):
        return normalize_probs({"negative": 0.0, "neutral": rest, "positive": base})
    if lab.startswith("neg"):
        return normalize_probs({"negative": base, "neutral": rest, "positive": 0.0})
    # neutral or unknown
    return normalize_probs({"negative": 0.10, "neutral": 0.80, "positive": 0.10})


class SentimentEnsemble:
    """
    Weighted soft-vote ensemble over multiple BaseSentiment models.
    AV items can be anchored to Alpha Vantage labels or blended with model votes.
    """

    def __init__(
        self,
        finbert: Optional[BaseSentiment] = None,
        deberta: Optional[BaseSentiment] = None,
        roberta: Optional[BaseSentiment] = None,
        *,
        weights: Optional[Dict[str, float]] = None,     # model weights for soft vote
        av_trust_mode: str = "anchor",                  # "anchor" | "blend" | "ignore"
        av_weight: float = 0.50,                        # only when av_trust_mode="blend"
        neutral_cap: Optional[float] = None,
        neg_gate: Optional[float] = None,
        entropy_weighting: bool = True,
        five_band: bool = False,
        five_band_thresholds: Tuple[float, float] = (0.20, 0.05),  # (strong, mild)
    ) -> None:
        self.models: Dict[str, BaseSentiment] = {}
        if finbert: self.models["finbert"] = finbert
        if deberta: self.models["deberta"] = deberta
        if roberta: self.models["roberta"] = roberta

        self.weights = weights or {"finbert": 0.45, "deberta": 0.35, "roberta": 0.20}
        self.av_trust_mode = av_trust_mode
        self.av_weight = max(0.0, min(1.0, av_weight))
        self.neutral_cap = neutral_cap
        self.neg_gate = neg_gate
        self.entropy_weighting = bool(entropy_weighting)
        self.five_band = bool(five_band)
        self.five_band_thresholds = five_band_thresholds
        logger.info(
            "SentimentEnsemble initialized: models=%s weights=%s av_mode=%s av_weight=%.2f",
            list(self.models.keys()),
            self.weights,
            self.av_trust_mode,
            self.av_weight,
        )

    def load(self) -> None:
        for m in self.models.values():
            m.load()

    # ---- public API
    def predict_one(self, item: Mapping[str, Any]) -> SentimentResult:
        start_time = time.time()
        provider = (item.get("provider") or "").lower()
        text_id = item.get("id", "unknown")

        logger.info(
            "Ensemble predict_one: provider=%s text_id=%s models=%s",
            provider,
            text_id,
            list(self.models.keys()),
        )
        if len(self.models) < 2:
            logger.warning("Ensemble has <2 models; results may mirror the only available model(s)")

        logger.info(f"🎯 [Ensemble] Starting prediction for text_id={text_id}, provider={provider}")
        logger.debug(f"   Models: {list(self.models.keys())}, Weights: {self.weights}")

        # AV path
        if provider == "alpha_vantage":
            av_lab = ((item.get("av_meta") or {}).get("overall_sentiment_label") or "").lower()
            av_score = (item.get("av_meta") or {}).get("overall_sentiment_score")
            av_probs = _av_probs_from_label(av_lab, av_score)
            logger.info(f"   📊 Alpha Vantage label: {av_lab}, score: {av_score}, probs: {av_probs}")

            if self.av_trust_mode == "anchor" or not self.models:
                logger.info(f"   🔒 Using AV anchor mode (trust_mode={self.av_trust_mode})")
                mixed = av_probs
            elif self.av_trust_mode == "blend":
                logger.info(f"   🔀 Blending AV with models (av_weight={self.av_weight})")
                model_probs, model_conf = self._models_vote(item)
                w_av = self.av_weight
                w_md = 1.0 - w_av
                blended = {
                    "negative": w_av * av_probs["negative"] + w_md * model_probs["negative"],
                    "neutral":  w_av * av_probs["neutral"]  + w_md * model_probs["neutral"],
                    "positive": w_av * av_probs["positive"] + w_md * model_probs["positive"],
                }
                mixed = normalize_probs(blended)
                logger.info(f"   ✓ Blended result: {mixed}")
            else:  # "ignore"
                logger.info(f"   🚫 Ignoring AV, using only models")
                mixed, model_conf = self._models_vote(item)
        else:
            # YF (or other) path → model vote
            logger.info(f"   🤖 Using model ensemble voting")
            mixed, model_conf = self._models_vote(item)

        # Calibration
        logger.debug(f"   📐 Pre-calibration probs: {mixed}")
        if self.neutral_cap is not None:
            before = mixed.copy()
            mixed = _apply_neutral_cap(mixed, self.neutral_cap)
            if mixed != before:
                logger.debug(f"   ✓ After neutral_cap={self.neutral_cap}: {mixed}")
        if self.neg_gate is not None:
            before = mixed.copy()
            mixed = _apply_neg_gate(mixed, self.neg_gate)
            if mixed != before:
                logger.debug(f"   ✓ After neg_gate={self.neg_gate}: {mixed}")
        mixed = normalize_probs(mixed)

        conf = _conf_from_probs(mixed)
        score = _score_from_probs(mixed)
        ent = _entropy([mixed["negative"], mixed["neutral"], mixed["positive"]])
        label = self._probs_to_label(mixed)

        elapsed = time.time() - start_time
        logger.info(
            f"✅ [Ensemble] Result: {label5_to_str(label)} | "
            f"Score: {score:+.3f} | Conf: {conf:.3f} | "
            f"Probs: neg={mixed['negative']:.3f} neu={mixed['neutral']:.3f} pos={mixed['positive']:.3f} | "
            f"Time: {elapsed:.3f}s"
        )

        # Compose result using BaseSentiment-compatible fields
        meta = {
            "source": item.get("source"),
            "url": item.get("url"),
            "provider_article": item.get("provider"),
            "ticker": item.get("ticker"),
            "time_published": item.get("time_published"),
            "ensemble": {
                "strategy": "soft",
                "weights": self.weights,
                "used_models": list(self.models.keys()),
                "av_trust_mode": self.av_trust_mode,
            },
        }

        res = SentimentResult(
            label=label,
            probs=mixed,
            score=score,
            confidence=conf,
            provider="ensemble",
            text_id=text_id,
            title_used=bool(item.get("title")),
            body_used=bool(item.get("body")),
            tokens=None,
            entropy=ent,
            meta=meta,
        )

        if self.five_band:
            # add five-band tag inside meta, keep 3-class label for compatibility
            strong, mild = self.five_band_thresholds
            fb = _map_five_band(score, strong, mild)
            res.meta["ensemble"]["five_band"] = fb
            logger.debug(f"   🎚️  Five-band classification: {fb}")

        return res

    def predict_batch(self, items: Iterable[Mapping[str, Any]]) -> List[SentimentResult]:
        return [self.predict_one(it) for it in items]

    # ---- internals
    def _models_vote(self, item: Mapping[str, Any]) -> Tuple[Dict[str, float], float]:
        title = (item.get("title") or "").strip()
        body = (item.get("body") or "").strip()
        wc = int(item.get("word_count") or 0)

        logger.info("      Models available for vote: %s", list(self.models.keys()))
        logger.debug(f"   🗳️  Starting ensemble vote with {len(self.models)} models")
        logger.debug(f"      Title length: {len(title)}, Body: {len(body)} chars, {wc} words")

        probs_by_model: Dict[str, Dict[str, float]] = {}
        conf_by_model: Dict[str, float] = {}

        for name, model in self.models.items():
            model_start = time.time()
            r = model.predict_one(title=title, body=body, text_id=item.get("id"), body_word_count=wc, meta=None)
            model_time = time.time() - model_start

            probs_by_model[name] = r.probs
            conf_by_model[name] = r.confidence

            logger.info(
                f"      📊 [{name}] Probs: neg={r.probs['negative']:.3f} neu={r.probs['neutral']:.3f} pos={r.probs['positive']:.3f} | "
                f"Conf: {r.confidence:.3f} | Time: {model_time:.3f}s"
            )

        # entropy-aware weighting (optional)
        weights = dict(self.weights)
        logger.debug(f"      ⚖️  Static weights: {weights}")

        if self.entropy_weighting:
            logger.debug(f"      📈 Applying entropy-based confidence weighting")
            # multiply each static weight by confidence, then renormalize
            adj = {n: max(0.0, weights.get(n, 0.0)) * max(0.0, min(1.0, conf_by_model.get(n, 0.0)))
                   for n in probs_by_model.keys()}
            s = sum(adj.values()) or 0.0
            if s > 0:
                weights = {n: v / s for n, v in adj.items()}
                logger.info(f"      ⚖️  Adjusted weights (conf-weighted): {weights}")
            else:
                # fallback to static weights of models present
                present = {n: weights.get(n, 0.0) for n in probs_by_model.keys()}
                s2 = sum(present.values()) or 1.0
                weights = {n: v / s2 for n, v in present.items()}
                logger.warning(f"      ⚠️  Fallback to static weights: {weights}")

        mixed = _soft_vote(probs_by_model, weights)
        logger.info(f"      ✅ Ensemble vote result: {mixed}")

        # average confidence just for diagnostics
        avg_conf = sum(conf_by_model.values()) / max(1, len(conf_by_model))
        logger.debug(f"      📊 Average model confidence: {avg_conf:.3f}")

        return mixed, avg_conf

    @staticmethod
    def _probs_to_label(p: Mapping[str, float]) -> int:
        # map to {-1,0,1} by argmax over NEG/NEU/POS
        triples = [("negative", -1), ("neutral", 0), ("positive", 1)]
        best = max(triples, key=lambda kv: p.get(kv[0], 0.0))
        return best[1]

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for compatibility with workflow."""
        model_names = list(self.models.keys())
        return {
            "name": "SentimentEnsemble",
            "version": "1.0",
            "type": "ensemble",
            "strategy": "weighted_soft_vote",
            "models": model_names,
            "weights": self.weights,
            "av_trust_mode": self.av_trust_mode
        }

    def analyze(self, text: str):
        """Legacy API compatibility - analyze a single text string.

        Args:
            text: Text to analyze

        Returns:
            LegacySentimentResult-like object with label, confidence, probabilities
        """
        from dataclasses import dataclass

        # Create a mock news item for the ensemble
        item = {
            "id": "legacy_text",
            "title": text,
            "body": "",
            "word_count": len(text.split()),
            "provider": "legacy",
            "source": "unknown"
        }

        # Use the ensemble's predict_one method
        result = self.predict_one(item)

        # Convert to legacy 3-class format
        label_map = {
            -2: "negative",
            -1: "negative",
            0: "neutral",
            1: "positive",
            2: "positive"
        }
        label_3class = label_map.get(int(result.label), "neutral")

        # Return object with legacy API attributes
        @dataclass
        class LegacyResult:
            label: str
            confidence: float
            probabilities: Dict[str, float]

        return LegacyResult(
            label=label_3class,
            confidence=result.confidence,
            probabilities=result.probs
        )


# ---------------- CLI (optional helper to demo mix of AV + YF) ----------------
def _cli() -> None:
    import argparse, json
    from textwrap import shorten

    # Lazy imports to avoid TF start-up if not needed here
    from .finbert_model import FinBertSentiment
    from .deberta_model import DeBertaSentiment
    from .roberta_model import RobertaSentiment

    ap = argparse.ArgumentParser(description="Ensemble soft-vote over FinBERT/DeBERTa/RoBERTa + AV handling.")
    ap.add_argument("ticker", help="Ticker symbol, e.g., AAPL")
    ap.add_argument("--max-items-yf", type=int, default=10)
    ap.add_argument("--max-items-av", type=int, default=10)
    ap.add_argument("--av-trust", choices=["anchor", "blend", "ignore"], default="blend")
    ap.add_argument("--av-weight", type=float, default=0.5)
    ap.add_argument("--neutral-cap", type=float, default=None)
    ap.add_argument("--neg-gate", type=float, default=None)
    ap.add_argument("--no-entropy-weight", action="store_true")
    ap.add_argument("--five-band", action="store_true")
    args = ap.parse_args()

    print(f"\nFetching YF({args.max_items_yf}) + AV({args.max_items_av}) for {args.ticker}\n")
    yf_items = []
    av_items = []
    try:
        yf_items = fetch_news_yf_only(args.ticker, max_items=args.max_items_yf)
    except Exception as e:
        print(f"YF fetch failed: {e}")
    try:
        # key is resolved internally by the AV module's CLI path; here we expect env vars already set
        from os import getenv
        key = getenv("ALPHA_VANTAGE_API_KEY") or getenv("ALPHAVANTAGE_KEY")
        if key:
            av_items = fetch_alpha_vantage_news(args.ticker, api_key=key, max_items=args.max_items_av)
        else:
            print("AV key missing; skipping AV fetch.")
    except Exception as e:
        print(f"AV fetch failed: {e}")

    items = av_items + yf_items

    finbert = FinBertSentiment()
    deberta = DeBertaSentiment()
    roberta = RobertaSentiment()

    ens = SentimentEnsemble(
        finbert=finbert,
        deberta=deberta,
        roberta=roberta,
        av_trust_mode=args.av_trust,
        av_weight=args.av_weight,
        neutral_cap=args.neutral_cap,
        neg_gate=args.neg_gate,
        entropy_weighting=not args.no_entropy_weight,
        five_band=args.five_band,
    )

    ens.load()
    results = ens.predict_batch(items)

    # Pretty print
    for i, (it, r) in enumerate(zip(items, results), start=1):
        tag = "AV" if (it.get("provider") == "alpha_vantage") else "YF"
        src = (it.get("source") or "").lower() or tag.lower()
        tpub = it.get("time_published") or "-"
        title = shorten((it.get("title") or "").strip(), width=92, placeholder=" [...]")
        lab = "Negative" if r.probs["negative"] == max(r.probs.values()) else ("Positive" if r.probs["positive"] == max(r.probs.values()) else "Neutral")
        print(f"{tag} {i:02d}. [{lab:8}] [{src:10}] {tpub}  {title}")

    print("\n--- JSON OUTPUT ---")
    print(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
