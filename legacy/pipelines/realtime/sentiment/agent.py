# pipelines/realtime/sentiment/agent.py
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

# If your repo already loads envs elsewhere you can remove this.
try:
    from dotenv import load_dotenv  # optional dependency
    load_dotenv()
except Exception:
    pass


# ---------- Public data models ----------

@dataclass
class SentimentAgentInput:
    """Inputs for the sentiment agent."""
    tickers: str                         # e.g., "AAPL,MSFT"
    topics: Optional[str] = None         # e.g., "earnings,technology"
    time_from: Optional[str] = None      # ISO or YYYYMMDDTHHMM
    time_to: Optional[str] = None
    limit: int = 50
    pages: int = 1
    save_csv_path: Optional[str] = None  # e.g., "alpha_sentiment.csv"
    neutral_cap: Optional[float] = None  # override pipeline NEUTRAL_CAP at runtime
    weights: Optional[Dict[str, float]] = None  # override WEIGHTS_BASE at runtime


@dataclass
class ArticleSentiment:
    time_published: Optional[str]
    source: Optional[str]
    url: str
    title: str
    pred_label: str
    pred_conf: float
    alpha_label: Optional[str]
    alpha_score: Optional[float]
    body_chars: int


@dataclass
class SentimentSummary:
    total: int
    positive: int
    neutral: int
    negative: int


@dataclass
class SentimentAgentOutput:
    inputs: Dict[str, Any]
    summary: SentimentSummary
    articles: List[ArticleSentiment]
    csv_path: Optional[str] = None


# ---------- Internals ----------

def _require_alpha_vantage_key() -> str:
    """
    Centralized env check so importing the pipeline can safely assume the key exists.
    If your project has pipelines/realtime/api_keys.py, prefer that helper.
    """
    try:
        from pipelines.realtime.api_keys import alpha_vantage_key  # your existing helper
        return alpha_vantage_key()
    except Exception:
        key = os.getenv("ALPHAVANTAGE_KEY")
        if not key:
            raise RuntimeError("ALPHAVANTAGE_KEY not set. Put it in your environment or .env.")
        return key


def _summarize(rows: List[Dict[str, Any]]) -> SentimentSummary:
    pos = sum(1 for r in rows if r.get("pred_label") == "positive")
    neu = sum(1 for r in rows if r.get("pred_label") == "neutral")
    neg = sum(1 for r in rows if r.get("pred_label") == "negative")
    return SentimentSummary(total=len(rows), positive=pos, neutral=neu, negative=neg)


# ---------- Public API ----------

def run_sentiment_agent(params: SentimentAgentInput) -> SentimentAgentOutput:
    """
    Execute the sentiment pipeline for the given inputs and return a structured result.
    This function is side-effect free except for optional CSV writing when save_csv_path is set.
    """
    # Ensure key exists *before* importing the pipeline (which may assert on import)
    _require_alpha_vantage_key()

    # Lazy import so callers can set env vars before calling this function.
    from . import sentiment_pipeline as pipe  # noqa: WPS433

    # Optional runtime overrides (affect module-level knobs inside the pipeline)
    if params.neutral_cap is not None:
        pipe.NEUTRAL_CAP = float(params.neutral_cap)
    if params.weights:
        w = dict(pipe.WEIGHTS_BASE)
        w.update({str(k): float(v) for k, v in params.weights.items()})
        s = sum(w.values())
        if s > 0:
            pipe.WEIGHTS_BASE = {k: v / s for k, v in w.items()}

    # Run the batch fetch + scoring
    rows = pipe.analyze_alpha_news_batch(
        tickers=params.tickers,
        topics=params.topics,
        time_from=params.time_from,
        time_to=params.time_to,
        limit=params.limit,
        pages=params.pages,
    )

    csv_path = None
    if params.save_csv_path and rows:
        pipe.save_results_csv(rows, params.save_csv_path)
        csv_path = params.save_csv_path

    # Shape the output
    articles = [ArticleSentiment(**{
        "time_published": r.get("time_published"),
        "source": r.get("source"),
        "url": r.get("url", ""),
        "title": r.get("title", ""),
        "pred_label": r.get("pred_label", "neutral"),
        "pred_conf": float(r.get("pred_conf", 0.0)),
        "alpha_label": r.get("alpha_label"),
        "alpha_score": r.get("alpha_score"),
        "body_chars": int(r.get("body_chars", 0)),
    }) for r in rows]

    summary = _summarize(rows)
    return SentimentAgentOutput(
        inputs=asdict(params),
        summary=summary,
        articles=articles,
        csv_path=csv_path,
    )


__all__ = [
    "SentimentAgentInput",
    "ArticleSentiment",
    "SentimentSummary",
    "SentimentAgentOutput",
    "run_sentiment_agent",
]
