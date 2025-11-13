# FreshStart/models/sentiment/alpha_vantage_sentiment.py
from __future__ import annotations

import json
import os
import hashlib
from typing import Dict, List, Optional, Tuple

import requests
from datetime import datetime, timezone
from textwrap import shorten

# Reuse title cleaner from YF fetcher if available
try:
    from FreshStart.data.fetchers.news_data import _clean_title  # type: ignore
except Exception:
    def _clean_title(s: str, *, max_len: int = 140) -> str:
        s = (s or "").replace("\u00a0", " ").strip()
        return (s[:max_len].rstrip() + "...") if len(s) > max_len else s

# Optional: dotenv support
try:
    from dotenv import load_dotenv  # type: ignore
    _HAS_DOTENV = True
except Exception:
    _HAS_DOTENV = False

# Optional: legacy helper
try:
    from legacy.pipelines.realtime.api_keys import get_api_key as _legacy_get_api_key  # type: ignore
except Exception:
    _legacy_get_api_key = None  # type: ignore

AV_BASE = "https://www.alphavantage.co/query"
AV_MAX = 10  # hard cap per your 50/50 design

# ----------------------------- helpers -----------------------------
def _sha1_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"|")
    return h.hexdigest()

def _to_iso8601(ts: Optional[str]) -> str:
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""

def _resolve_api_key(cli_key: Optional[str]) -> Tuple[Optional[str], str]:
    """
    Resolve Alpha Vantage key with precedence:
      1) CLI --api-key
      2) Env: ALPHA_VANTAGE_API_KEY or ALPHAVANTAGE_KEY
      3) legacy get_api_key('alpha_vantage')
      4) .env (load, then env again)
    Returns: (key_or_none, source_label)
    """
    if cli_key and cli_key.strip():
        return cli_key.strip(), "cli"

    env_key = os.getenv("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHAVANTAGE_KEY")
    if env_key and env_key.strip():
        return env_key.strip(), "env"

    if _legacy_get_api_key:
        try:
            legacy = _legacy_get_api_key("alpha_vantage")
            if legacy and legacy.strip():
                return legacy.strip(), "legacy"
        except Exception:
            pass

    if _HAS_DOTENV:
        try:
            load_dotenv()
            env_key2 = os.getenv("ALPHA_VANTAGE_API_KEY") or os.getenv("ALPHAVANTAGE_KEY")
            if env_key2 and env_key2.strip():
                return env_key2.strip(), ".env"
        except Exception:
            pass

    return None, "none"

def _normalize_source(s: str) -> str:
    s = (s or "").strip()
    return s if s else "AlphaVantage"

# ----------------------------- fetch -----------------------------
def fetch_alpha_vantage_news(ticker: str, *, api_key: str, max_items: int = AV_MAX) -> List[Dict]:
    """
    Fetch up to `max_items` Alpha Vantage news items for `ticker`.

    Output schema (aligned with sentiment pipeline / YF fetcher):
    {
      "id", "ticker", "time_published", "source", "title", "url",
      "provider": "alpha_vantage",
      "body": "", "word_count": 0,
      "meta": {"av": {"overall_sentiment_label": "...", "overall_sentiment_score": float}}
    }
    """
    limit = max(1, min(int(max_items), AV_MAX))
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "apikey": api_key,
        "sort": "LATEST",
        "limit": limit,
    }
    headers = {
        "User-Agent": "FreshStart/1.0 (+capstone) Python-requests",
        "Accept": "application/json",
    }

    r = requests.get(AV_BASE, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()

    # AV-specific error surfaces
    if isinstance(data, dict):
        if "Note" in data:
            raise RuntimeError(f"Alpha Vantage rate limit: {data.get('Note')}")
        if "Information" in data:
            raise RuntimeError(f"Alpha Vantage error: {data.get('Information')}")
        if "Error Message" in data:
            raise RuntimeError(f"Alpha Vantage error: {data.get('Error Message')}")

    feed = data.get("feed") or []
    out: List[Dict] = []
    for it in feed[:limit]:
        title = _clean_title(it.get("title") or "")
        url = (it.get("url") or "").strip()
        time_pub = _to_iso8601(it.get("time_published") or "")
        source = _normalize_source(it.get("source") or "")

        av_sent = (it.get("overall_sentiment_label") or "").strip().lower()
        av_score = it.get("overall_sentiment_score", None)

        uid = _sha1_id("av", url, title, time_pub, ticker)

        out.append({
            "id": uid,
            "ticker": ticker,
            "time_published": time_pub,
            "source": source,
            "title": title,
            "url": url,
            "provider": "alpha_vantage",
            "body": "",
            "word_count": 0,
            "meta": {
                "av": {
                    "overall_sentiment_label": av_sent,
                    "overall_sentiment_score": av_score,
                }
            },
        })
    return out

# ----------------------------- CLI -----------------------------
def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Fetch Alpha Vantage news (max 10) normalized for SentimentAgent.")
    ap.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    ap.add_argument("--api-key", help="Alpha Vantage API key (overrides env/legacy)")
    ap.add_argument("--max-items", type=int, default=AV_MAX, help="Max AV items (hard-capped at 10)")
    args = ap.parse_args()

    key, source = _resolve_api_key(args.api_key)
    if key:
        tail = key[-4:] if len(key) >= 4 else "****"
        print(f"[AV] Using key from {source}: ...{tail}")
    else:
        print("[AV] No key found in env/legacy/.env/cli.")
        print("Set ALPHA_VANTAGE_API_KEY or ALPHAVANTAGE_KEY, or pass --api-key.")
        return

    cap = min(args.max_items, AV_MAX)
    print(f"\nFetching Alpha Vantage news for {args.ticker} (max {cap})\n")

    try:
        items = fetch_alpha_vantage_news(args.ticker, api_key=key, max_items=cap)
    except Exception as e:
        print(f"Failed to fetch AV news: {e}")
        items = []

    print(f"Returned: {len(items)} articles\n")

    for i, it in enumerate(items, start=1):
        src = (it.get("source") or "av").lower()
        tpub = it.get("time_published") or "-"
        title = shorten(_clean_title(it.get("title") or ""), width=90, placeholder=" [...]")
        print(f"AV {i:02d}. [{src}] {tpub}  {title}")

    print("\n--- JSON OUTPUT ---")
    print(json.dumps(items, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    _cli()
