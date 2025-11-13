"""
Yahoo Finance news fetcher (yfinance-only) for sentiment preprocessing.

Behavior:
- Pull up to `max_items` Yahoo Finance articles per ticker (defaults to 25; many tickers return ~10).
- Best-effort extraction of article body text with fallbacks.
- Output is normalized for the sentiment pipeline (title/body/word_count present).

Normalized item:
{
  "id": <sha1(url|title|time|ticker)>,
  "ticker": "AAPL",
  "time_published": "YYYY-MM-DDTHH:MM:SSZ",
  "source": "reuters",
  "title": "Headline...",
  "url": "https://...",
  "provider": "yfinance",
  "body": "full text (truncated, if available)",
  "word_count": 123
}
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests

# Optional deps
try:
    from newspaper import Article as _NPArticle
    _NEWSPAPER_AVAILABLE = True
except Exception:
    _NEWSPAPER_AVAILABLE = False

try:
    from readability import Document as _ReadabilityDoc
    _READABILITY_AVAILABLE = True
except Exception:
    _READABILITY_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except Exception:
    _BS4_AVAILABLE = False

# yfinance
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except Exception:
    _YF_AVAILABLE = False

log = logging.getLogger(__name__)

# Tunables
DEFAULT_MAX_YF = 25
FINAL_CAP = 50
MAX_BODY_CHARS = 2000
MIN_BODY_WORDS = 60

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
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(ts, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""

def _first_str(*values) -> str:
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict):
            for k in ("title", "text", "label", "name", "provider", "publisher",
                      "source", "canonicalUrl", "url", "link", "headline"):
                s = v.get(k)
                if isinstance(s, str) and s.strip():
                    return s.strip()
        if isinstance(v, (list, tuple)) and v:
            for s in v:
                if isinstance(s, str) and s.strip():
                    return s.strip()
    return ""

def _normalize_domain(netloc: str) -> str:
    return re.sub(r"^(www\.|finance\.)", "", netloc or "", flags=re.I).lower()

def _clean_title(t: str, *, max_len: int = 140) -> str:
    """Remove stray characters and trim titles nicely."""
    if not t:
        return ""
    t = t.replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"^[\-\–\—\·\•\|]+", "", t)
    t = t.strip(" -–—·•|")
    t = t.strip()
    if t.endswith("?"):  # remove question marks
        t = t[:-1].strip()
    return (t[:max_len].rstrip() + "...") if len(t) > max_len else t

# ---------------------- body extraction helpers --------------------
def _looks_like_bot_block(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(s in low for s in ["are you a robot", "enable javascript", "access denied"])

def _enforce_quality_or_empty(text: str) -> str:
    if not text or len(text.split()) < MIN_BODY_WORDS:
        return ""
    if _looks_like_bot_block(text):
        return ""
    return text

def _fetch_html(url: str, *, timeout: int = 15) -> str:
    if not url:
        return ""
    headers = {
        "User-Agent": random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6)",
        ]),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code >= 400:
            return ""
        return r.text
    except Exception:
        return ""

def _clean_text(txt: str) -> str:
    """Remove junk lines, whitespace, and HTML residue."""
    if not txt:
        return ""
    txt = txt.replace("\u00a0", " ").strip()
    junk = ["Read more", "Subscribe", "Sign in", "Advertisement",
            "All rights reserved", "Cookie", "Privacy Policy"]
    lines = []
    for line in txt.splitlines():
        s = line.strip()
        if not s:
            continue
        if any(j.lower() in s.lower() for j in junk):
            continue
        lines.append(s)
    txt = " ".join(lines)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

# ----------------------------- yfinance -----------------------------
def _normalize_feed_item_yf(raw: dict, ticker: str) -> dict:
    title = _first_str(raw.get("title"), (raw.get("content") or {}).get("title"), raw.get("headline"))
    title = _clean_title(title)
    url = _first_str(raw.get("link"), raw.get("url"), (raw.get("content") or {}).get("canonicalUrl"))
    source = _first_str(raw.get("publisher"), raw.get("source"), raw.get("provider")) or "yahoo"
    t_val = raw.get("providerPublishTime") or raw.get("pubDate") or raw.get("date")
    time_pub = ""
    if isinstance(t_val, (int, float)) or (isinstance(t_val, str) and t_val.isdigit()):
        epoch = int(t_val)
        if epoch > 10**12:
            epoch //= 1000
        time_pub = datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    elif isinstance(t_val, str):
        time_pub = _to_iso8601(t_val)
    uid = _sha1_id(url, title, time_pub, ticker)
    return {
        "id": uid, "ticker": ticker, "time_published": time_pub,
        "source": source, "title": title, "url": url, "provider": "yfinance"
    }

def fetch_yfinance_news(ticker: str, *, max_items: int = DEFAULT_MAX_YF) -> List[Dict]:
    if not _YF_AVAILABLE:
        raise RuntimeError("yfinance not installed.")
    tkr = yf.Ticker(ticker)
    raw_list = tkr.news or []
    return [_normalize_feed_item_yf(a, ticker) for a in raw_list[:min(max_items, FINAL_CAP)]]

# -------------------- body enrichment ----------------------
def enrich_with_body(news_items: List[Dict], *, max_chars: int = MAX_BODY_CHARS) -> List[Dict]:
    for it in news_items:
        it["body"] = ""
        it["word_count"] = 0
        url = it.get("url")
        if not url:
            continue

        html = _fetch_html(url)
        text = ""

        if _NEWSPAPER_AVAILABLE:
            try:
                art = _NPArticle(url)
                art.download()
                art.parse()
                text = art.text or ""
            except Exception:
                pass

        if not text and _READABILITY_AVAILABLE and html:
            try:
                doc = _ReadabilityDoc(html)
                soup = BeautifulSoup(doc.summary(), "html.parser")
                text = " ".join(p.get_text(separator=" ", strip=True) for p in soup.find_all("p"))
            except Exception:
                pass

        if not text and _BS4_AVAILABLE and html:
            try:
                soup = BeautifulSoup(html, "html.parser")
                ps = soup.find_all("p")
                text = " ".join(p.get_text(separator=" ", strip=True) for p in ps)
            except Exception:
                pass

        text = _clean_text(text)
        text = _enforce_quality_or_empty(text)
        if text:
            if len(text) > max_chars:
                text = text[:max_chars].rstrip() + "..."
            it["body"] = text
            it["word_count"] = len(text.split())

        time.sleep(0.1)
    return news_items

def fetch_news_yf_only(ticker: str, *, max_items: int = DEFAULT_MAX_YF) -> List[Dict]:
    """Fetch and enrich Yahoo Finance articles for sentiment models."""
    yf_items = []
    try:
        yf_items = fetch_yfinance_news(ticker, max_items=max_items)
    except Exception as e:
        log.warning("yfinance fetch failed: %s", e)
    return enrich_with_body(yf_items, max_chars=MAX_BODY_CHARS)[:min(max_items, FINAL_CAP)]

# ----------------------------- CLI -----------------------------
if __name__ == "__main__":
    import argparse
    from textwrap import shorten

    ap = argparse.ArgumentParser(description="Fetch Yahoo Finance news only.")
    ap.add_argument("ticker", help="Ticker symbol (e.g., AAPL, MSFT)")
    ap.add_argument("--max-items", type=int, default=DEFAULT_MAX_YF)
    ap.add_argument("--print-json", action="store_true")
    args = ap.parse_args()

    print(f"\nFetching Yahoo Finance news for {args.ticker} (max {args.max_items})\n")
    items = fetch_news_yf_only(args.ticker, max_items=args.max_items)
    print(f"Returned: {len(items)} articles\n")

    for i, item in enumerate(items, start=1):
        src = item.get("source", "yahoo").lower()
        time_pub = item.get("time_published") or "-"
        title = shorten(item.get("title") or "", width=70, placeholder="...")
        print(f"YF {i:02d}. [{src:>8}] {time_pub}  {title}")

    if args.print_json:
        print("\n--- JSON ---")
        print(json.dumps(items, indent=2, ensure_ascii=False))
