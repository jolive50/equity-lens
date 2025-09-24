from __future__ import annotations

import argparse
import os
from typing import Iterable, List

from .yahoo_finance import YahooNewsConfig, poll_yahoo_news
from .newsapi import NewsAPIConfig, poll_newsapi_headlines


def _normalize_tickers(values: Iterable[str]) -> List[str]:
    if isinstance(values, str):
        return [values.strip().upper()]
    return [v.strip().upper() for v in values if v]


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect news from Yahoo Finance and NewsAPI")
    ap.add_argument("--tickers", nargs="+", default=["AAPL"], help="Tickers to collect news for")
    ap.add_argument("--newsapi-key", default=None, help="NewsAPI key (optional, falls back to env NEWSAPI_API_KEY)")
    args = ap.parse_args()

    tickers = _normalize_tickers(args.tickers)

    # Yahoo Finance (no API key required)
    yahoo_cfg = YahooNewsConfig(tickers=tickers)
    try:
        poll_yahoo_news(yahoo_cfg)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"Yahoo Finance polling failed: {exc}")

    # NewsAPI (optional if key available)
    api_key = args.newsapi_key or os.getenv("NEWSAPI_API_KEY")
    if api_key:
        newsapi_cfg = NewsAPIConfig(api_key=api_key, tickers=tickers)
        try:
            poll_newsapi_headlines(newsapi_cfg)
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"NewsAPI polling failed: {exc}")
    else:
        print("NEWSAPI_API_KEY not provided; skipping NewsAPI.")


if __name__ == "__main__":
    main()


