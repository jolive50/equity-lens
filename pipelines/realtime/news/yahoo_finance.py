"""Yahoo Finance news polling scaffolding via yfinance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(slots=True)
class YahooNewsConfig:
    """Configuration for polling Yahoo Finance headlines via yfinance."""

    tickers: Iterable[str]
    language: Optional[str] = None
    destination: str = "data/realtime/landing/news/yahoo_finance/"
    poll_interval_seconds: int = 300


def poll_yahoo_news(config: YahooNewsConfig) -> None:
    """Poll Yahoo Finance for the latest headlines and land them into the streaming zone."""

    raise NotImplementedError(
        "Implement yfinance news polling, deduplication, and sink writing for downstream sentiment."
    )
