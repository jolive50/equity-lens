"""NewsAPI headline ingestion scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(slots=True)
class NewsAPIConfig:
    """Configuration parameters for NewsAPI polling."""

    api_key: str
    tickers: Iterable[str]
    page_size: int = 100
    language: str = "en"
    poll_interval_seconds: int = 900
    destination: str = "data/realtime/landing/news/newsapi/"
    sort_by: str = "publishedAt"
    from_param: Optional[str] = None


def poll_newsapi_headlines(config: NewsAPIConfig) -> None:
    """Poll NewsAPI for ticker-linked headlines and push to the landing zone."""

    raise NotImplementedError(
        "Implement NewsAPI requests, symbol keyword expansion, and incremental checkpointing."
    )
