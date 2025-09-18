"""Finnhub news streaming scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(slots=True)
class FinnhubNewsConfig:
    """Configuration options for Finnhub news streaming."""

    api_key: str
    symbols: Iterable[str]
    destination: str = "data/realtime/landing/news/finnhub/"
    poll_interval_seconds: int = 60
    max_calls_per_minute: int = 60
    session_retries: int = 3
    since: Optional[str] = None


def stream_finnhub_news(config: FinnhubNewsConfig) -> None:
    """Stream Finnhub news and persist to a low-latency landing store."""

    raise NotImplementedError(
        "Implement Finnhub polling schedule, API rate accounting, and append-only persistence."
    )
