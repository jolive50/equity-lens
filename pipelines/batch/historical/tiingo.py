"""Tiingo ingestion scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(slots=True)
class TiingoConfig:
    """Configuration options for Tiingo price downloads."""

    api_key: str
    tickers: Iterable[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    frequency: str = "daily"
    destination: str = "data/batch/raw/market/tiingo/"
    session_retries: int = 3


def fetch_tiingo_prices(config: TiingoConfig) -> None:
    """Ingest Tiingo end-of-day prices into the batch landing area."""

    raise NotImplementedError(
        "Implement Tiingo client session management, pagination, and persistence logic."
    )
