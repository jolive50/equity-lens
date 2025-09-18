"""Alpha Vantage ingestion scaffolding for daily/weekly time series."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional


@dataclass(slots=True)
class AlphaVantageConfig:
    """Configuration required to orchestrate Alpha Vantage downloads."""

    api_key: str
    symbols: Iterable[str]
    function: Literal["TIME_SERIES_DAILY", "TIME_SERIES_DAILY_ADJUSTED", "TIME_SERIES_WEEKLY"] = (
        "TIME_SERIES_DAILY_ADJUSTED"
    )
    outputsize: Literal["compact", "full"] = "full"
    throttle_seconds: float = 15.0
    destination: str = "data/batch/raw/market/alpha_vantage/"
    retries: int = 3
    timeout: Optional[int] = 30


def fetch_alpha_vantage_timeseries(config: AlphaVantageConfig) -> None:
    """Pull Alpha Vantage OHLCV data and land it in object storage."""

    raise NotImplementedError(
        "Implement Alpha Vantage REST calls, rate limit handling, and incremental upserts."
    )
