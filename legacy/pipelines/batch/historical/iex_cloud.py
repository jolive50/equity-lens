"""IEX Cloud ingestion scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(slots=True)
class IEXCloudConfig:
    """Configuration contract for IEX Cloud message-based downloads."""

    api_key: str
    symbols: Iterable[str]
    base_url: str = "https://cloud.iexapis.com/stable"
    endpoint: str = "/stock/{symbol}/chart/1y"
    token_query_param: str = "token"
    destination: str = "data/batch/raw/market/iex_cloud/"
    max_messages: int = 50000
    throttle_seconds: float = 0.25
    retries: int = 2
    timeout: Optional[int] = 30


def fetch_iex_cloud_eod(config: IEXCloudConfig) -> None:
    """Stream chart data from IEX Cloud and persist to object storage."""

    raise NotImplementedError(
        "Implement message counting, error retries, and checkpointed persistence for IEX Cloud."
    )
