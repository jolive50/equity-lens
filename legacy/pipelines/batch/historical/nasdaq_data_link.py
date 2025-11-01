"""Nasdaq Data Link (Quandl) ingestion scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class NasdaqDataLinkConfig:
    """Configuration for downloading datasets from Nasdaq Data Link."""

    api_key: Optional[str]
    dataset_code: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    collapse: Optional[str] = None
    transform: Optional[str] = None
    destination: str = "data/batch/raw/market/nasdaq_data_link/"


def fetch_nasdaq_dataset(config: NasdaqDataLinkConfig) -> None:
    """Download Nasdaq Data Link tables and persist them into Parquet."""

    raise NotImplementedError(
        "Implement dataset download, schema harmonization, and metadata logging for Nasdaq Data Link."
    )
