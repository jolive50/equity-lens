"""Tiingo EOD price ingestion with pagination and CSV persistence."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import requests
import pandas as pd

_LOGGER = logging.getLogger(__name__)

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
    """Ingest Tiingo end-of-day prices into the batch landing area.

    Docs: https://api.tiingo.com/documentation/end-of-day
    """

    dest = Path(config.destination).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"Content-Type": "application/json"})

    for ticker in config.tickers:
        url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        params = {
            "token": config.api_key,
            "startDate": config.start_date,
            "endDate": config.end_date,
            "resampleFreq": config.frequency,
            "format": "json",
        }
        try:
            resp = session.get(url, params={k: v for k, v in params.items() if v is not None}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # pragma: no cover - network dependent
            _LOGGER.error("Tiingo request failed for %s: %s", ticker, exc)
            continue

        if not data:
            _LOGGER.info("No Tiingo rows for %s", ticker)
            continue

        frame = pd.DataFrame(data)
        if "date" in frame:
            frame["date"] = pd.to_datetime(frame["date"]).dt.tz_convert(None)
            frame = frame.sort_values("date")

        out_dir = dest / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_file = out_dir / f"{ticker}_{config.frequency}_{ts}.csv"
        frame.to_csv(out_file, index=False)
        _LOGGER.info("Saved Tiingo %s rows for %s to %s", len(frame), ticker, out_file)
