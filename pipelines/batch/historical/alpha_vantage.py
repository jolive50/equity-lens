"""Alpha Vantage ingestion for daily/weekly time series with CSV persistence."""

from __future__ import annotations

import time
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal, Optional

import requests
import pandas as pd

_LOGGER = logging.getLogger(__name__)

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
    """Pull Alpha Vantage OHLCV data and land it in the batch landing area.

    Notes:
        - Respects `throttle_seconds` to avoid rate limits.
        - Writes one CSV per symbol per run under destination/<SYMBOL>/.
    """

    dest = Path(config.destination).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    base = "https://www.alphavantage.co/query"

    for sym in config.symbols:
        params = {
            "function": config.function,
            "symbol": sym,
            "apikey": config.api_key,
            "datatype": "json",
            "outputsize": config.outputsize,
        }
        try:
            resp = requests.get(base, params=params, timeout=config.timeout)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # pragma: no cover - network dependent
            _LOGGER.error("Alpha Vantage request failed for %s: %s", sym, exc)
            continue

        # Identify time series key by function
        ts_key_map = {
            "TIME_SERIES_DAILY": "Time Series (Daily)",
            "TIME_SERIES_DAILY_ADJUSTED": "Time Series (Daily)",
            "TIME_SERIES_WEEKLY": "Weekly Time Series",
        }
        ts_key = ts_key_map[config.function]
        series = payload.get(ts_key)
        if not series:
            _LOGGER.warning("No series in Alpha Vantage payload for %s: %s", sym, payload.get("Note") or list(payload)[:3])
            time.sleep(config.throttle_seconds)
            continue

        frame = (
            pd.DataFrame(series).T
            .rename(columns=lambda c: c.split(". ")[-1].lower())
            .rename(columns={"adjusted close": "adj_close"})
        )
        frame.index.name = "date"
        frame = frame.sort_index()

        out_dir = dest / sym
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_file = out_dir / f"{sym}_{config.function.lower()}_{ts}.csv"
        frame.to_csv(out_file)
        _LOGGER.info("Saved Alpha Vantage %s rows for %s to %s", len(frame), sym, out_file)

        time.sleep(config.throttle_seconds)
