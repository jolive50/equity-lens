"""yfinance-powered historical price ingestion scaffolding."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import pandas as pd
import yfinance as yf

try:  # pragma: no cover - optional dependency
    from great_expectations.dataset import PandasDataset
except Exception:  # pragma: no cover - graceful degradation
    PandasDataset = None


_LOGGER = logging.getLogger(__name__)
_ALLOWED_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"}


@dataclass(slots=True)
class YahooFinanceConfig:
    """Configuration contract for yfinance historical downloads."""

    tickers: Iterable[str]
    start: Optional[str] = None
    end: Optional[str] = None
    interval: Literal[
        "1m",
        "2m",
        "5m",
        "15m",
        "30m",
        "60m",
        "90m",
        "1h",
        "1d",
        "5d",
        "1wk",
        "1mo",
        "3mo",
    ] = "1d"
    auto_adjust: bool = True
    progress: bool = False
    destination: str = "../data/batch/raw/market/yahoo_finance/"

    def __post_init__(self) -> None:
        if self.interval not in _ALLOWED_INTERVALS:
            raise ValueError(f"Unsupported interval '{self.interval}'.")


def _normalise_tickers(tickers: Iterable[str]) -> Sequence[str]:
    if isinstance(tickers, str):
        return [tickers.strip().upper()]

    cleaned = [ticker.strip().upper() for ticker in tickers if ticker]
    return list(dict.fromkeys(cleaned))


def _validate_frame(ticker: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise ValueError(f"Downloaded dataframe for {ticker} is empty.")

    if PandasDataset is None:
        if frame.isna().all().any():  # pragma: no cover - defensive branch
            raise ValueError(f"Dataframe for {ticker} contains only null columns.")
        return

    dataset = PandasDataset(frame.reset_index())
    dataset.expect_column_values_to_not_be_null("Date")
    dataset.expect_table_row_count_to_be_between(min_value=1)
    for column in ("Open", "High", "Low", "Close", "Volume"):
        if column in dataset.columns:
            dataset.expect_column_values_to_not_be_null(column)

    validation = dataset.validate()
    if not validation.success:
        raise ValueError(f"Great Expectations validation failed for {ticker}: {validation}")


def _persist_frame(destination: Path, ticker: str, frame: pd.DataFrame, interval: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    ticker_dir = destination / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)

    file_path = ticker_dir / f"{ticker}_{interval}_{timestamp}.csv"
    try:
        frame.to_csv(file_path)
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        frame.to_parquet(file_path)
    return file_path


def fetch_yahoo_prices(config: YahooFinanceConfig) -> None:
    """Download and persist Yahoo Finance history to the batch landing zone."""

    destination = Path(config.destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    tickers = _normalise_tickers(config.tickers)
    if not tickers:
        raise ValueError("No tickers were provided to fetch.")

    any_saved = False

    for ticker in tickers:
        _LOGGER.info("Downloading %s data from Yahoo Finance.", ticker)
        try:
            frame = yf.download(  # type: ignore[call-arg]
                ticker,
                start=config.start,
                end=config.end,
                interval=config.interval,
                auto_adjust=config.auto_adjust,
                progress=config.progress,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            _LOGGER.error("Failed to download %s: %s", ticker, exc)
            continue

        if not isinstance(frame, pd.DataFrame):
            _LOGGER.warning("Unexpected payload type for %s: %s", ticker, type(frame))
            continue

        if frame.empty:
            _LOGGER.warning("No rows returned for %s from Yahoo Finance.", ticker)
            continue

        if isinstance(frame, pd.DataFrame) and frame.columns.nlevels > 1:
            frame = frame.droplevel(0, axis=1)

        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index)

        if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
            frame.index = frame.index.tz_convert(None)

        frame = frame.sort_index()

        _validate_frame(ticker, frame)
        file_path = _persist_frame(destination, ticker, frame, config.interval)
        _LOGGER.info("Persisted %s rows for %s to %s", len(frame), ticker, file_path)

        any_saved = True

    if not any_saved:
        raise RuntimeError("No Yahoo Finance downloads succeeded. Check network connectivity or ticker symbols.")

