#!/usr/bin/env python3
"""StockSense training-data collection script.

The script pulls real market data for a set of tickers and stores the results
under ``data/training`` so that the modelling pipelines can consume a consistent
dataset.  It uses the public Yahoo Finance endpoints exposed via ``yfinance``.

Usage
-----
    python scripts/collect_training_data.py --tickers AAPL MSFT GOOGL

If no tickers are provided the script falls back to a curated starter list of
liquid S&P 500 constituents.  The output folders are created automatically.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import yfinance as yf

logger = logging.getLogger("stocksense.collect_training_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TRAINING_ROOT = Path("data") / "training"
TRAINING_SUBFOLDERS = ("market_data", "fundamentals", "labels")

# Classification thresholds expressed as daily percentage moves.
UP_THRESHOLD = 0.02
DOWN_THRESHOLD = -0.02


def ensure_directories(base_dir: Path = TRAINING_ROOT) -> Dict[str, Path]:
    """Create the training directory tree if it does not exist."""

    paths: Dict[str, Path] = {}
    for name in TRAINING_SUBFOLDERS:
        folder = base_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured directory exists: %s", folder.resolve())
        paths[name] = folder
    return paths


def fetch_historical_prices(ticker: str, period: str) -> List[Dict[str, float]]:
    """Download historical OHLCV data for ``ticker``."""

    logger.info("Fetching %s of historical prices for %s", period, ticker)
    history = yf.Ticker(ticker).history(period=period, auto_adjust=True)

    if history.empty:
        raise RuntimeError(f"No historical price data returned for {ticker}")

    records: List[Dict[str, float]] = []
    for date, row in history.iterrows():
        records.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "adj_close": float(row["Close"]),
                "volume": int(row["Volume"]),
            }
        )
    return records


def fetch_fundamentals(ticker: str) -> Dict[str, float]:
    """Extract a curated subset of fundamentals reported by Yahoo Finance."""

    logger.info("Fetching fundamentals for %s", ticker)
    info = yf.Ticker(ticker).info
    if not info:
        raise RuntimeError(f"No fundamentals found for {ticker}")

    metrics = {
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "dividend_yield": info.get("dividendYield"),
        "profit_margin": info.get("profitMargins"),
        "return_on_equity": info.get("returnOnEquity"),
        "debt_to_equity": info.get("debtToEquity"),
        "beta": info.get("beta"),
        "revenue_growth": info.get("revenueGrowth"),
    }

    # Normalise missing values so downstream code does not break.
    return {key: float(value) if isinstance(value, (int, float)) else 0.0 for key, value in metrics.items()}


def build_direction_labels(market_data: Sequence[Dict[str, float]], lookahead_days: int = 5) -> List[Dict[str, float]]:
    """Create directional labels based on forward returns."""

    labels: List[Dict[str, float]] = []
    for idx in range(len(market_data) - lookahead_days):
        current = market_data[idx]
        future = market_data[idx + lookahead_days]

        current_price = current["close"]
        future_price = future["close"]
        change_pct = (future_price - current_price) / current_price

        if change_pct >= UP_THRESHOLD:
            direction = "up"
        elif change_pct <= DOWN_THRESHOLD:
            direction = "down"
        else:
            direction = "neutral"

        labels.append(
            {
                "date": current["date"],
                "direction": direction,
                "price_change_pct": round(change_pct, 4),
                "current_price": current_price,
                "future_price": future_price,
                "lookahead_days": lookahead_days,
            }
        )
    logger.info(
        "Generated %s labels (%s up / %s down / %s neutral)",
        len(labels),
        sum(1 for item in labels if item["direction"] == "up"),
        sum(1 for item in labels if item["direction"] == "down"),
        sum(1 for item in labels if item["direction"] == "neutral"),
    )
    return labels


def save_json(payload: Dict | List, destination: Path) -> None:
    """Persist ``payload`` as pretty-printed JSON."""

    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Saved %s", destination.resolve())


def collect_for_ticker(ticker: str, *, period: str, folders: Dict[str, Path]) -> None:
    """Collect prices, fundamentals and labels for ``ticker``."""

    prices = fetch_historical_prices(ticker, period=period)
    fundamentals = fetch_fundamentals(ticker)
    labels = build_direction_labels(prices)

    save_json(prices, folders["market_data"] / f"{ticker}.json")
    save_json(fundamentals, folders["fundamentals"] / f"{ticker}.json")
    save_json(labels, folders["labels"] / f"{ticker}.json")


def default_tickers() -> List[str]:
    """Provide a starter list of widely traded symbols."""

    return [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "META",
        "NVDA",
        "TSLA",
        "JPM",
        "V",
        "WMT",
        "JNJ",
        "PG",
    ]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect historical market data for training.")
    parser.add_argument("--tickers", nargs="*", help="Ticker symbols to collect (default: curated S&P 500 sample).")
    parser.add_argument("--period", default="2y", help="Lookback window understood by yfinance (default: 2y).")
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Politeness pause between requests in seconds (default: 2.0).",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    tickers = args.tickers or default_tickers()

    logger.info("Collecting training data for %s tickers: %s", len(tickers), ", ".join(tickers))
    folders = ensure_directories()

    successes = 0
    for index, ticker in enumerate(tickers, start=1):
        logger.info("[%s/%s] Processing %s", index, len(tickers), ticker)
        try:
            collect_for_ticker(ticker.upper(), period=args.period, folders=folders)
            successes += 1
        except Exception as exc:  # noqa: BLE001 - we want to keep looping
            logger.exception("Failed to collect data for %s: %s", ticker, exc)
        finally:
            if index < len(tickers) and args.sleep > 0:
                time.sleep(args.sleep)

    logger.info(
        "Completed data collection: %s succeeded / %s failed (output root: %s)",
        successes,
        len(tickers) - successes,
        TRAINING_ROOT.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
