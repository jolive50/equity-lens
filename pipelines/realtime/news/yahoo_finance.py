"""Yahoo Finance news polling via yfinance with simple dedup and JSONL sink."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import yfinance as yf


_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class YahooNewsConfig:
    """Configuration for polling Yahoo Finance headlines via yfinance.

    Notes:
        - This function performs a one-shot fetch per ticker and writes new items.
        - Deduplication is achieved with a per-ticker checkpoint file of seen IDs/URLs.
    """

    tickers: Iterable[str]
    language: Optional[str] = None
    destination: str = "data/realtime/landing/news/yahoo_finance/"
    checkpoint_dir: str = "data/realtime/feature_store/checkpoints/yahoo_finance_news/"


def _normalise_tickers(tickers: Iterable[str]) -> List[str]:
    if isinstance(tickers, str):
        return [tickers.strip().upper()]
    return [t.strip().upper() for t in tickers if t]


def _load_checkpoint(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return set(str(x) for x in data)
    except Exception:  # pragma: no cover - defensive
        return set()
    return set()


def _save_checkpoint(path: Path, seen: Set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(seen)), encoding="utf-8")


def _news_identity(item: Dict) -> Optional[str]:
    # yfinance typically provides an 'uuid' or 'id'; fall back to 'link'
    for key in ("uuid", "id", "link", "url"):
        value = item.get(key)
        if value:
            return str(value)
    return None


def _serialise_item(ticker: str, item: Dict) -> Dict:
    ts = item.get("providerPublishTime") or item.get("pubDate") or item.get("published_at")
    try:
        published_ts = int(ts) if ts is not None else None
    except Exception:
        published_ts = None
    return {
        "ticker": ticker,
        "source": "yahoo_finance",
        "id": _news_identity(item),
        "title": item.get("title"),
        "description": item.get("summary") or item.get("description"),
        "url": item.get("link") or item.get("url"),
        "publisher": (item.get("publisher") or {}).get("name") if isinstance(item.get("publisher"), dict) else item.get("publisher"),
        "published_time_unix": published_ts,
        "raw": item,
        "ingested_at": datetime.utcnow().isoformat() + "Z",
    }


def poll_yahoo_news(config: YahooNewsConfig) -> None:
    """Fetch latest Yahoo Finance news for tickers and append unseen items to JSONL files.

    Side effects:
        - Creates destination directory per ticker and writes one JSONL record per item.
        - Maintains a simple per-ticker checkpoint JSON list of seen IDs/URLs.
    """

    destination = Path(config.destination).expanduser().resolve()
    checkpoint_root = Path(config.checkpoint_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    tickers = _normalise_tickers(config.tickers)
    if not tickers:
        raise ValueError("No tickers provided for Yahoo Finance news polling.")

    any_written = False

    for ticker in tickers:
        try:
            news_items = yf.Ticker(ticker).news or []  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - network dependent
            _LOGGER.error("Failed to fetch Yahoo news for %s: %s", ticker, exc)
            continue

        if not isinstance(news_items, list) or not news_items:
            _LOGGER.info("No news returned for %s", ticker)
            continue

        checkpoint_path = checkpoint_root / f"{ticker}.json"
        seen = _load_checkpoint(checkpoint_path)

        out_dir = destination / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "news.jsonl"

        new_ids: List[str] = []

        with out_file.open("a", encoding="utf-8") as f:
            for item in news_items:
                identity = _news_identity(item)
                if not identity or identity in seen:
                    continue
                record = _serialise_item(ticker, item)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                new_ids.append(identity)

        if new_ids:
            seen.update(new_ids)
            _save_checkpoint(checkpoint_path, seen)
            _LOGGER.info("Appended %d new Yahoo Finance news items for %s", len(new_ids), ticker)
            any_written = True
        else:
            _LOGGER.info("No new Yahoo Finance news items for %s", ticker)

    if not any_written:
        _LOGGER.info("Yahoo Finance news polling completed with no new items.")
