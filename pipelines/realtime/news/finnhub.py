"""Finnhub company news polling with checkpoints and JSONL sink."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import requests

@dataclass(slots=True)
class FinnhubNewsConfig:
    """Configuration options for Finnhub news streaming."""

    api_key: str
    symbols: Iterable[str]
    destination: str = "data/realtime/landing/news/finnhub/"
    poll_interval_seconds: int = 60
    max_calls_per_minute: int = 60
    session_retries: int = 3
    since: Optional[str] = None  # ISO date like "2024-01-01"
    checkpoint_dir: str = "data/realtime/feature_store/checkpoints/finnhub_news/"
    base_url: str = "https://finnhub.io/api/v1/company-news"


def stream_finnhub_news(config: FinnhubNewsConfig) -> None:
    """Fetch company news for symbols and append unseen articles to JSONL files."""

    logger = logging.getLogger(__name__)
    destination = Path(config.destination).expanduser().resolve()
    checkpoint_root = Path(config.checkpoint_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    def _normalise(values: Iterable[str]):
        if isinstance(values, str):
            return [values.strip().upper()]
        return [v.strip().upper() for v in values if v]

    def _load_seen(path: Path):
        if not path.exists():
            return set()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return set(str(x) for x in data)
        except Exception:  # pragma: no cover
            return set()
        return set()

    def _save_seen(path: Path, seen):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(seen)), encoding="utf-8")

    symbols = _normalise(config.symbols)
    if not symbols:
        raise ValueError("No symbols provided for Finnhub news.")

    # Determine from/to window
    today = datetime.utcnow().date()
    frm = datetime.strptime(config.since, "%Y-%m-%d").date() if config.since else today - timedelta(days=7)
    to = today

    for sym in symbols:
        params = {"symbol": sym, "from": str(frm), "to": str(to), "token": config.api_key}
        try:
            resp = requests.get(config.base_url, params=params, timeout=30)
            resp.raise_for_status()
            articles = resp.json() or []
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error("Finnhub request failed for %s: %s", sym, exc)
            continue

        checkpoint_path = checkpoint_root / f"{sym}.json"
        seen = _load_seen(checkpoint_path)

        out_dir = destination / sym
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "news.jsonl"

        new_ids = []
        with out_file.open("a", encoding="utf-8") as fp:
            for art in articles:
                url = art.get("url") or art.get("id")
                if not url or url in seen:
                    continue
                record = {
                    "ticker": sym,
                    "source": "finnhub",
                    "id": url,
                    "title": art.get("headline"),
                    "description": art.get("summary"),
                    "url": art.get("url"),
                    "publisher": art.get("source"),
                    "published_time_unix": art.get("datetime"),
                    "raw": art,
                    "ingested_at": datetime.utcnow().isoformat() + "Z",
                }
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                new_ids.append(str(url))

        if new_ids:
            seen.update(new_ids)
            _save_seen(checkpoint_path, seen)
            logger.info("Appended %d Finnhub articles for %s", len(new_ids), sym)
        else:
            logger.info("No new Finnhub articles for %s", sym)
