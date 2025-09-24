"""NewsAPI headline polling with incremental checkpointing and JSONL sink."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set

import requests


@dataclass(slots=True)
class NewsAPIConfig:
    """Configuration parameters for NewsAPI polling."""

    api_key: str
    tickers: Iterable[str]
    page_size: int = 100
    language: str = "en"
    destination: str = "data/realtime/landing/news/newsapi/"
    sort_by: str = "publishedAt"
    checkpoint_dir: str = "data/realtime/feature_store/checkpoints/newsapi/"
    from_param: Optional[str] = None  # ISO8601
    base_url: str = "https://newsapi.org/v2/everything"


def poll_newsapi_headlines(config: NewsAPIConfig) -> None:
    """Poll NewsAPI for headlines per ticker and append unseen records to JSONL.

    Notes:
        - Uses simple per-ticker checkpoint of seen article URLs.
        - If `from_param` not provided, uses last checkpointed timestamp if available.
    """

    logger = logging.getLogger(__name__)
    destination = Path(config.destination).expanduser().resolve()
    checkpoint_root = Path(config.checkpoint_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    def _normalise_tickers(values: Iterable[str]) -> List[str]:
        if isinstance(values, str):
            return [values.strip().upper()]
        return [v.strip().upper() for v in values if v]

    def _load_seen(path: Path) -> Set[str]:
        if not path.exists():
            return set()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return set(str(x) for x in data)
        except Exception:  # pragma: no cover
            return set()
        return set()

    def _save_seen(path: Path, seen: Set[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sorted(seen)), encoding="utf-8")

    def _request(query: str, page: int) -> dict:
        params = {
            "q": query,
            "language": config.language,
            "sortBy": config.sort_by,
            "pageSize": config.page_size,
            "page": page,
        }
        if config.from_param:
            params["from"] = config.from_param
        headers = {"X-Api-Key": config.api_key}
        resp = requests.get(config.base_url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()

    tickers = _normalise_tickers(config.tickers)
    if not tickers:
        raise ValueError("No tickers provided for NewsAPI polling.")

    any_written = False

    for ticker in tickers:
        query = f"{ticker} OR {ticker} stock"
        checkpoint_path = checkpoint_root / f"{ticker}.json"
        seen = _load_seen(checkpoint_path)

        out_dir = destination / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "news.jsonl"

        page = 1
        new_urls: List[str] = []
        with out_file.open("a", encoding="utf-8") as fp:
            while True:
                try:
                    payload = _request(query, page)
                except Exception as exc:  # pragma: no cover - network dependent
                    logger.error("NewsAPI request failed for %s page %s: %s", ticker, page, exc)
                    break

                articles = payload.get("articles") or []
                if not articles:
                    break

                for art in articles:
                    url = art.get("url")
                    if not url or url in seen:
                        continue
                    record = {
                        "ticker": ticker,
                        "source": "newsapi",
                        "id": url,
                        "title": art.get("title"),
                        "description": art.get("description"),
                        "url": url,
                        "publisher": (art.get("source") or {}).get("name"),
                        "published_time": art.get("publishedAt"),
                        "raw": art,
                        "ingested_at": datetime.utcnow().isoformat() + "Z",
                    }
                    fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                    new_urls.append(url)

                if len(articles) < config.page_size:
                    break
                page += 1

        if new_urls:
            seen.update(new_urls)
            _save_seen(checkpoint_path, seen)
            logger.info("Appended %d NewsAPI articles for %s", len(new_urls), ticker)
            any_written = True
        else:
            logger.info("No new NewsAPI articles for %s", ticker)

    if not any_written:
        logger.info("NewsAPI polling completed with no new items.")
