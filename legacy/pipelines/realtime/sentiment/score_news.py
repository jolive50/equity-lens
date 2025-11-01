from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from .finbert import score_with_finbert, FinBERTConfig


def _read_jsonl(path: Path, limit: int | None = None) -> List[str]:
    texts: List[str] = []
    if not path.exists():
        return texts
    with path.open("r", encoding="utf-8") as fp:
        for i, line in enumerate(fp):
            if limit is not None and i >= limit:
                break
            try:
                obj = json.loads(line)
                text = obj.get("title") or obj.get("description") or obj.get("raw") or ""
                if isinstance(text, dict):
                    text = text.get("title") or text.get("body") or ""
                if text:
                    texts.append(str(text))
            except Exception:
                continue
    return texts


def collect_news_texts(tickers: Iterable[str], base: Path) -> List[str]:
    items: List[str] = []
    for ticker in tickers:
        for source in ("yahoo_finance", "newsapi", "finnhub"):
            path = base / source / ticker / "news.jsonl"
            items.extend(_read_jsonl(path))
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description="Score latest news with FinBERT and write sentiment features")
    ap.add_argument("--tickers", nargs="+", default=["AAPL"], help="Tickers to aggregate news from")
    ap.add_argument("--limit", type=int, default=500, help="Max texts to score")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()

    news_base = Path("data/realtime/landing/news").resolve()
    texts = collect_news_texts(args.tickers, news_base)
    if not texts:
        print("No news found to score. Run news collectors first.")
        return

    texts = texts[-args.limit :]
    cfg = FinBERTConfig(device=args.device)
    out = score_with_finbert(texts, cfg)
    print(f"Wrote sentiment features to {out}")


if __name__ == "__main__":
    main()


