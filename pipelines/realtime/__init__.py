"""LangChain-only orchestration utilities for realtime pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

from langchain_core.runnables import RunnableLambda, RunnableParallel

from .news.yahoo_finance import YahooNewsConfig, poll_yahoo_news
from .news.newsapi import NewsAPIConfig, poll_newsapi_headlines


_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CollectNewsRequest:
    tickers: Iterable[str]
    newsapi_key: Optional[str] = None


def _yahoo_node(req: CollectNewsRequest) -> str:
    cfg = YahooNewsConfig(tickers=req.tickers)
    poll_yahoo_news(cfg)
    return "yahoo_ok"


def _newsapi_node(req: CollectNewsRequest) -> str:
    if not req.newsapi_key:
        return "newsapi_skipped"
    cfg = NewsAPIConfig(api_key=req.newsapi_key, tickers=req.tickers)
    poll_newsapi_headlines(cfg)
    return "newsapi_ok"


def run_collect_news(req: CollectNewsRequest) -> dict:
    """Run a simple parallel LangChain graph to collect news from sources."""

    graph = RunnableParallel(
        yahoo=RunnableLambda(_yahoo_node),
        newsapi=RunnableLambda(_newsapi_node),
    )
    result = graph.invoke(req)
    _LOGGER.info("CollectNews result: %s", result)
    return result


