"""Reddit sentiment ingestion scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(slots=True)
class RedditStreamConfig:
    """Configuration for streaming Reddit posts/comments via third-party wrappers."""

    client_id: str
    client_secret: str
    user_agent: str
    subreddits: Iterable[str]
    destination: str = "data/realtime/landing/social/reddit/"
    poll_interval_seconds: int = 60
    max_items_per_poll: int = 100
    since: Optional[str] = None


def stream_reddit_posts(config: RedditStreamConfig) -> None:
    """Stream subreddit activity such as r/WallStreetBets for sentiment analysis."""

    raise NotImplementedError(
        "Implement PRAW/Pushshift integration, text normalization, and FinBERT-ready serialization."
    )
