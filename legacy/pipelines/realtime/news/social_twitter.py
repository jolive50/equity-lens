"""Twitter/X sentiment ingestion scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(slots=True)
class TwitterStreamConfig:
    """Configuration for streaming Twitter/X financial chatter."""

    bearer_token: str
    queries: Iterable[str]
    destination: str = "data/realtime/landing/social/twitter/"
    backfill_minutes: Optional[int] = 15
    poll_interval_seconds: int = 30
    max_results: int = 100


def stream_twitter_posts(config: TwitterStreamConfig) -> None:
    """Connect to the X API or third-party stream and land posts for scoring."""

    raise NotImplementedError(
        "Implement filtered stream creation, reconnect handling, and sentiment feature extraction hooks."
    )
