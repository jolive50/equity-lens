"""Realtime news ingestion scaffolding."""

from .finnhub import FinnhubNewsConfig, stream_finnhub_news
from .newsapi import NewsAPIConfig, poll_newsapi_headlines
from .social_reddit import RedditStreamConfig, stream_reddit_posts
from .social_twitter import TwitterStreamConfig, stream_twitter_posts
from .yahoo_finance import YahooNewsConfig, poll_yahoo_news

__all__ = [
    "FinnhubNewsConfig",
    "NewsAPIConfig",
    "RedditStreamConfig",
    "TwitterStreamConfig",
    "YahooNewsConfig",
    "poll_newsapi_headlines",
    "poll_yahoo_news",
    "stream_finnhub_news",
    "stream_reddit_posts",
    "stream_twitter_posts",
]
