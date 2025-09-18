"""Batch ingestion scaffolding for historical market datasets."""

from .alpha_vantage import AlphaVantageConfig, fetch_alpha_vantage_timeseries
from .iex_cloud import IEXCloudConfig, fetch_iex_cloud_eod
from .nasdaq_data_link import NasdaqDataLinkConfig, fetch_nasdaq_dataset
from .tiingo import TiingoConfig, fetch_tiingo_prices
from .yahoo_finance import YahooFinanceConfig, fetch_yahoo_prices

__all__ = [
    "AlphaVantageConfig",
    "IEXCloudConfig",
    "NasdaqDataLinkConfig",
    "TiingoConfig",
    "YahooFinanceConfig",
    "fetch_alpha_vantage_timeseries",
    "fetch_iex_cloud_eod",
    "fetch_nasdaq_dataset",
    "fetch_tiingo_prices",
    "fetch_yahoo_prices",
]
