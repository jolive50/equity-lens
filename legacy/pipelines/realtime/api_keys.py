"""Centralised helpers for working with third-party API keys.

This module keeps a single source of truth for which environment variables
map to each upstream service (Alpha Vantage, Tiingo, Finnhub, NewsAPI, OpenAI).
By funnelling all lookups through these functions, we guarantee that the rest
of the codebase never attempts to call an API we don't actually have credentials for.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, Optional

# Map friendly service names to their expected environment variable.
_API_ENV_MAP: Dict[str, str] = {
    "alpha_vantage": "ALPHA_VANTAGE_API_KEY",
    "tiingo": "TIINGO_API_KEY",
    "finnhub": "FINNHUB_API_KEY",
    "newsapi": "NEWSAPI_KEY",
    "openai": "OPENAI_API_KEY",
}


def get_api_key(service: str, *, required: bool = False) -> Optional[str]:
    """Fetch the API key for a given service if it exists.

    What: Returns the configured key for `service`, or None when missing.
    Why: Centralises validation so calling code doesn't sprinkle os.getenv checks.
    How: Look up the environment variable name in `_API_ENV_MAP` and read it.
    Data: Reads process environment variables; never stores or logs the key itself.
    """
    env_var = _API_ENV_MAP.get(service)
    if env_var is None:
        raise KeyError(f"Unknown service '{service}' - add it to _API_ENV_MAP first")

    value = os.getenv(env_var)
    if value:
        return value.strip()

    if required:
        raise ValueError(f"{env_var} environment variable is required for {service}")
    return None


def get_available_api_keys(*services: str) -> Dict[str, str]:
    """Return a dictionary of services that currently have keys configured.

    What: Builds a filtered dictionary of {service: key} pairs.
    Why: Allows callers to dynamically enable only the integrations that are ready to use.
    How: Iterate over the requested service names (or all known services) and reuse `get_api_key`.
    Data: Keys are returned exactly as stored in the environment with surrounding whitespace stripped.
    """
    names: Iterable[str] = services or _API_ENV_MAP.keys()
    available: Dict[str, str] = {}
    for service in names:
        try:
            key = get_api_key(service)
        except KeyError:
            # Ignore unknown entries so callers can request custom names safely.
            continue
        if key:
            available[service] = key
    return available


def missing_api_keys(services: Iterable[str]) -> Dict[str, str]:
    """Return a mapping of services to the environment variable they require when missing."""
    missing: Dict[str, str] = {}
    for service in services:
        env_var = _API_ENV_MAP.get(service)
        if not env_var:
            continue
        if not os.getenv(env_var):
            missing[service] = env_var
    return missing
