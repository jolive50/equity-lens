"""Utility functions for agent initialization."""

import os
from langchain_core.runnables import Runnable


def build_openai_llm(model: str = "gpt-4o-mini") -> Runnable:
    """Build OpenAI LLM for agent use.

    Args:
        model: OpenAI model name (default: gpt-4o-mini)

    Returns:
        Runnable LLM instance

    Raises:
        ValueError: If OPENAI_API_KEY not found in environment
    """
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required but not found in environment")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0.1
    )
