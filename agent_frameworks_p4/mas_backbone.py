"""MAS LLM backbone selection for P4/P10 (Claude-4 vs GPT-4o)."""

from __future__ import annotations

import os
from typing import Any


def backbone() -> str:
    return os.getenv("MAS_BACKBONE", "claude-4").strip().lower()


def is_gpt4o() -> bool:
    return backbone() in ("gpt-4o", "gpt4o", "openai")


def mas_source_for_dataset(dataset_prefix: str) -> str:
    """Source tag for results.tex / evaluation (MAS-GPT4o or MAS-Claude4)."""
    tag = "MAS-GPT4o" if is_gpt4o() else "MAS-Claude4"
    return tag


def autogen_model_client() -> Any:
    if is_gpt4o():
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        return OpenAIChatCompletionClient(
            model="gpt-4o",
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    from autogen_ext.models.anthropic import AnthropicChatCompletionClient

    return AnthropicChatCompletionClient(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )


def crewai_llm() -> Any:
    from crewai import LLM

    if is_gpt4o():
        return LLM(
            model=os.getenv("CREWAI_OPENAI_MODEL", "openai/gpt-4o"),
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    return LLM(
        model=os.getenv("CREWAI_ANTHROPIC_MODEL", "anthropic/claude-sonnet-4-20250514"),
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )


def langgraph_chat_model():
    from langchain_core.messages import HumanMessage  # noqa: F401

    if is_gpt4o():
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        temperature=0,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
