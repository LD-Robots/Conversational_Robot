# src/tools/__init__.py
"""Tools module for LLM function calling."""

from src.tools.websearch import web_search, WEB_SEARCH_TOOL_DEFINITION

__all__ = ["web_search", "WEB_SEARCH_TOOL_DEFINITION"]
