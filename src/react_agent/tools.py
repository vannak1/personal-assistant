"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

import uuid
from typing import Any, Callable, List, Optional, TypedDict, cast

from langchain_core.tools import tool
from langchain_tavily import TavilySearch  # type: ignore[import-not-found]

from react_agent.configuration import Configuration


async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_context()
    wrapped = TavilySearch(max_results=configuration.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))

# --- Existing Tools ---
# Add the new tool to the list
TOOLS: List[Callable[..., Any]] = [search]  # Add any other existing tools back here, e.g., TOOLS = [manage_user_session, existing_tool_1, existing_tool_2]

# Example of adding to existing tools (replace with your actual existing tools):
# from existing_tool_module import search_tool, another_tool 
# TOOLS = [manage_user_session, search_tool, another_tool]