"""This module provides tools for web scraping and search functionality.

It includes enhanced Tavily search functions with domain filtering and formatted results.

These tools are designed to support the web search agent and other components
that require web search capabilities.
"""

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, TypedDict, cast

from langchain_core.tools import tool
from langchain_tavily import TavilySearch  # type: ignore[import-not-found]

from react_agent.configuration import Configuration

logger = logging.getLogger(__name__)

class SearchResult:
    """A single search result from Tavily."""

    def __init__(self, url: str, title: str, content: str, score: float = 0.0):
        self.url = url
        self.title = title
        self.content = content
        self.score = score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "score": self.score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchResult":
        """Create a SearchResult from a dictionary."""
        return cls(
            url=data.get("url", ""),
            title=data.get("title", ""),
            content=data.get("content", ""),
            score=data.get("score", 0.0)
        )


@tool
async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.

    Args:
        query: The search query to execute

    Returns:
        Raw search results from Tavily
    """
    configuration = Configuration.from_context()
    wrapped = TavilySearch(max_results=configuration.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


@tool
async def web_search(query: str, include_domains: Optional[str] = None,
                   exclude_domains: Optional[str] = None) -> str:
    """
    Search the web for information on a specific query with domain filtering options.

    Args:
        query: The search query to use
        include_domains: Optional comma-separated list of domains to limit search results to
        exclude_domains: Optional comma-separated list of domains to exclude from search results

    Returns:
        Formatted search results as text
    """
    try:
        # Process domain lists if provided
        include_list = include_domains.split(",") if include_domains else None
        exclude_list = exclude_domains.split(",") if exclude_domains else None

        # Prepare search parameters
        configuration = Configuration.from_context()
        search_client = TavilySearch(max_results=configuration.max_search_results)

        # Construct search arguments
        search_kwargs = {"query": query}
        if include_list:
            search_kwargs["include_domains"] = include_list
        if exclude_list:
            search_kwargs["exclude_domains"] = exclude_list

        # Execute search
        results = await search_client.ainvoke(search_kwargs)

        # Format results
        if not results:
            return "No search results found for your query."

        formatted_results = f"Search results for: {query}\n\n"

        for i, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", "")

            formatted_results += f"{i}. {title}\n"
            formatted_results += f"   URL: {url}\n"
            formatted_results += f"   {content[:200]}...\n\n"

        return formatted_results

    except Exception as e:
        logger.error(f"Error during web search: {str(e)}")
        return f"Error performing search: {str(e)}"


# --- Tools Listing ---
TOOLS: List[Callable[..., Any]] = [search, web_search]