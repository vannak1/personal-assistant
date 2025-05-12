"""This module provides tools for web scraping and search functionality.

It includes enhanced Tavily search functions for web search capabilities.

These tools are designed to support the web search agent and other components
that require web search capabilities.
"""

import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, TypedDict, cast

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

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
async def search(query: str, include_domains: Optional[str] = None,
               exclude_domains: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """Search the web for information.

    This function performs a web search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results from the internet. It's particularly
    useful for answering questions about current events, facts, and finding relevant web content.

    Args:
        query: The search query to execute
        include_domains: Optional comma-separated list of domains to limit search results to
        exclude_domains: Optional comma-separated list of domains to exclude from search results

    Returns:
        A list of search result objects, each containing title, content, and URL
    """
    print(f"🔍 Executing search for: '{query}'")
    configuration = Configuration.from_context()

    # Process domain lists if provided
    include_list = include_domains.split(",") if include_domains else None
    exclude_list = exclude_domains.split(",") if exclude_domains else None

    # Initialize the search client with configuration
    search_client = TavilySearch(
        max_results=configuration.max_search_results,
        topic="general",
    )

    # Construct search arguments
    search_kwargs = {"query": query}
    if include_list:
        search_kwargs["include_domains"] = include_list
    if exclude_list:
        search_kwargs["exclude_domains"] = exclude_list

    try:
        # Get raw results from Tavily
        raw_results = await search_client.ainvoke(search_kwargs)

        # Format the results into a consistent structure
        formatted_results = []
        if raw_results and "results" in raw_results:
            formatted_results = [
                {
                    "title": item.get("title", "Untitled"),
                    "url": item.get("url", ""),
                    "content": item.get("content", "No content available")
                }
                for item in raw_results["results"]
            ]

        print(f"✓ Search completed. Found {len(formatted_results)} results.")
        return formatted_results
    except Exception as e:
        print(f"✗ Search error: {str(e)}")
        # Return a formatted error message that will be usable by the web search agent
        return [{"title": "Search Error", "url": "", "content": f"Error performing search: {str(e)}"}]


# --- Tools Listing ---
TOOLS: List[Callable[..., Any]] = [search]