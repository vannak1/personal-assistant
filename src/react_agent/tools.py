# tools.py
"""This module provides tools for web scraping, search, and basic session management.

It includes enhanced Tavily search functions and a simulated user session tool.

These tools are designed to support the multi-agent system components.
"""

import logging
import uuid
import json # Make sure json is imported
from typing import Any, Callable, Dict, List, Optional, TypedDict, cast

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

# Assuming Configuration is defined elsewhere and provides context if needed
# from react_agent.configuration import Configuration

logger = logging.getLogger(__name__)

# --- In-Memory Session Simulation ---
# WARNING: This is for demonstration only and will reset every time the script runs.
# In a real app, use a database, external cache, or proper session backend.
_simulated_session_store: Dict[str, Any] = {}
_SESSION_KEY = "current_user" # Use a fixed key for simplicity

# --- Tool Definitions ---

class SearchResult:
    """A single search result from Tavily."""
    # ... (keep existing SearchResult class as is) ...
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
               exclude_domains: Optional[str] = None) -> str:
    """Search the web for information using Tavily.

    Args:
        query: The search query to execute.
        include_domains: Optional comma-separated list of domains to focus search on.
        exclude_domains: Optional comma-separated list of domains to omit from search.

    Returns:
        A JSON string containing a list of search results or an error message.
    """
    print(f"🔍 Executing search for: '{query}'")
    # configuration = Configuration.from_context() # Uncomment if config needed

    include_list = include_domains.split(",") if include_domains else None
    exclude_list = exclude_domains.split(",") if exclude_domains else None

    try:
        # Assuming TavilySearch is configured globally or doesn't need specific config here
        search_client = TavilySearch(
            # max_results=configuration.max_search_results, # Example if using config
            max_results=5, # Default if config not used
            topic="general",
        )

        search_kwargs = {"query": query}
        if include_list:
            search_kwargs["include_domains"] = include_list
        if exclude_list:
            search_kwargs["exclude_domains"] = exclude_list

        raw_results = await search_client.ainvoke(search_kwargs) # Corrected: use search_kwargs

        formatted_results = []
        # Tavily structure adjustment: results are directly in the list
        if isinstance(raw_results, list):
            formatted_results = [
                {
                    "title": item.get("title", "Untitled"),
                    "url": item.get("url", ""),
                    "content": item.get("content", "No content available")
                }
                for item in raw_results # Iterate directly over the list
            ]
        # Handle potential dict response structure if Tavily changes or for other tools
        elif isinstance(raw_results, dict) and "results" in raw_results:
             formatted_results = [
                {
                    "title": item.get("title", "Untitled"),
                    "url": item.get("url", ""),
                    "content": item.get("content", "No content available")
                }
                for item in raw_results["results"]
            ]


        print(f"✓ Search completed. Found {len(formatted_results)} results.")
        return json.dumps(formatted_results)

    except Exception as e:
        print(f"✗ Search error: {str(e)}")
        error_result = [{"title": "Search Error", "url": "", "content": f"Error performing search: {str(e)}"}]
        return json.dumps(error_result)

@tool
def manage_user_session(user_name_to_set: Optional[str] = None) -> str:
    """
    Manages a simulated user session.
    If 'user_name_to_set' is provided, it saves/updates the user's name and generates a UID.
    If 'user_name_to_set' is not provided, it checks for an existing user session.

    Args:
        user_name_to_set: The name to save for the current user session.

    Returns:
        A JSON string containing the user session details ('user_name', 'user_uid')
        or null values if no session exists or couldn't be created.
    """
    global _simulated_session_store # Access the global store

    if user_name_to_set:
        # --- Save/Update Session ---
        print(f"🔧 Managing session: Setting user name to '{user_name_to_set}'")
        user_uid = _simulated_session_store.get(_SESSION_KEY, {}).get("user_uid", str(uuid.uuid4())) # Keep UID if exists, else generate
        _simulated_session_store[_SESSION_KEY] = {
            "user_name": user_name_to_set,
            "user_uid": user_uid
        }
        print(f"✓ Session updated: Name='{user_name_to_set}', UID='{user_uid}'")
        return json.dumps({
            "user_name": user_name_to_set,
            "user_uid": user_uid
        })
    else:
        # --- Check Session ---
        print("🔧 Managing session: Checking for existing user...")
        existing_session = _simulated_session_store.get(_SESSION_KEY)
        if existing_session and isinstance(existing_session, dict):
            user_name = existing_session.get("user_name")
            user_uid = existing_session.get("user_uid")
            if user_name and user_uid:
                print(f"✓ Existing session found: Name='{user_name}', UID='{user_uid}'")
                return json.dumps({
                    "user_name": user_name,
                    "user_uid": user_uid
                })

        # --- No Session Found ---
        print("✗ No active session found.")
        return json.dumps({
            "user_name": None,
            "user_uid": None
        })

# --- Tools Listing ---
# Add the new tool to the list
TOOLS: List[Callable[..., Any]] = [search, manage_user_session]