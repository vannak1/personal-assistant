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


# --- Simulated Session Store ---
# In a real app, this would interact with browser cookies, backend sessions, etc.
_simulated_session_store = {}


class SessionInfo(TypedDict):
    user_name: Optional[str]
    user_uid: Optional[str]


@tool
def manage_user_session(user_name_to_set: Optional[str] = None) -> SessionInfo:
    \"\"\"Manages user session information. 
    If user_name_to_set is provided, it creates or updates the session with that name and returns the session info (including a generated UID).
    If user_name_to_set is None, it attempts to retrieve existing session info.
    Returns the user's name and a unique ID if a session exists or is created, otherwise returns None for both.\"\"\"
    global _simulated_session_store

    session_key = "default_user"  # Simulate a single session for this example

    if user_name_to_set:
        # Create or update session
        if session_key not in _simulated_session_store or not _simulated_session_store[session_key].get("user_uid"):
            _simulated_session_store[session_key] = {
                "user_name": user_name_to_set,
                "user_uid": str(uuid.uuid4())  # Generate a unique ID
            }
        else:
            # Update name if UID already exists
            _simulated_session_store[session_key]["user_name"] = user_name_to_set
        print(f"[Session Tool] Session created/updated for {user_name_to_set}. UID: {_simulated_session_store[session_key]['user_uid']}")
        return {
            "user_name": _simulated_session_store[session_key]["user_name"],
            "user_uid": _simulated_session_store[session_key]["user_uid"]
        }
    else:
        # Check for existing session
        if session_key in _simulated_session_store:
            print(f"[Session Tool] Existing session found for {_simulated_session_store[session_key]['user_name']}")
            return {
                "user_name": _simulated_session_store[session_key].get("user_name"),
                "user_uid": _simulated_session_store[session_key].get("user_uid")
            }
        else:
            print("[Session Tool] No existing session found.")
            return {"user_name": None, "user_uid": None}


# --- Existing Tools ---
# Add the new tool to the list
TOOLS: List[Callable[..., Any]] = [manage_user_session]  # Add any other existing tools back here, e.g., TOOLS = [manage_user_session, existing_tool_1, existing_tool_2]

# Example of adding to existing tools (replace with your actual existing tools):
# from existing_tool_module import search_tool, another_tool 
# TOOLS = [manage_user_session, search_tool, another_tool]

# If TOOLS was previously defined, ensure manage_user_session is added:
# try:
#    TOOLS.append(manage_user_session)
# except NameError:
#    TOOLS = [manage_user_session]
