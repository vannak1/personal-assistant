"""This module provides example tools for web scraping and search functionality.

It includes tools for web search and can be extended with additional functionality.

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List

# Import tools from the tools package
from react_agent.tools import web_search

# --- Existing Tools ---
# List of all available tools
TOOLS: List[Callable[..., Any]] = [web_search]  # Add any other existing tools here