"""Web search tool implementation using Tavily."""

import logging
from typing import Dict, Any, List, Optional

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

class WebSearchTool:
    """A tool for searching the web using Tavily."""
    
    def __init__(self, max_results: int = 5):
        """Initialize the web search tool."""
        self.max_results = max_results
        self._search_client = TavilySearch(max_results=max_results)
        
    async def search(self, query: str, include_domains: Optional[List[str]] = None, 
                    exclude_domains: Optional[List[str]] = None) -> List[SearchResult]:
        """
        Search the web for the given query.
        
        Args:
            query: The search query
            include_domains: Optional list of domains to include in search results
            exclude_domains: Optional list of domains to exclude from search results
            
        Returns:
            List of search results
        """
        try:
            search_kwargs = {"query": query}
            
            # Add domain filtering if specified
            if include_domains:
                search_kwargs["include_domains"] = include_domains
            if exclude_domains:
                search_kwargs["exclude_domains"] = exclude_domains
                
            # Execute search
            results = await self._search_client.ainvoke(search_kwargs)
            
            # Process results
            search_results = []
            for result in results:
                search_result = SearchResult(
                    url=result.get("url", ""),
                    title=result.get("title", ""),
                    content=result.get("content", ""),
                    score=result.get("score", 0.0)
                )
                search_results.append(search_result)
                
            return search_results
            
        except Exception as e:
            logger.error(f"Error during web search: {str(e)}")
            return []

# Create the tool function for LangChain
@tool
async def web_search(query: str, include_domains: Optional[str] = None, 
                    exclude_domains: Optional[str] = None) -> str:
    """
    Search the web for information on a specific query.
    
    Args:
        query: The search query to use
        include_domains: Optional comma-separated list of domains to limit search results to
        exclude_domains: Optional comma-separated list of domains to exclude from search results
        
    Returns:
        Formatted search results as text
    """
    # Process domain lists if provided
    include_list = include_domains.split(",") if include_domains else None
    exclude_list = exclude_domains.split(",") if exclude_domains else None
    
    # Initialize the search tool
    configuration = Configuration.from_context()
    search_tool = WebSearchTool(max_results=configuration.max_search_results)
    
    # Execute search
    results = await search_tool.search(query, include_list, exclude_list)
    
    # Format results
    if not results:
        return "No search results found for your query."
    
    formatted_results = f"Search results for: {query}\n\n"
    
    for i, result in enumerate(results, 1):
        formatted_results += f"{i}. {result.title}\n"
        formatted_results += f"   URL: {result.url}\n"
        formatted_results += f"   {result.content[:200]}...\n\n"
    
    return formatted_results