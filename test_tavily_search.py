#!/usr/bin/env python3
"""
Test script for Tavily search implementation.
This script tests the updated Tavily search functionality.
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

from src.react_agent.tools import search

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables (for API keys)
load_dotenv()

async def test_tavily_search():
    """Test the basic search functionality."""
    try:
        query = "What is LangChain?"
        logger.info(f"Testing basic search with query: {query}")
        results = await search(query)
        logger.info(f"Search returned {len(results)} results")
        logger.info("Search test successful")
        return True
    except Exception as e:
        logger.error(f"Error in search test: {str(e)}")
        return False

async def test_search_with_domains():
    """Test the search functionality with domain filtering."""
    try:
        query = "What is LangChain?"
        logger.info(f"Testing search with domain filtering: {query}")

        # Test with domain filtering
        included_domains = "langchain.com"
        logger.info(f"Testing search with included domains: {included_domains}")
        filtered_results = await search(query, include_domains=included_domains)
        logger.info(f"Search with domain filtering returned {len(filtered_results)} results")
        logger.info("Search with domain filtering successful")

        # Test with domain exclusion
        excluded_domains = "wikipedia.org"
        logger.info(f"Testing search with excluded domains: {excluded_domains}")
        filtered_results_exclusion = await search(query, exclude_domains=excluded_domains)
        logger.info(f"Search with domain exclusion returned {len(filtered_results_exclusion)} results")
        logger.info("Search with domain exclusion successful")

        return True
    except Exception as e:
        logger.error(f"Error in search with domains test: {str(e)}")
        return False

async def main():
    """Run all tests."""
    logger.info("Starting Tavily search tests")

    # Check if TAVILY_API_KEY is set
    if not os.getenv("TAVILY_API_KEY"):
        logger.error("TAVILY_API_KEY environment variable not set. Please set it before running tests.")
        return

    search_test = await test_tavily_search()
    domain_search_test = await test_search_with_domains()

    if search_test and domain_search_test:
        logger.info("All tests passed!")
    else:
        logger.error("Some tests failed")

if __name__ == "__main__":
    asyncio.run(main())