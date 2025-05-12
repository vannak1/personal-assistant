#!/usr/bin/env python3
"""Test script for verifying tool registration and execution."""

import asyncio
import logging
from dotenv import load_dotenv

from src.react_agent.tools import search_tool, manage_user_session_tool

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def test_tools_registration():
    """Test that both tools are properly registered and can be executed."""
    logger.info("Testing tools registration and execution...")

    # Test search tool
    logger.info("Testing search tool...")
    try:
        query = "What is LangGraph?"
        results = await search_tool.ainvoke(query)
        logger.info(f"Search tool test: SUCCESS")
    except Exception as e:
        logger.error(f"Search tool test: FAILED - {str(e)}")

    # Test manage_user_session tool
    logger.info("Testing manage_user_session tool...")
    try:
        # Test without a user name (should return empty session)
        session_check = manage_user_session_tool.invoke({})
        logger.info("Session check test: SUCCESS")

        # Test with a user name (should create a session)
        test_user = "TestUser"
        session_create = manage_user_session_tool.invoke({"user_name_to_set": test_user})
        logger.info(f"Session creation test: SUCCESS")
    except Exception as e:
        logger.error(f"Session management test: FAILED - {str(e)}")

async def main():
    """Run all tests."""
    logger.info("Starting tool registration and execution tests")
    await test_tools_registration()
    logger.info("All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())