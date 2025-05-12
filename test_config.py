#!/usr/bin/env python3
"""Test script for verifying configuration."""

import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_configuration():
    """Test that configuration file contains web_search_model."""
    logger.info("Testing configuration file...")
    
    try:
        # Read the configuration file
        with open("src/react_agent/configuration.py", "r") as f:
            content = f.read()
        
        # Check if web_search_model is defined
        if "web_search_model" in content:
            logger.info("web_search_model found in Configuration class")
            logger.info("Configuration test: SUCCESS")
        else:
            logger.error("web_search_model not found in Configuration class")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Configuration test: FAILED - {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_configuration()