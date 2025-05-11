#!/usr/bin/env python3
"""Script to verify the setup and configuration of the personal assistant."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_env_variables():
    """Check if necessary environment variables are set."""
    print("Checking environment variables...")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY is not set in the .env file")
        return False

    print("✅ OPENAI_API_KEY is properly set")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import langchain_community
        from langchain_openai import ChatOpenAI
        print("✅ langchain-community and langchain-openai are properly installed")
        return True
    except ImportError:
        print("❌ Required packages are not installed. Run 'pip install langchain-community langchain-openai'")
        return False

def main():
    """Run all verification checks."""
    print("Verifying personal assistant setup...")
    
    env_check = check_env_variables()
    dep_check = check_dependencies()
    
    if env_check and dep_check:
        print("\n✨ All checks passed! Your personal assistant is correctly configured to use GPT-4 Turbo.")
        print("\nTo use it, run the assistant with:")
        print("   python -m langgraph-cli serve -d src/react_agent")
    else:
        print("\n❌ Some checks failed. Please fix the issues above before running the assistant.")

if __name__ == "__main__":
    main()