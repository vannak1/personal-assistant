#!/usr/bin/env python3
"""Script to verify the setup and configuration of the multi-agent architecture."""

import os
from dotenv import load_dotenv
import importlib
import inspect
import sys

# Load environment variables
load_dotenv()

def check_env_variables():
    """Check if necessary environment variables are set."""
    print("Checking environment variables...")
    
    # Check DeepSeek API key
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        print("❌ DEEPSEEK_API_KEY is not set in the .env file")
        status_deepseek = False
    else:
        print("✅ DEEPSEEK_API_KEY is properly set")
        status_deepseek = True
    
    # Check OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("⚠️ OPENAI_API_KEY is not set in the .env file (needed for PA and specialist agents)")
        status_openai = False
    else:
        print("✅ OPENAI_API_KEY is properly set")
        status_openai = True
    
    return status_deepseek and status_openai

def check_dependencies():
    """Check if required dependencies are installed."""
    
    print("\nChecking required dependencies...")
    
    all_ok = True
    
    try:
        import langchain_community
        print("✅ langchain-community is properly installed")
    except ImportError:
        print("❌ langchain-community is not installed. Run 'pip install langchain-community'")
        all_ok = False
    
    try:
        import langgraph
        print("✅ langgraph is properly installed")
    except ImportError:
        print("❌ langgraph is not installed. Run 'pip install langgraph'")
        all_ok = False
    
    try:
        from langchain_core.messages import AIMessage
        print("✅ langchain-core is properly installed")
    except ImportError:
        print("❌ langchain-core is not installed. Run 'pip install langchain-core'")
        all_ok = False
    
    return all_ok

def check_architecture():
    """Check if the new architecture with supervisor and PA agents is properly implemented."""
    
    print("\nVerifying architecture implementation...")
    
    all_ok = True
    
    # Check if important modules can be imported
    try:
        # Import configuration
        from react_agent.configuration import Configuration
        print("✅ Configuration class successfully imported")
        
        # Check for agent-specific models in configuration
        if hasattr(Configuration, 'supervisor_model') and hasattr(Configuration, 'personal_assistant_model'):
            print("✅ Model-specific configurations found")
        else:
            print("❌ Model-specific configurations not found in Configuration class")
            all_ok = False
        
        # Import state
        from react_agent.state import State
        print("✅ State class successfully imported")
        
        # Check for new state fields
        state_fields = [attr for attr in dir(State) if not attr.startswith('_')]
        required_fields = ['routing_reason', 'specialist_results', 'first_message', 'session_state']
        missing_fields = [field for field in required_fields if field not in state_fields]
        
        if not missing_fields:
            print("✅ All required state fields found")
        else:
            print(f"❌ Missing state fields: {', '.join(missing_fields)}")
            all_ok = False
        
        # Import graph
        from react_agent.graph import supervisor_agent, personal_assistant_agent
        print("✅ Agent functions successfully imported")
        
        # Check for separate supervisor and PA functions
        if 'supervisor_agent' in locals() and 'personal_assistant_agent' in locals():
            print("✅ Separate supervisor and PA agent functions found")
        else:
            print("❌ Separate supervisor and PA agent functions not found")
            all_ok = False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        all_ok = False
    except Exception as e:
        print(f"❌ Error: {e}")
        all_ok = False
    
    return all_ok

def check_prompts():
    """Check if the prompts are updated for the new architecture."""
    
    print("\nVerifying prompt updates...")
    
    all_ok = True
    
    try:
        from react_agent import prompts
        
        # Check for SUPERVISOR_PROMPT content
        if hasattr(prompts, 'SUPERVISOR_PROMPT') and "routing" in prompts.SUPERVISOR_PROMPT.lower():
            print("✅ SUPERVISOR_PROMPT found and contains routing information")
        else:
            print("❌ SUPERVISOR_PROMPT not found or doesn't mention routing")
            all_ok = False
        
        # Check for updated PERSONAL_ASSISTANT_PROMPT content
        if hasattr(prompts, 'PERSONAL_ASSISTANT_PROMPT') and "interface" in prompts.PERSONAL_ASSISTANT_PROMPT.lower():
            print("✅ PERSONAL_ASSISTANT_PROMPT found and contains interface role")
        else:
            print("❌ PERSONAL_ASSISTANT_PROMPT not updated for new role")
            all_ok = False
        
    except ImportError:
        print("❌ Could not import prompts module")
        all_ok = False
    
    return all_ok

def main():
    """Run all verification checks."""
    print("Verifying multi-agent architecture setup...")
    
    env_check = check_env_variables()
    dep_check = check_dependencies()
    arch_check = check_architecture()
    prompt_check = check_prompts()
    
    if env_check and dep_check and arch_check and prompt_check:
        print("\n✨ All checks passed! Your multi-agent architecture is correctly configured.")
        print("\nTo use it:")
        print("1. Make sure to add your OpenAI API key to the .env file")
        print("2. Run the assistant with: python -m langgraph-cli serve -d src/react_agent")
    else:
        print("\n⚠️ Some checks failed. Please fix the issues above before running the assistant.")
        
        # Give specific advice based on what failed
        if not env_check:
            print("\nEnvironment Variable Tips:")
            print("- Make sure your .env file contains both DEEPSEEK_API_KEY and OPENAI_API_KEY")
            print("- Verify that the API keys are valid")
        
        if not arch_check or not prompt_check:
            print("\nCode Implementation Tips:")
            print("- Ensure all architectural changes are properly implemented")
            print("- Verify that the supervisor and personal assistant agents are correctly separated")
            print("- Check that routing logic is properly implemented")

if __name__ == "__main__":
    main()