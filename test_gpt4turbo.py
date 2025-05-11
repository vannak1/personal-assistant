#!/usr/bin/env python3
"""Test script for GPT-4 Turbo integration."""

import os
from dotenv import load_dotenv
from src.react_agent.utils import load_chat_model

# Load environment variables from .env file
load_dotenv()

def test_gpt4turbo():
    """Test the GPT-4 Turbo integration."""
    print("Testing GPT-4 Turbo integration...")
    
    # Check if the API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return False
    
    print("API key found, loading model...")
    
    try:
        # Load the GPT-4 Turbo model
        model = load_chat_model("openai/gpt-4-turbo")
        
        # Test the model with a simple query
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello and introduce yourself in one sentence."}
        ]
        
        print("Sending request to OpenAI API...")
        response = model.invoke(messages)
        
        print("\nResponse from GPT-4 Turbo:")
        print(f"{response.content}")
        
        return True
    except Exception as e:
        print(f"ERROR: Failed to test GPT-4 Turbo integration: {e}")
        return False

if __name__ == "__main__":
    success = test_gpt4turbo()
    print("\nTest", "succeeded" if success else "failed")