#!/usr/bin/env python3
"""Test script for DeepSeek integration."""

import os
from dotenv import load_dotenv
from src.react_agent.utils import load_chat_model

# Load environment variables from .env file
load_dotenv()

def test_deepseek():
    """Test the DeepSeek integration."""
    print("Testing DeepSeek integration...")
    
    # Check if the API key is set
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY environment variable not set")
        return False
    
    print("API key found, loading model...")
    
    try:
        # Load the DeepSeek model
        model = load_chat_model("deepseek/deepseek-chat")
        
        # Test the model with a simple query
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello and introduce yourself in one sentence."}
        ]
        
        print("Sending request to DeepSeek API...")
        response = model.invoke(messages)
        
        print("\nResponse from DeepSeek:")
        print(f"{response.content}")
        
        return True
    except Exception as e:
        print(f"ERROR: Failed to test DeepSeek integration: {e}")
        return False

if __name__ == "__main__":
    success = test_deepseek()
    print("\nTest", "succeeded" if success else "failed")