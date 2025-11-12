#!/usr/bin/env python3
"""
Simple test script to verify streaming API functionality in rl4co/heuristic_finder/llm.py
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the module directly
from rl4co.heuristic_finder.llm import (
    generate_candidates_via_deepseek,
    generate_candidates_via_kimi,
    generate_candidates_via_glm,
    generate_candidates_via_ollama,
)

def test_streaming_api():
    """Test streaming functionality for different API providers."""
    
    # Test prompt
    test_prompt = "Generate a simple Python function that returns the sum of two numbers."
    
    print("Testing streaming API functionality...")
    print("=" * 60)
    
    # Test DeepSeek API (if API key is available)
    if os.environ.get("DEEPSEEK_API_KEY"):
        print("Testing DeepSeek API with streaming...")
        try:
            results = generate_candidates_via_deepseek(
                prompt=test_prompt,
                n=1,
                debug=True,
                stream=True
            )
            print(f"DeepSeek streaming test: {'SUCCESS' if results else 'FAILED'}")
            if results:
                print(f"Generated code length: {len(results[0])} characters")
        except Exception as e:
            print(f"DeepSeek streaming test FAILED: {e}")
    else:
        print("DeepSeek API key not found, skipping DeepSeek test")
    
    print("-" * 60)
    
    # Test Kimi API (if API key is available)
    if os.environ.get("KIMI_API_KEY"):
        print("Testing Kimi API with streaming...")
        try:
            results = generate_candidates_via_kimi(
                prompt=test_prompt,
                n=1,
                debug=True,
                stream=True
            )
            print(f"Kimi streaming test: {'SUCCESS' if results else 'FAILED'}")
            if results:
                print(f"Generated code length: {len(results[0])} characters")
        except Exception as e:
            print(f"Kimi streaming test FAILED: {e}")
    else:
        print("Kimi API key not found, skipping Kimi test")
    
    print("-" * 60)
    
    # Test GLM API (if API key is available)
    if os.environ.get("GLM_API_KEY"):
        print("Testing GLM API with streaming...")
        try:
            results = generate_candidates_via_glm(
                prompt=test_prompt,
                n=1,
                debug=True,
                stream=True
            )
            print(f"GLM streaming test: {'SUCCESS' if results else 'FAILED'}")
            if results:
                print(f"Generated code length: {len(results[0])} characters")
        except Exception as e:
            print(f"GLM streaming test FAILED: {e}")
    else:
        print("GLM API key not found, skipping GLM test")
    
    print("-" * 60)
    
    # Test Ollama (if available)
    print("Testing Ollama with streaming...")
    try:
        results = generate_candidates_via_ollama(
            model="qwen2.5-coder:7b",
            prompt=test_prompt,
            n=1,
            debug=True,
            stream=True
        )
        print(f"Ollama streaming test: {'SUCCESS' if results else 'FAILED'}")
        if results:
            print(f"Generated code length: {len(results[0])} characters")
    except Exception as e:
        print(f"Ollama streaming test FAILED: {e}")
    
    print("=" * 60)
    print("Streaming API tests completed!")

if __name__ == "__main__":
    test_streaming_api()
