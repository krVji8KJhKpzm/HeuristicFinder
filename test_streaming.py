#!/usr/bin/env python3
"""
Test script to verify streaming API functionality in rl4co/heuristic_finder/llm.py
"""

import os
import sys
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

def test_non_streaming_comparison():
    """Compare streaming vs non-streaming results."""
    
    test_prompt = "Generate a simple Python function that returns the sum of two numbers."
    
    print("\nTesting streaming vs non-streaming comparison...")
    print("=" * 60)
    
    # Test with DeepSeek if available
    if os.environ.get("DEEPSEEK_API_KEY"):
        print("Comparing DeepSeek streaming vs non-streaming...")
        
        # Non-streaming
        try:
            results_non_stream = generate_candidates_via_deepseek(
                prompt=test_prompt,
                n=1,
                debug=False,
                stream=False
            )
            print(f"Non-streaming result length: {len(results_non_stream[0]) if results_non_stream else 0}")
        except Exception as e:
            print(f"Non-streaming test failed: {e}")
            results_non_stream = []
        
        # Streaming
        try:
            results_stream = generate_candidates_via_deepseek(
                prompt=test_prompt,
                n=1,
                debug=False,
                stream=True
            )
            print(f"Streaming result length: {len(results_stream[0]) if results_stream else 0}")
        except Exception as e:
            print(f"Streaming test failed: {e}")
            results_stream = []
        
        # Compare results
        if results_non_stream and results_stream:
            print(f"Results match: {results_non_stream[0] == results_stream[0]}")
    else:
        print("DeepSeek API key not found, skipping comparison test")

if __name__ == "__main__":
    test_streaming_api()
    test_non_streaming_comparison()
