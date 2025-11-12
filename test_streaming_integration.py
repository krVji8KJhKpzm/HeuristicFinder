#!/usr/bin/env python3
"""
Integration test script to verify streaming API functionality with environment variables
"""

import os
import sys

# Add the parent directory to Python path to import rl4co modules
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rl4co'))

import heuristic_finder.llm as llm

def test_streaming_with_env_vars():
    """Test streaming functionality with environment variables."""
    
    print("Testing streaming API with environment variables...")
    print("=" * 60)
    
    # Test 1: Test with Kimi provider and streaming enabled
    print("Test 1: Kimi provider with streaming enabled")
    os.environ["LLM_API_PROVIDER"] = "kimi"
    os.environ["KIMI_STREAM"] = "true"
    
    try:
        # This should use streaming=True for Kimi
        results = llm.generate_candidates_via_kimi(
            prompt="Generate a simple Python function that returns the sum of two numbers.",
            n=1,
            debug=True,
            stream=True
        )
        print(f"Kimi streaming test: {'SUCCESS' if results else 'FAILED'}")
        if results:
            print(f"Generated code length: {len(results[0])} characters")
    except Exception as e:
        print(f"Kimi streaming test FAILED: {e}")
    
    print("-" * 60)
    
    # Test 2: Test with DeepSeek provider and streaming disabled
    print("Test 2: DeepSeek provider with streaming disabled")
    os.environ["LLM_API_PROVIDER"] = "deepseek"
    os.environ["DEEPSEEK_STREAM"] = "false"
    
    try:
        # This should use streaming=False for DeepSeek
        results = llm.generate_candidates_via_deepseek(
            prompt="Generate a simple Python function that returns the sum of two numbers.",
            n=1,
            debug=True,
            stream=False
        )
        print(f"DeepSeek non-streaming test: {'SUCCESS' if results else 'FAILED'}")
        if results:
            print(f"Generated code length: {len(results[0])} characters")
    except Exception as e:
        print(f"DeepSeek non-streaming test FAILED: {e}")
    
    print("-" * 60)
    
    # Test 3: Test environment variable parsing
    print("Test 3: Environment variable parsing")
    
    # Test various true values
    true_values = ["true", "1", "yes", "True", "YES"]
    for val in true_values:
        os.environ["KIMI_STREAM"] = val
        provider = os.environ.get("LLM_API_PROVIDER", "deepseek").lower()
        if provider == "kimi":
            stream = os.environ.get("KIMI_STREAM", "false").lower() in ("true", "1", "yes")
        elif provider == "deepseek":
            stream = os.environ.get("DEEPSEEK_STREAM", "false").lower() in ("true", "1", "yes")
        elif provider == "glm":
            stream = os.environ.get("GLM_STREAM", "false").lower() in ("true", "1", "yes")
        print(f"KIMI_STREAM='{val}' -> stream={stream}")
    
    # Test various false values
    false_values = ["false", "0", "no", "False", "NO", ""]
    for val in false_values:
        os.environ["KIMI_STREAM"] = val
        provider = os.environ.get("LLM_API_PROVIDER", "deepseek").lower()
        if provider == "kimi":
            stream = os.environ.get("KIMI_STREAM", "false").lower() in ("true", "1", "yes")
        elif provider == "deepseek":
            stream = os.environ.get("DEEPSEEK_STREAM", "false").lower() in ("true", "1", "yes")
        elif provider == "glm":
            stream = os.environ.get("GLM_STREAM", "false").lower() in ("true", "1", "yes")
        print(f"KIMI_STREAM='{val}' -> stream={stream}")
    
    print("=" * 60)
    print("Environment variable integration tests completed!")

def test_eoh_functions_with_streaming():
    """Test EoH functions with streaming parameter."""
    
    print("\nTesting EoH functions with streaming parameter...")
    print("=" * 60)
    
    # Test eoh_llm_i1 with streaming
    print("Testing eoh_llm_i1 with streaming...")
    try:
        results = llm.eoh_llm_i1(
            model=None,
            n=1,
            env_name="tsp",
            debug=True,
            stream=True
        )
        print(f"eoh_llm_i1 streaming test: {'SUCCESS' if results else 'FAILED'}")
        if results:
            print(f"Generated code length: {len(results[0])} characters")
    except Exception as e:
        print(f"eoh_llm_i1 streaming test FAILED: {e}")
    
    print("-" * 60)
    
    # Test eoh_llm_e1 with streaming
    print("Testing eoh_llm_e1 with streaming...")
    try:
        test_parents = [{"algorithm": "test algorithm", "code": "def phi(state): return torch.tensor([1.0])"}]
        results = llm.eoh_llm_e1(
            model=None,
            parents=test_parents,
            n=1,
            env_name="tsp",
            debug=True,
            stream=True
        )
        print(f"eoh_llm_e1 streaming test: {'SUCCESS' if results else 'FAILED'}")
        if results:
            print(f"Generated code length: {len(results[0])} characters")
    except Exception as e:
        print(f"eoh_llm_e1 streaming test FAILED: {e}")
    
    print("=" * 60)
    print("EoH functions streaming tests completed!")

if __name__ == "__main__":
    test_streaming_with_env_vars()
    test_eoh_functions_with_streaming()
