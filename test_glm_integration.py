#!/usr/bin/env python3
"""
Test script to verify GLM API integration in the rl4co heuristic finder.

This script demonstrates how to use the GLM (General Language Model) API 
for generating potential functions in the rl4co framework.

Environment Variables:
    GLM_API_KEY: Your GLM API key (required)
    GLM_API_BASE: Optional custom API base URL (default: https://open.bigmodel.cn/api/paas/v4)
    GLM_MODEL: Optional custom model name (default: glm-4)
    GLM_TEMPERATURE: Optional temperature setting (default: 0.0)
    GLM_MAX_TOKENS: Optional max tokens setting (default: 1024)

Usage:
    1. Set your GLM API key: export GLM_API_KEY="your-api-key"
    2. Run the test: python test_glm_integration.py
"""

import os
import sys
from rl4co.heuristic_finder.llm import generate_candidates_via_glm, format_prompt

def test_glm_integration():
    """Test GLM API integration with a simple prompt."""
    
    # Check if GLM API key is set
    if not os.environ.get("GLM_API_KEY"):
        print("❌ GLM_API_KEY environment variable is not set.")
        print("Please set your GLM API key: export GLM_API_KEY='your-api-key'")
        return False
    
    print("🧪 Testing GLM API integration...")
    
    # Create a simple test prompt
    prompt = format_prompt(env_name="tsp", guidance="Test prompt for GLM integration.")
    
    try:
        # Generate candidates using GLM
        print("🤖 Calling GLM API...")
        candidates = generate_candidates_via_glm(
            prompt=prompt,
            n=1,
            debug=True,
            model="glm-4"  # You can specify a different model if needed
        )
        
        if candidates:
            print(f"✅ Successfully generated {len(candidates)} candidate(s) using GLM API")
            print("\n📝 Generated code:")
            print("=" * 80)
            print(candidates[0])
            print("=" * 80)
            return True
        else:
            print("❌ No candidates generated. Check your API key and network connection.")
            return False
            
    except Exception as e:
        print(f"❌ Error during GLM API call: {e}")
        return False

def test_provider_resolution():
    """Test that GLM provider is properly resolved."""
    from rl4co.heuristic_finder.llm import _resolve_llm_api_provider
    
    print("\n🔍 Testing provider resolution...")
    
    # Test with GLM_API_KEY set
    os.environ["GLM_API_KEY"] = "test-key"
    provider = _resolve_llm_api_provider()
    print(f"Provider with GLM_API_KEY set: {provider}")
    
    # Test explicit GLM preference
    provider = _resolve_llm_api_provider("glm")
    print(f"Provider with explicit 'glm' preference: {provider}")
    
    # Clean up
    if os.environ.get("GLM_API_KEY") == "test-key":
        del os.environ["GLM_API_KEY"]
    
    print("✅ Provider resolution test completed")

def test_glm_with_different_models():
    """Test GLM with different model configurations."""
    
    print("\n🔧 Testing GLM with different model configurations...")
    
    if not os.environ.get("GLM_API_KEY"):
        print("⚠️  Skipping model configuration test (GLM_API_KEY not set)")
        return
    
    # Test with default model
    print("Testing with default model (glm-4)...")
    candidates = generate_candidates_via_glm(
        prompt=format_prompt("tsp"),
        n=1,
        debug=False
    )
    print(f"Default model result: {'✅ Success' if candidates else '❌ Failed'}")
    
    # Test with custom model
    print("Testing with custom model (glm-3-turbo)...")
    candidates = generate_candidates_via_glm(
        prompt=format_prompt("tsp"),
        n=1,
        model="glm-3-turbo",
        debug=False
    )
    print(f"Custom model result: {'✅ Success' if candidates else '❌ Failed'}")

def main():
    """Main test function."""
    print("🚀 GLM Integration Test Suite")
    print("=" * 50)
    
    # Test 1: Basic integration
    success1 = test_glm_integration()
    
    # Test 2: Provider resolution
    test_provider_resolution()
    
    # Test 3: Different model configurations
    test_glm_with_different_models()
    
    print("\n" + "=" * 50)
    if success1:
        print("✅ GLM integration test completed successfully!")
        print("\n📚 Usage Examples:")
        print("1. Basic usage:")
        print("   from rl4co.heuristic_finder.llm import generate_candidates_via_glm")
        print("   candidates = generate_candidates_via_glm(prompt='Your prompt here')")
        print("\n2. With custom model:")
        print("   candidates = generate_candidates_via_glm(prompt='Your prompt', model='glm-3-turbo')")
        print("\n3. Using the unified interface:")
        print("   os.environ['LLM_API_PROVIDER'] = 'glm'")
        print("   from rl4co.heuristic_finder.llm import generate_candidates")
        print("   candidates = generate_candidates(prompt='Your prompt')")
    else:
        print("❌ Some tests failed. Please check your GLM API configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main()
