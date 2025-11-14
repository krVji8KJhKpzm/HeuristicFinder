#!/usr/bin/env python3
"""
Unified LLM API test script for rl4co heuristic_finder.

Supports providers: deepseek, kimi, glm, ollama, or auto (env based).
Allows toggling streaming and model selection. Uses existing llm.py helpers.

Examples:
  - Auto provider (based on env):
      python scripts/test_api.py --prompt "Write a sum function" --stream

  - Force DeepSeek, non-streaming:
      python scripts/test_api.py --provider deepseek --n 1 --stream 0

  - GLM with custom model:
      python scripts/test_api.py --provider glm --api-model glm-4 --stream

  - Ollama local model:
      python scripts/test_api.py --provider ollama --ollama-model qwen2.5-coder:7b --stream

Environment variables (typical):
  - DEEPSEEK_API_KEY, DEEPSEEK_API_BASE, DEEPSEEK_MODEL
  - KIMI_API_KEY, KIMI_API_BASE, KIMI_MODEL
  - GLM_API_KEY,  GLM_API_BASE,  GLM_MODEL
  - LLM_API_PROVIDER (deepseek|kimi|glm) when --provider auto
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import List, Optional


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _check_api_key(provider: str) -> bool:
    prov = provider.lower()
    if prov == "ollama":
        return True  # local runtime
    env_map = {
        "deepseek": "DEEPSEEK_API_KEY",
        "kimi": "KIMI_API_KEY",
        "glm": "GLM_API_KEY",
    }
    key_env = env_map.get(prov)
    if not key_env:
        return False
    if not os.environ.get(key_env):
        print(f"[WARN] {key_env} is not set; {provider} calls will likely fail.")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Test LLM API providers with streaming/non-streaming")
    parser.add_argument("--provider", default="auto", choices=["auto", "deepseek", "kimi", "glm", "ollama"], help="Which provider to use")
    parser.add_argument("--prompt", default="Generate a simple Python function that returns the sum of two numbers.")
    parser.add_argument("--env-name", default="tsp", help="Environment name used by format_prompt")
    parser.add_argument("--n", type=int, default=1, help="# of candidates")
    parser.add_argument("--stream", type=int, default=1, help="1 to enable streaming, 0 to disable")
    parser.add_argument("--debug", type=int, default=1, help="Verbose printing inside llm helpers")
    parser.add_argument("--api-model", default=None, help="Model name for API providers (deepseek/kimi/glm)")
    parser.add_argument("--ollama-model", default=None, help="Local Ollama model (e.g., qwen2.5-coder:7b)")
    parser.add_argument("--two-stage", action="store_true", help="Use two-stage (reasoner+coder) pipeline")
    parser.add_argument("--reasoner-model", default=None, help="Reasoner model for two-stage (provider)")
    parser.add_argument("--coder-ollama-model", default=None, help="Coder model for two-stage (ollama)")
    args = parser.parse_args()

    # Deferred imports to keep startup fast
    from rl4co.heuristic_finder.llm import (
        generate_candidates,
        generate_candidates_via_deepseek,
        generate_candidates_via_kimi,
        generate_candidates_via_glm,
        generate_candidates_via_ollama,
        two_stage_generate_candidates,
        format_prompt,
        _resolve_llm_api_provider,
    )

    prompt = args.prompt
    # If prompt looks short, enrich via format_prompt to align with phi generation style
    if len(prompt) < 64:
        prompt = format_prompt(args.env_name, guidance=f"\nTask: {args.prompt}\n")

    provider = args.provider.lower()
    if provider == "auto":
        provider = _resolve_llm_api_provider()
        print(f"[INFO] Auto-selected provider: {provider}")

    _print_header(f"Testing provider='{provider}', stream={bool(args.stream)}, n={args.n}")
    _check_api_key(provider)

    results: List[str]
    try:
        if args.two_stage and provider in {"deepseek", "kimi", "glm"}:
            results = two_stage_generate_candidates(
                prompt=prompt,
                n=args.n,
                env_name=args.env_name,
                debug=bool(args.debug),
                reasoner_model=args.reasoner_model,
                coder_ollama_model=args.coder_ollama_model,
                api_provider=provider,
            )
        else:
            if provider == "ollama":
                if not args.ollama_model:
                    print("[ERROR] --ollama-model is required when provider=ollama")
                    return 2
                results = generate_candidates_via_ollama(
                    model=args.ollama_model,
                    prompt=prompt,
                    n=args.n,
                    debug=bool(args.debug),
                    stream=bool(args.stream),
                )
            elif provider == "deepseek":
                results = generate_candidates_via_deepseek(
                    prompt=prompt,
                    n=args.n,
                    model=args.api_model,
                    debug=bool(args.debug),
                    stream=bool(args.stream),
                )
            elif provider == "kimi":
                results = generate_candidates_via_kimi(
                    prompt=prompt,
                    n=args.n,
                    model=args.api_model,
                    debug=bool(args.debug),
                    stream=bool(args.stream),
                )
            elif provider == "glm":
                results = generate_candidates_via_glm(
                    prompt=prompt,
                    n=args.n,
                    model=args.api_model,
                    debug=bool(args.debug),
                    stream=bool(args.stream),
                )
            else:
                # Fallback to unified interface
                results = generate_candidates(
                    prompt=prompt,
                    n=args.n,
                    debug=bool(args.debug),
                    ollama_model=args.ollama_model,
                    api_model=args.api_model,
                )

        print("\n\n--- Summary ---")
        if not results:
            print("No candidates returned.")
            return 1
        for i, code in enumerate(results):
            print(f"Candidate {i+1}: length={len(code)} chars")
            preview = code[:200]
            print(preview + ("..." if len(code) > 200 else ""))
            print("-" * 40)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 130
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

