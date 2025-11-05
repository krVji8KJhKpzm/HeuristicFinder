from __future__ import annotations

from typing import List


def format_prompt(env_name: str = "tsp", guidance: str = "") -> str:
    return (
        "You are designing a potential function Phi(state) for PBRS in "
        f"combinatorial optimization env '{env_name}'. "
        "Return ONLY a Python function named 'phi(state)' using torch operations. "
        "state exposes: current_loc(), unvisited_locs(), num_remaining(), etc.\n"
        + guidance
    )


def _strip_think_tags(text: str) -> str:
    """Remove '<think> ... </think>' blocks that some models emit."""
    try:
        import re
        return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    except Exception:
        return text


def _extract_code_block(text: str) -> str:
    """Extract code from Markdown-style fenced blocks if present; otherwise return raw text."""
    if not text:
        return text
    # look for ```python ... ``` or ``` ... ```
    start = None
    lang_markers = ("```python", "```py", "```")
    for lm in lang_markers:
        idx = text.find(lm)
        if idx != -1:
            start = idx + len(lm)
            break
    if start is None:
        return text
    end = text.find("```", start)
    if end == -1:
        return text
    return text[start:end].strip()


def generate_candidates_via_ollama(
    model: str, prompt: str, n: int = 1, debug: bool = False
) -> List[str]:
    """Generate code snippets via Ollama. Requires 'ollama' available in runtime.

    Returns a list of code strings (each must define def phi(state): ...).

    Note: This function is not executed in typical offline environments.
    """
    try:
        import ollama  # type: ignore
    except Exception as e:
        if debug:
            print("[HeuristicFinder] Ollama Python package not found or failed to import:", e, flush=True)
        return []

    out: List[str] = []
    for i in range(n):
        try:
            resp = ollama.generate(model=model, prompt=prompt, stream=False)
            if isinstance(resp, dict):
                raw = resp.get("response", "")
            else:
                raw = str(resp)
            cleaned = _strip_think_tags(raw)
            code = _extract_code_block(cleaned)
            if debug:
                print("=" * 80, flush=True)
                print(code, flush=True)
                print("=" * 80, flush=True)
            out.append(code)
        except Exception as e:
            if debug:
                print(f"[HeuristicFinder] Ollama generate failed at sample {i}: {e}", flush=True)
            continue
    return out
