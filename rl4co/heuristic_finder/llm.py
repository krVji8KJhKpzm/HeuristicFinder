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


def generate_candidates_via_ollama(
    model: str, prompt: str, n: int = 1
) -> List[str]:
    """Generate code snippets via Ollama. Requires 'ollama' available in runtime.

    Returns a list of code strings (each must define def phi(state): ...).

    Note: This function is not executed in typical offline environments.
    """
    try:
        import ollama  # type: ignore
    except Exception:
        return []

    out: List[str] = []
    for _ in range(n):
        resp = ollama.generate(model=model, prompt=prompt)
        code = resp.get("response", "")
        out.append(code)
    return out

