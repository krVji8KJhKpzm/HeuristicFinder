from __future__ import annotations

import ast
from dataclasses import dataclass
import re
import textwrap
from types import MappingProxyType
from typing import Callable, Dict, Optional

import torch

from rl4co.envs.routing.tsp.pbrs_env import TSPStateView


@dataclass
class PotentialSpec:
    """Container for a potential function and its source code."""

    name: str
    code: str
    fn: Callable[[TSPStateView], torch.Tensor]


def _safe_exec_namespace() -> Dict[str, object]:
    """Create a minimal, safe namespace for executing LLM-proposed code.

    Disables imports and dunder access; exposes torch and simple ops.
    """

    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "float": float,
        "int": int,
        "range": range,
        "enumerate": enumerate,
        # no __import__, open, exec, eval, etc.
    }
    ns = {"__builtins__": MappingProxyType(safe_builtins)}
    ns.update({
        "torch": torch,
        "nan": float("nan"),
    })
    return ns


def compile_potential(code: str) -> Callable[[TSPStateView], torch.Tensor]:
    """Compile a potential function from text containing `def phi(state): ...`.

    Be robust to extra prose/markdown from LLMs by extracting the function body.
    The function must return a tensor-like broadcastable to [batch, 1].
    """
    def sanitize(c: str) -> str:
        c = c.replace("\r\n", "\n").replace("\r", "\n")
        # Extract fenced code if present
        if "```" in c:
            try:
                start = None
                for lm in ("```python", "```py", "```"):
                    idx = c.find(lm)
                    if idx != -1:
                        start = idx + len(lm)
                        break
                if start is not None:
                    end = c.find("```", start)
                    if end != -1:
                        c = c[start:end]
            except Exception:
                pass
        c = c.strip()
        # Always trim to the first occurrence of the phi function to drop any prose
        m = re.search(r"def\s+phi\s*\(.*?\):[\s\S]*", c)
        if m:
            c = m.group(0)
        # Dedent for consistent indentation
        c = textwrap.dedent(c).strip()
        # Fix common incorrect torch.Tensor.to usage from LLMs:
        # `.to(torch.float32, device=...)` -> `.to(dtype=torch.float32, device=...)`
        try:
            c = re.sub(
                r"\.to\(\s*(torch\.[A-Za-z0-9_]+)\s*,\s*device\s*=",
                r".to(dtype=\1, device=",
                c,
            )
        except Exception:
            pass
        return c

    code = sanitize(code)

    # Parse AST with safety checks
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # last attempt: extract only the phi function and parse again
        m = re.search(r"def\s+phi\s*\(.*?\):[\s\S]*", code)
        if not m:
            raise
        code = textwrap.dedent(m.group(0)).strip()
        tree = ast.parse(code)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
            raise ValueError("Unsafe statement in potential code")

    # Build parent links (ast does not expose them by default)
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            setattr(child, "parent", parent)

    # Enforce restricted access to state.* helpers: allow only raw N-dependent helpers.
    # This prevents LLM outputs from using convenience methods you have disabled.
    allowed_state_calls = {
        "action_mask",
        "visited_mask",
        "unvisited_mask",
        "current_node_index",
        "first_node_index",
        "distance_matrix",
        # newly exposed helpers
        "all_node_coords",
        "partial_path_indices",
    }

    def _root_name(attr: ast.AST) -> str:
        # Walk down Attribute.value until reaching a Name
        cur = attr
        while isinstance(cur, ast.Attribute):
            cur = cur.value  # type: ignore[attr-defined]
        if isinstance(cur, ast.Name):
            return cur.id
        return ""

    # Reject any access like state.xxx that is not explicitly allowed
    for node in ast.walk(tree):
        # Check function calls state.xxx(...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and _root_name(node.func) == "state":
            if node.func.attr not in allowed_state_calls:
                raise ValueError(f"Forbidden state helper: state.{node.func.attr}(...) is not allowed")
        # Check attribute access on state that isn't immediately a call
        if isinstance(node, ast.Attribute) and _root_name(node) == "state":
            parent = getattr(node, "parent", None)
            is_call_func = isinstance(parent, ast.Call) and parent.func is node
            if (not is_call_func) and node.attr not in allowed_state_calls:
                raise ValueError(f"Forbidden state attribute access: state.{node.attr} is not allowed")

    ns: Dict[str, object] = _safe_exec_namespace()
    exec(compile(tree, filename="<potential>", mode="exec"), ns, ns)
    if "phi" not in ns or not callable(ns["phi"]):
        raise ValueError("Potential code must define a callable `phi(state)`")
    return ns["phi"]  # type: ignore


# Note: built-in seed potentials were removed. Use LLM-based initialization instead.
