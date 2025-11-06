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

    ns: Dict[str, object] = _safe_exec_namespace()
    exec(compile(tree, filename="<potential>", mode="exec"), ns, ns)
    if "phi" not in ns or not callable(ns["phi"]):
        raise ValueError("Potential code must define a callable `phi(state)`")
    return ns["phi"]  # type: ignore


def seed_potentials() -> Dict[str, PotentialSpec]:
    """Provide a few simple seed potentials as starting points."""

    seeds: Dict[str, str] = {}

    # 1) Encourage visiting closer unvisited nodes: negative of nearest distance
    seeds["neg_nearest_unvisited"] = (
        """
def phi(state):
    # state.unvisited_locs(): [B,N,2] with NaNs for visited
    cur = state.current_loc()  # [B,2]
    locs = state.unvisited_locs()  # [B,N,2]
    diff = locs - cur.unsqueeze(1)  # [B,N,2]
    d = torch.linalg.norm(diff, dim=-1, ord=2)  # [B,N]
    d = torch.nan_to_num(d, nan=1e9)
    nearest = d.min(dim=-1).values  # [B]
    return (-nearest).unsqueeze(-1)
"""
    )

    # 2) Encourage progress: negative remaining nodes
    seeds["neg_remaining"] = (
        """
def phi(state):
    return (-state.num_remaining().float()).unsqueeze(-1)
"""
    )

    # 3) Mixed: small weight on nearest plus progress
    seeds["mix_near_progress"] = (
        """
def phi(state):
    cur = state.current_loc()
    locs = state.unvisited_locs()
    diff = locs - cur.unsqueeze(1)
    d = torch.linalg.norm(diff, dim=-1, ord=2)
    d = torch.nan_to_num(d, nan=1e9)
    nearest = d.min(dim=-1).values
    return (-(0.7*nearest + 0.3*state.num_remaining().float())).unsqueeze(-1)
"""
    )

    compiled: Dict[str, PotentialSpec] = {}
    for name, code in seeds.items():
        fn = compile_potential(code)
        compiled[name] = PotentialSpec(name=name, code=code, fn=fn)
    return compiled
