from __future__ import annotations

import os
from typing import Any

from rl4co.heuristic_finder.potential import compile_potential, seed_potentials
from rl4co.models.zoo.pomo.pbrs_model import POMOPBRS


def _resolve_potential_fn(potential: str):
    """Resolve a potential function from a string descriptor.

    Formats:
    - "seed:<name>": use one of the built-in seeds (e.g., seed:neg_remaining)
    - "file:<path>": read a file containing `def phi(state): ...` and compile
    - otherwise: treat the whole string as code and compile
    """
    if potential.startswith("seed:"):
        name = potential.split(":", 1)[1]
        seeds = seed_potentials()
        if name not in seeds:
            raise ValueError(f"Unknown seed potential '{name}'. Available: {list(seeds)}")
        return seeds[name].fn
    if potential.startswith("file:"):
        path = potential.split(":", 1)[1]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Potential file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
        return compile_potential(code)
    # raw code
    return compile_potential(potential)


def build_pomopbrs_model(env, potential: str = "seed:neg_remaining", **kwargs: Any):
    """Hydra-friendly builder to instantiate a POMOPBRS with a chosen potential.

    Example usages (Hydra):
      model:
        _target_: rl4co.heuristic_finder.builders.build_pomopbrs_model
        potential: seed:neg_remaining
        num_starts: 8
        batch_size: 512
        train_data_size: 200000
    """
    fn = _resolve_potential_fn(potential)
    return POMOPBRS(env=env, potential_fn=fn, **kwargs)

