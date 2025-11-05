from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from rl4co.heuristic_finder.evaluate import train_fitness_phi_on_tsp20
from rl4co.heuristic_finder.llm import (
    format_prompt,
    generate_candidates_via_ollama,
)
from rl4co.heuristic_finder.potential import PotentialSpec, compile_potential, seed_potentials


@dataclass
class EvoConfig:
    population_size: int = 4
    survivors: int = 2
    iterations: int = 2
    offspring_per_iter: int = 2
    epochs_per_eval: int = 1
    batch_size: int = 64
    train_size: int = 1000
    val_size: int = 256
    num_starts: int = 8
    device: str = "cpu"
    ollama_model: Optional[str] = None


def jitter_numbers_in_code(code: str, scale: float = 0.2) -> str:
    """Simple mutation: jitter numeric literals by a percentage."""
    def repl(m):
        s = m.group(0)
        try:
            v = float(s)
        except Exception:
            return s
        dv = (random.random() * 2 - 1) * scale * max(1.0, abs(v))
        nv = v + dv
        # keep reasonable precision
        return f"{nv:.6f}"

    # match integers or floats (rudimentary)
    return re.sub(r"(?<![A-Za-z_])(\d+\.\d+|\d+)(?![A-Za-z_])", repl, code)


def make_population_from_seeds(k: int) -> List[PotentialSpec]:
    seeds = list(seed_potentials().values())
    return seeds[:k] if k <= len(seeds) else seeds


def compile_candidates(codes: List[str]) -> List[PotentialSpec]:
    out: List[PotentialSpec] = []
    for i, code in enumerate(codes):
        try:
            fn = compile_potential(code)
        except Exception:
            continue
        out.append(PotentialSpec(name=f"llm_{i}", code=code, fn=fn))
    return out


def propose_offspring(
    parents: List[PotentialSpec], cfg: EvoConfig
) -> List[PotentialSpec]:
    offspring: List[PotentialSpec] = []

    # 1) LLM proposals
    if cfg.ollama_model:
        guidance = (
            "Optimize for larger (less negative) validation reward after 1 epoch "
            "on TSP-20 with POMO (num_starts=8). Keep it simple and stable."
        )
        best = max(parents, key=lambda s: len(s.code))  # naive context pick
        prompt = format_prompt("tsp", guidance) + "\nCurrent best: \n" + best.code
        codes = generate_candidates_via_ollama(
            cfg.ollama_model, prompt, n=cfg.offspring_per_iter, debug=True
        )
        if not codes:
            print("[HeuristicFinder] No LLM candidates produced (check ollama install/model).", flush=True)
        offspring.extend(compile_candidates(codes))

    # 2) Local mutations
    while len(offspring) < cfg.offspring_per_iter:
        p = random.choice(parents)
        mutated_code = jitter_numbers_in_code(p.code)
        try:
            fn = compile_potential(mutated_code)
        except Exception:
            continue
        offspring.append(PotentialSpec(name=p.name+"_mut", code=mutated_code, fn=fn))

    return offspring


def evolution_search(cfg: EvoConfig) -> List[Tuple[PotentialSpec, float]]:
    # init population from seeds and possibly from LLM in the first round
    population = make_population_from_seeds(cfg.population_size)
    scored: List[Tuple[PotentialSpec, float]] = []

    # evaluate initial population
    for spec in population:
        score = train_fitness_phi_on_tsp20(
            spec,
            epochs=cfg.epochs_per_eval,
            batch_size=cfg.batch_size,
            train_data_size=cfg.train_size,
            val_data_size=cfg.val_size,
            num_starts=cfg.num_starts,
            device=cfg.device,
        )
        scored.append((spec, score))

    # iterate
    for _ in range(cfg.iterations):
        # select survivors (higher reward is better)
        scored.sort(key=lambda x: x[1], reverse=True)
        survivors = [s for s, _ in scored[: cfg.survivors]]

        # propose offspring
        offspring = propose_offspring(survivors, cfg)
        # evaluate offspring
        for spec in offspring:
            score = train_fitness_phi_on_tsp20(
                spec,
                epochs=cfg.epochs_per_eval,
                batch_size=cfg.batch_size,
                train_data_size=cfg.train_size,
                val_data_size=cfg.val_size,
                num_starts=cfg.num_starts,
                device=cfg.device,
            )
            scored.append((spec, score))

        # keep top-K as new population
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[: cfg.population_size]

    # final ranking
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
