from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from rl4co.heuristic_finder.evaluate import EvalResult, evaluate_phi_on_tsp20
from rl4co.heuristic_finder.potential import PotentialSpec, compile_potential, seed_potentials


@dataclass
class Candidate:
    name: str
    code: str


class HeuristicFinder:
    """Simple evolutionary loop scaffold for potential search.

    This is a minimal, pluggable structure: it enumerates seed potentials and
    evaluates them. Mutation/crossover and LLM prompting can be added later.
    """

    def __init__(
        self,
        evaluator: Callable[[PotentialSpec], EvalResult] = evaluate_phi_on_tsp20,
    ):
        self.evaluator = evaluator

    def run(self, max_candidates: int = 3) -> List[tuple[PotentialSpec, EvalResult]]:
        seeds = seed_potentials()
        results: List[tuple[PotentialSpec, EvalResult]] = []
        for idx, (name, spec) in enumerate(seeds.items()):
            if idx >= max_candidates:
                break
            res = self.evaluator(spec)
            results.append((spec, res))
        # Sort by average base reward (more is better because reward=-tour_length)
        results.sort(key=lambda x: x[1].avg_base_reward, reverse=True)
        return results

