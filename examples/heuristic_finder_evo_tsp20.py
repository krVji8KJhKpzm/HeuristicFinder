"""
EoH-style Phi(s) search example on TSP-20 using offline MSE as fitness.
If Ollama is available, use it to generate candidates; otherwise falls back to the DeepSeek or Kimi API.

Fitness is 1 / MSE between Phi(s) and the Monte Carlo value V(s) estimated
from a fixed baseline policy on randomly generated TSP instances.
"""

from rl4co.heuristic_finder.evosearch import EvoConfig, evolution_search
import math


def main():
    cfg = EvoConfig(
        n_pops=2,
        pop_size=4,
        generations=3,
        operators=["e1", "e2", "m1", "m2", "m3"],
        operator_weights=[1, 1, 1, 1, 1],
        epochs_per_eval=1,
        batch_size=64,
        train_size=1000,
        val_size=256,
        num_starts=8,
        device="cpu",
        # Enable if Ollama and model are available:
        # ollama_model="qwen3:32b",
        dump_dir="runs/eoh_demo",
        dedup_within_pop=True,
        dedup_global=True,
        memetic_repair_prob=0.1,
        archive_top_k=8,
        elite_parent_k=0,
    )
    results = evolution_search(cfg)
    print("=== Top Candidates ===")
    for cand, score in results:
        mse = None
        if isinstance(cand.stats, dict):
            mse = cand.stats.get("mse", None)
        if mse is None and score > 0:
            mse = 1.0 / float(score)
        if mse is not None and mse > 0 and math.isfinite(mse):
            extra = f"mse={mse:.6f}"
        else:
            extra = "mse=n/a"
        print(f"{cand.spec.name} [gamma={cand.gamma:+.3f}]: fitness={score:.4f}, {extra}")


if __name__ == "__main__":
    main()

