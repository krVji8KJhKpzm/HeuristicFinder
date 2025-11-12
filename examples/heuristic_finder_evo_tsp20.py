"""
EoH-style Phi(s) search example on TSP-20 using short POMOPBRS training as fitness.
If Ollama is available, use it to generate candidates; otherwise falls back to the DeepSeek or Kimi API.
"""

from rl4co.heuristic_finder.evosearch import EvoConfig, evolution_search


def main():
    cfg = EvoConfig(
        n_pops=2,
        pop_size=4,
        generations=3,
        operators=["e1", "e2", "m1", "m2"],
        operator_weights=[1, 1, 1, 1],
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
        print(f"{cand.spec.name} [gamma={cand.gamma:+.3f}]: val/reward={score:.4f}")


if __name__ == "__main__":
    main()
