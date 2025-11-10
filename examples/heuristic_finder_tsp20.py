"""
Minimal demo for Heuristic-Finder on TSP-20.

It evaluates a few seed potential functions in PBRS on top of POMO policy actions
without training (diagnostic). Extend this by adding training loops or LLM generation.
"""

from rl4co.heuristic_finder.search import HeuristicFinder


def main():
    hf = HeuristicFinder()
    results = hf.run(max_candidates=3)
    for spec, res in results:
        print(f"Candidate: {spec.name}")
        print(f"  avg_base_reward: {res.avg_base_reward:.4f}")
        print(f"  steps: {res.steps}")
        print(
            f"  shaped step reward: mean={res.shaped_step_reward_mean:.4f}, std={res.shaped_step_reward_std:.4f}"
        )
        print("---")


if __name__ == "__main__":
    main()

