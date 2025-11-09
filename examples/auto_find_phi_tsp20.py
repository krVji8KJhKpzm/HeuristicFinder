"""
End-to-end pipeline to automatically search for the best Phi(s) on TSP-20 and optionally start full training.

Steps:
1) Evolutionary search with short POMOPBRS training as fitness
2) Save the best candidate to a file (phi_best.py)
3) (Optional) Kick off full training using Hydra run.py with the saved potential

Example:
  # Use local Ollama (preferred when available)
  python examples/auto_find_phi_tsp20.py --ollama-model qwen3:32b --population-size 6 --iterations 3 --train-after \
      --train-epochs 100 --gpus 1 --batch-size 512

  # Or, omit --ollama-model to use DeepSeek API (set DEEPSEEK_API_KEY)
  DEEPSEEK_API_KEY=sk-... python examples/auto_find_phi_tsp20.py --population-size 6 --iterations 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import List

from rl4co.heuristic_finder.evosearch import EvoConfig, evolution_search


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # Evo config (Ollama-only, EoH-enabled)
    p.add_argument("--ollama-model", type=str, default=None, help="Ollama model name, e.g., qwen3:32b")
    p.add_argument("--frac-crossover", type=float, default=0.5, help="Fraction of offspring generated via crossover (rest via mutation)")
    p.add_argument("--tournament-k", type=int, default=2, help="Tournament size when picking parents")
    p.add_argument("--novelty-weight", type=float, default=0.0, help="Add small novelty pressure to survivor selection")
    p.add_argument("--population-size", type=int, default=4)
    p.add_argument("--survivors", type=int, default=2)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--offspring-per-iter", type=int, default=2)
    p.add_argument("--epochs-per-eval", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-size", type=int, default=1000)
    p.add_argument("--val-size", type=int, default=256)
    p.add_argument("--num-starts", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--gpu-ids", type=str, default=None, help="Comma-separated GPU ids for parallel short-training, e.g., 0,1,2,3")

    # PBRS tuning
    p.add_argument("--gamma-choices", type=str, default="1.0,-0.1,0.1", help="Comma-separated gamma candidates to scan per phi")
    p.add_argument("--reward-scale", type=str, default=None, help="Advantage scaling: None|scale|norm")
    p.add_argument("--center-dphi", action="store_true", help="Center Delta-Phi within batch during shaping")
    p.add_argument("--norm-dphi", action="store_true", help="Normalize Delta-Phi std within batch during shaping")

    # Save
    p.add_argument("--save-path", type=str, default="phi_best.py")
    p.add_argument("--topk", type=int, default=3, help="Print top-K after search")
    p.add_argument("--dump-dir", type=str, default=None, help="Directory to dump all candidate phi codes per generation")
    p.add_argument("--seed", type=int, default=None, help="Fixed seed for reproducible short-training fitness evaluation")

    # Train after search
    p.add_argument("--train-after", action="store_true")
    p.add_argument("--train-epochs", type=int, default=100)
    p.add_argument("--train-batch-size", type=int, default=512)
    p.add_argument("--train-size-full", type=int, default=200_000)
    p.add_argument("--val-size-full", type=int, default=10_000)
    p.add_argument("--test-size-full", type=int, default=10_000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gpus", type=int, default=0, help="#GPUs to use. 0 means CPU.")
    return p.parse_args()


def main():
    args = parse_args()

    # 1) Evolutionary search
    # parse gpu ids list
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]

    # parse gamma list
    gamma_choices = [float(x) for x in args.gamma_choices.split(",") if x.strip() != ""] if args.gamma_choices else []

    # If no Ollama model, evolution falls back to DeepSeek API using DEEPSEEK_API_KEY

    cfg = EvoConfig(
        population_size=args.population_size,
        survivors=args.survivors,
        iterations=args.iterations,
        offspring_per_iter=args.offspring_per_iter,
        epochs_per_eval=args.epochs_per_eval,
        batch_size=args.batch_size,
        train_size=args.train_size,
        val_size=args.val_size,
        num_starts=args.num_starts,
        device=args.device,
        ollama_model=args.ollama_model,
        frac_crossover=args.frac_crossover,
        tournament_k=args.tournament_k,
        novelty_weight=args.novelty_weight,
        gpu_ids=gpu_ids,
        dump_dir=args.dump_dir,
        seed=args.seed,
        pbrs_gamma_choices=gamma_choices if gamma_choices else None,
        reward_scale=args.reward_scale,
        center_dphi=bool(args.center_dphi),
        norm_dphi=bool(args.norm_dphi),
    )
    results = evolution_search(cfg)

    # 2) Save best
    if not results:
        raise SystemExit("No candidates produced by evolution search.")

    best_cand, best_score = results[0]
    save_path = os.path.abspath(args.save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(best_cand.spec.code)
    print("=== Top Candidates ===")
    for cand, score in results[: args.topk]:
        print(f"{cand.spec.name} [gamma={cand.gamma:+.3f}]: val/reward={score:.4f}")
    # 3) Optional: Start full training via run.py (Hydra)
    if args.train_after:
        cmd: List[str] = ["python", "run.py", "experiment=routing/pomopbrs-tsp20.yaml"]
        cmd.append(f"model.potential=file:{save_path}")
        cmd.append(f"trainer.max_epochs={args.train_epochs}")
        cmd.append(f"model.batch_size={args.train_batch_size}")
        cmd.append(f"model.train_data_size={args.train_size_full}")
        cmd.append(f"model.val_data_size={args.val_size_full}")
        cmd.append(f"model.test_data_size={args.test_size_full}")
        cmd.append(f"model.optimizer_kwargs.lr={args.lr}")
        if args.gpus and args.gpus > 0:
            cmd.append("trainer.accelerator=gpu")
            cmd.append(f"trainer.devices={args.gpus}")
        print("Launching full training:")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
