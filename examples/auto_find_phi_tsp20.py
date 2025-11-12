"""
EoH-style multi-population search for Phi(state) on TSP-20 with optional full training.

Steps:
1) Multi-population evolutionary search (short POMOPBRS training as fitness)
2) Save best candidate to phi_best.py and dump all candidates with Thought/score/hash
3) (Optional) Launch full training via Hydra run.py

Examples:
  # Local Ollama (recommended for EoH replication)
  python examples/auto_find_phi_tsp20.py \
    --ollama-model qwen3:32b \
    --n-pops 4 --pop-size 8 --generations 10 \
    --operators e1,e2,m1,m2 --operator-weights 1,1,1,1 \
    --memetic-repair-prob 0.25 --elite-parent-k 2 --archive-top-k 32 \
    --epochs-per-eval 1 --batch-size 64 --train-size 1000 --val-size 256 \
    --dump-dir runs/eoh --train-after --train-epochs 100 --gpus 1

  # Remote API (set DEEPSEEK_API_KEY or KIMI_API_KEY + LLM_API_PROVIDER=kimi)
  DEEPSEEK_API_KEY=sk-... python examples/auto_find_phi_tsp20.py \
    --n-pops 2 --pop-size 6 --generations 6 --dump-dir runs/eoh
"""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import List

from rl4co.heuristic_finder.evosearch import EvoConfig, evolution_search
from dotenv import load_dotenv

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

provider = os.getenv("LLM_API_PROVIDER", "deepseek").lower()
print("LLM API Provider:", provider)
if provider == "kimi":
    print("KIMI_API_KEY:", os.getenv("KIMI_API_KEY"))
else:
    print("DEEPSEEK_API_KEY:", os.getenv("DEEPSEEK_API_KEY"))

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    # LLM / Evolution settings (EoH-style)
    p.add_argument("--ollama-model", type=str, default=None, help="Ollama model name, e.g., qwen3:32b")
    p.add_argument("--n-pops", type=int, default=1, help="# independent populations (EoH ec_n_pop)")
    p.add_argument("--pop-size", type=int, default=4, help="population size per pop (EoH ec_pop_size)")
    p.add_argument("--generations", type=int, default=2, help="# evolutionary generations")
    p.add_argument("--operators", type=str, default="e1,e2,m1,m2", help="comma list of operators")
    p.add_argument("--operator-weights", type=str, default=None, help="comma list of per-operator probabilities")
    p.add_argument("--tournament-k", type=int, default=2, help="Tournament size")
    p.add_argument("--epochs-per-eval", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-size", type=int, default=1000)
    p.add_argument("--val-size", type=int, default=256)
    p.add_argument("--num-starts", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--gpu-ids", type=str, default=None, help="Comma-separated GPU ids for parallel short-training, e.g., 0,1,2,3")

    # PBRS tuning
    p.add_argument("--gamma-choices", type=str, default="1.0,-0.1,0.1", help="Comma-separated gamma candidates to sample for phi")
    p.add_argument("--reward-scale", type=str, default=None, help="Advantage scaling: None|scale|norm")
    p.add_argument("--center-dphi", action="store_true", help="Center Delta-Phi within batch during shaping")
    p.add_argument("--norm-dphi", action="store_true", help="Normalize Delta-Phi std within batch during shaping")

    # Save
    p.add_argument("--save-path", type=str, default="phi_best.py")
    p.add_argument("--topk", type=int, default=3, help="Print top-K after search")
    p.add_argument("--dump-dir", type=str, default=None, help="Directory to dump all candidate phi codes per generation")
    p.add_argument(
        "--seed-dump-dir",
        type=str,
        default=None,
        help="Path to an existing dump directory (e.g., runs/eoh) to seed the initial population",
    )
    p.add_argument("--seed", type=int, default=None, help="Fixed seed for reproducible short-training fitness evaluation")

    # Diversity / archive / memetic
    p.add_argument("--dedup-global", action="store_true", help="Enable global dedup (default true)")
    p.add_argument("--no-dedup-global", dest="dedup_global", action="store_false")
    p.set_defaults(dedup_global=True)
    p.add_argument("--dedup-within-pop", action="store_true", help="Enable within-pop dedup (default true)")
    p.add_argument("--no-dedup-within-pop", dest="dedup_within_pop", action="store_false")
    p.set_defaults(dedup_within_pop=True)
    p.add_argument("--memetic-repair-prob", type=float, default=0.0, help="Prob. to apply light repair to offspring")
    p.add_argument("--archive-top-k", type=int, default=8, help="Top-K to keep in elite archive")
    p.add_argument("--elite-parent-k", type=int, default=0, help="Inject K elites into parent pool")
    p.add_argument("--elite-replace-worst", type=int, default=0, help="Replace worst K with elites after merge")

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

    # Operators and weights
    ops = [x.strip() for x in args.operators.split(",") if x.strip()]
    opw = None
    if args.operator_weights:
        opw = [float(x) for x in args.operator_weights.split(",") if x.strip()]

    # If no Ollama model, evolution falls back to DeepSeek API using DEEPSEEK_API_KEY

    cfg = EvoConfig(
        n_pops=args.n_pops,
        pop_size=args.pop_size,
        generations=args.generations,
        epochs_per_eval=args.epochs_per_eval,
        batch_size=args.batch_size,
        train_size=args.train_size,
        val_size=args.val_size,
        num_starts=args.num_starts,
        device=args.device,
        ollama_model=args.ollama_model,
        tournament_k=args.tournament_k,
        operators=ops,
        operator_weights=opw,
        gpu_ids=gpu_ids,
        dump_dir=args.dump_dir,
        seed_dump_dir=args.seed_dump_dir,
        seed=args.seed,
        pbrs_gamma_choices=gamma_choices if gamma_choices else None,
        reward_scale=args.reward_scale,
        center_dphi=bool(args.center_dphi),
        norm_dphi=bool(args.norm_dphi),
        dedup_global=bool(args.dedup_global),
        dedup_within_pop=bool(args.dedup_within_pop),
        memetic_repair_prob=float(args.memetic_repair_prob),
        archive_top_k=int(args.archive_top_k),
        elite_parent_k=int(args.elite_parent_k),
        elite_replace_worst=int(args.elite_replace_worst),
    )
    results = evolution_search(cfg)

    # 2) Save best
    if not results:
        raise SystemExit("No candidates produced by evolution search.")

    best_cand, best_score = results[0]
    save_path = os.path.abspath(args.save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(best_cand.spec.code)
    # In our fitness, higher reward is better and equals negative tour length
    est_tour_len = -float(best_score)
    print(f"Saved best phi to: {save_path}")
    print(f"Estimated tour length (from val/reward): {est_tour_len:.4f}")
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

