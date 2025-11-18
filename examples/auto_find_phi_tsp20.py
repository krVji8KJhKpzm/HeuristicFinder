"""
EoH-style multi-population search for a symbolic Phi(state) on TSP-20.

Fitness is defined as the inverse mean-squared error between Phi(s) and an
offline Monte Carlo estimate V(s) (future tour length) computed from a fixed
baseline policy on randomly generated TSP instances. No RL training is run
inside the loop; the task is pure symbolic regression on the offline dataset.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from typing import List

from rl4co.heuristic_finder.evosearch import EvoConfig, evolution_search
from dotenv import load_dotenv
import math

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
    p.add_argument("--operators", type=str, default="e1,e2,m1,m2,m3", help="comma list of operators")
    p.add_argument("--operator-weights", type=str, default=None, help="comma list of per-operator probabilities")
    p.add_argument("--tournament-k", type=int, default=3, help="Tournament size")
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
    p.add_argument("--elite-parent-k", type=int, default=2, help="Inject K elites into parent pool")
    p.add_argument("--elite-replace-worst", type=int, default=1, help="Replace worst K with elites after merge")
    p.add_argument(
        "--log-evo-details",
        action="store_true",
        help="Print detailed evolution statistics (offspring counts, dedup effects, etc.)",
    )

    # Level 1 / Level 2 multi-stage config
    p.add_argument("--offline-traj-path", type=str, default="data/tsp20_offline_trajs.pt")
    p.add_argument(
        "--offline-traj-paths-multi",
        type=str,
        default=None,
        help="Comma-separated list of additional offline trajectory paths for multi-scale diagnostics (e.g., tsp20,tsp50,tsp100).",
    )
    p.add_argument("--cheap-level-weight", type=float, default=0.1)
    p.add_argument("--cheap-filter-threshold", type=float, default=-1e9)
    p.add_argument("--cheap-topk-ratio", type=float, default=0.3)
    p.add_argument("--max-candidates-rl-eval", type=int, default=8)
    p.add_argument("--max-step-shaping-ratio", type=float, default=10.0)
    p.add_argument("--max-episode-shaping-ratio", type=float, default=10.0)
    p.add_argument("--max-var-ratio-shaped-vs-base", type=float, default=10.0)
    p.add_argument("--min-abs-dphi-q95", type=float, default=1e-4)
    p.add_argument("--complexity-penalty-alpha", type=float, default=0.001)
    p.add_argument("--cheap-eval-device", type=str, default="cpu")
    p.add_argument("--cheap-eval-batch-states", type=int, default=4096)
    p.add_argument("--no-cheap-level", action="store_true", help="Disable Level-1 cheap evaluation; evaluate all candidates with RL")
    p.add_argument(
        "--no-level2-rl",
        action="store_true",
        help="Disable Level-2 short RL evaluation; rank candidates only by Level-1 credit-assignment score",
    )
    p.add_argument("--refine-top-k", type=int, default=0)
    p.add_argument("--refine-epochs", type=int, default=0)

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

    if args.offline_traj_paths_multi:
        offline_traj_paths_multi: List[str] = [
            p for p in args.offline_traj_paths_multi.split(",") if p.strip()
        ]
    else:
        offline_traj_paths_multi = None

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
        offline_traj_path=args.offline_traj_path,
        offline_traj_paths_multi=offline_traj_paths_multi,
        cheap_level_weight=float(args.cheap_level_weight),
        cheap_filter_threshold=float(args.cheap_filter_threshold),
        cheap_topk_ratio=float(args.cheap_topk_ratio),
        max_candidates_rl_eval=int(args.max_candidates_rl_eval),
        max_step_shaping_ratio=float(args.max_step_shaping_ratio),
        max_episode_shaping_ratio=float(args.max_episode_shaping_ratio),
        max_var_ratio_shaped_vs_base=float(args.max_var_ratio_shaped_vs_base),
        min_abs_dphi_q95=float(args.min_abs_dphi_q95),
        complexity_penalty_alpha=float(args.complexity_penalty_alpha),
        cheap_eval_device=str(args.cheap_eval_device),
        cheap_eval_batch_states=int(args.cheap_eval_batch_states),
        refine_top_k=int(args.refine_top_k),
        refine_epochs=int(args.refine_epochs),
        use_cheap_level=not bool(args.no_cheap_level),
        use_level2_rl=not bool(args.no_level2_rl),
        log_evo_details=bool(args.log_evo_details),
    )
    results = evolution_search(cfg)

    # 2) Save best
    if not results:
        raise SystemExit("No candidates produced by evolution search.")

    best_cand, best_score = results[0]
    save_path = os.path.abspath(args.save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(best_cand.spec.code)

    # Fitness is 1 / MSE; report both fitness and estimated MSE
    best_mse = None
    if isinstance(best_cand.stats, dict):
        best_mse = best_cand.stats.get("mse", None)
    if best_mse is None and best_score > 0:
        best_mse = 1.0 / float(best_score)

    print(f"Saved best phi to: {save_path}")
    if best_mse is not None and best_mse > 0 and math.isfinite(best_mse):
        print(f"Best fitness (1/MSE): {best_score:.6f}, estimated MSE: {best_mse:.6f}")
    else:
        print(f"Best fitness (1/MSE): {best_score:.6f}, estimated MSE unavailable")

    print("=== Top Candidates ===")
    for cand, score in results[: args.topk]:
        mse = None
        if isinstance(cand.stats, dict):
            mse = cand.stats.get("mse", None)
        if mse is None and score > 0:
            mse = 1.0 / float(score)
        if mse is not None and mse > 0 and math.isfinite(mse):
            extra = f"mse={mse:.6f}"
        else:
            extra = "mse=n/a"
        print(f"{cand.spec.name} [gamma={cand.gamma:+.3f}]: fitness={score:.6f}, {extra}")
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


