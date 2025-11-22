from __future__ import annotations

"""
Offline trajectory collection for TSP to support credit-assignment-aware Level-1
PBRS potential evaluation.

This module collects trajectories using a baseline POMO model without PBRS and
stores the minimal state needed to evaluate arbitrary potential functions
phi(s) *offline*, together with generic per-step credit signals (returns and
simple advantages) that can be reused by different credit-assignment metrics.

Usage:
  python -m rl4co.heuristic_finder.offline_data_tsp20 \
    --out-path data/tsp20_offline_trajs.pt \
    --num-episodes 512 --batch-size 128 --seed 1234 [--ckpt <path>]

Output file format (torch .pt): a dict with keys:
  episodes: List[dict] where each dict contains
    - locs: Tensor [N,2]
    - first_node: int
    - actions: LongTensor [T]
    - current_nodes: LongTensor [T]      (node before taking action t)
    - action_masks: BoolTensor [T,N]
    - base_step_reward: Tensor [T]       (negative edge length, r(s,a))
    - returns: Tensor [T]                (undiscounted Monte Carlo return from t)
    - advantages: Tensor [T]             (returns minus per-episode mean return)
    - final_reward: float                (negative tour length)
  meta: { num_loc, seed }

Note: We explicitly store base_step_reward as the negative edge length so it
matches the PBRS shaping convention used in DensePBRSTSPEnv, and derive
returns/advantages from it as generic per-step credit signals.
"""

import argparse
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import lightning as L

from tensordict.tensordict import TensorDict

from rl4co.envs.routing.tsp.env import TSPEnv, TSPGenerator
from rl4co.models.zoo.pomo import POMO
from rl4co.utils.ops import gather_by_index, get_tour_length


def _negative_edge_lengths(locs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Compute per-step negative edge length for a given tour.

    Args:
      locs: [N,2] or [B,N,2] coordinates
      actions: [T] indices of visited nodes in order
    Returns:
      Tensor [T] of negative euclidean distances between consecutive steps
    """
    if locs.dim() == 3:
        # assume batch size 1 if batched passed inadvertently
        locs = locs[0]
    # last node before action t is actions[t-1] for t>0; for t=0, last is 0 (start)
    last_nodes = torch.zeros_like(actions)
    if actions.numel() > 1:
        last_nodes[1:] = actions[:-1]
    last_locs = locs[last_nodes]
    curr_locs = locs[actions]
    d = torch.linalg.norm(curr_locs - last_locs, dim=-1, ord=2)
    return -d


_CONCORDE_SCALE = 100000.0
_LKH_SCALE = 100000.0


def _ensure_pyconcorde_available() -> None:
    try:
        import pyconcorde.tsp  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pyconcorde is required for solver='concorde'. "
            "Install it with `pip install pyconcorde` and make sure Concorde is available."
        ) from e


def _solve_single_concorde(coords: np.ndarray) -> np.ndarray:
    """Solve a single TSP instance with pyconcorde.

    Args:
      coords: [N, 2] coordinates in [0, 1]
    Returns:
      np.ndarray [N] with node indices of the optimal tour
    """
    from pyconcorde.tsp import TSPSolver

    xs = coords[:, 0] * _CONCORDE_SCALE
    ys = coords[:, 1] * _CONCORDE_SCALE
    solver = TSPSolver.from_data(xs, ys, norm="EUC_2D")
    solution = solver.solve()
    return np.asarray(solution.tour, dtype=np.int64)


def _solve_with_concorde_batch(locs: torch.Tensor, n_jobs: int = 1) -> torch.Tensor:
    """Solve a batch of TSP instances with pyconcorde.

    Args:
      locs: [B, N, 2] coordinates in [0, 1]
      n_jobs: number of parallel pyconcorde workers (<=1 means sequential)
    Returns:
      LongTensor [B, N] with node indices of the optimal tour
    """
    _ensure_pyconcorde_available()

    locs = locs.detach().cpu()
    B, N, _ = locs.shape
    coords_list = [locs[b].numpy() for b in range(B)]

    if n_jobs <= 1 or B == 1:
        tours = []
        for coords in coords_list:
            tour_np = _solve_single_concorde(coords)
            if tour_np.shape[0] != N:
                raise RuntimeError(
                    f"Concorde returned a tour of length {tour_np.shape[0]} for N={N}"
                )
            tours.append(torch.as_tensor(tour_np, dtype=torch.long))
    else:
        from concurrent.futures import ProcessPoolExecutor

        tours = []
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for tour_np in ex.map(_solve_single_concorde, coords_list):
                if tour_np.shape[0] != N:
                    raise RuntimeError(
                        f"Concorde returned a tour of length {tour_np.shape[0]} for N={N}"
                    )
                tours.append(torch.as_tensor(tour_np, dtype=torch.long))

    return torch.stack(tours, dim=0)


def _ensure_lkh_available(exe: str) -> None:
    import shutil

    if shutil.which(exe) is None:
        raise ImportError(
            f"LKH executable '{exe}' not found in PATH. "
            "Install LKH and/or set --lkh-exe to its path."
        )


def _solve_single_lkh(args: tuple[np.ndarray, str]) -> np.ndarray:
    """Solve a single TSP instance with LKH via TSPLIB files.

    Args:
      args: (coords [N,2] in [0,1], lkh_exe)
    Returns:
      np.ndarray [N] with node indices of the tour (0-based)
    """
    import os
    import subprocess
    import tempfile

    coords, exe = args
    N = coords.shape[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        tsp_path = os.path.join(tmpdir, "problem.tsp")
        par_path = os.path.join(tmpdir, "problem.par")
        tour_path = os.path.join(tmpdir, "problem.tour")

        # Write TSPLIB file
        with open(tsp_path, "w") as f:
            f.write("NAME: problem\n")
            f.write("TYPE: TSP\n")
            f.write(f"DIMENSION: {N}\n")
            f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
            f.write("NODE_COORD_SECTION\n")
            for i, (x, y) in enumerate(coords, start=1):
                f.write(
                    f"{i} {float(x) * _LKH_SCALE:.6f} {float(y) * _LKH_SCALE:.6f}\n"
                )
            f.write("EOF\n")

        # Write parameter file
        with open(par_path, "w") as f:
            f.write(f"PROBLEM_FILE = {tsp_path}\n")
            f.write(f"OUTPUT_TOUR_FILE = {tour_path}\n")
            f.write("RUNS = 1\n")
            f.write("TRACE_LEVEL = 0\n")

        # Run LKH
        subprocess.run(
            [exe, par_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Parse tour output
        tour: list[int] = []
        in_section = False
        with open(tour_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("TOUR_SECTION"):
                    in_section = True
                    continue
                if not in_section:
                    continue
                if line.startswith("-1") or line.startswith("EOF"):
                    break
                for tok in line.split():
                    node = int(tok)
                    if node == -1:
                        in_section = False
                        break
                    tour.append(node - 1)  # convert to 0-based

        if len(tour) != N:
            raise RuntimeError(f"LKH returned a tour of length {len(tour)} for N={N}")
        return np.asarray(tour, dtype=np.int64)


def _solve_with_lkh_batch(
    locs: torch.Tensor,
    n_jobs: int = 1,
    exe: str = "LKH",
) -> torch.Tensor:
    """Solve a batch of TSP instances with LKH.

    Args:
      locs: [B, N, 2] coordinates in [0, 1]
      n_jobs: number of parallel LKH workers (<=1 means sequential)
      exe: LKH executable name or path
    Returns:
      LongTensor [B, N] with node indices of the tour
    """
    _ensure_lkh_available(exe)

    locs = locs.detach().cpu()
    B, N, _ = locs.shape
    coords_list = [locs[b].numpy() for b in range(B)]
    args_list = [(coords, exe) for coords in coords_list]

    if n_jobs <= 1 or B == 1:
        tours = []
        for args in args_list:
            tour_np = _solve_single_lkh(args)
            if tour_np.shape[0] != N:
                raise RuntimeError(
                    f"LKH returned a tour of length {tour_np.shape[0]} for N={N}"
                )
            tours.append(torch.as_tensor(tour_np, dtype=torch.long))
    else:
        from concurrent.futures import ProcessPoolExecutor

        tours = []
        with ProcessPoolExecutor(max_workers=n_jobs) as ex_pool:
            for tour_np in ex_pool.map(_solve_single_lkh, args_list):
                if tour_np.shape[0] != N:
                    raise RuntimeError(
                        f"LKH returned a tour of length {tour_np.shape[0]} for N={N}"
                    )
                tours.append(torch.as_tensor(tour_np, dtype=torch.long))

    return torch.stack(tours, dim=0)


def collect_tsp20_trajectories(
    out_path: str,
    num_episodes: int = 512,
    batch_size: int = 128,
    seed: int = 1234,
    num_loc: Optional[int] = None,
    ckpt_path: Optional[str] = None,
    device: str = "cpu",
    solver: str = "pomo",
    concorde_workers: int = 0,
    lkh_exe: str = "LKH",
) -> str:
    """Collect offline trajectories for TSP.

    If solver == "pomo", roll out a baseline POMO policy (optionally from a
    checkpoint). If solver == "concorde", ignore ckpt_path and instead use
    pyconcorde to generate optimal tours.

    Returns the path written to.
    """
    if seed is not None:
        L.seed_everything(seed, workers=True)

    if num_loc is None:
        # Default to environment variable used elsewhere; fallback TSP-20
        import os

        try:
            num_loc = int(os.environ.get("TSP_NUM_LOC", "20"))
        except Exception:
            num_loc = 20

    gen = TSPGenerator(num_loc=num_loc)
    base_env = TSPEnv(generator=gen, seed=seed)

    solver = str(solver).lower()
    if solver not in {"pomo", "concorde", "lkh"}:
        raise ValueError(
            f"Unknown solver '{solver}', expected 'pomo', 'concorde', or 'lkh'"
        )

    model: Optional[POMO] = None
    if solver == "pomo":
        # Build policy and optionally load a checkpoint
        if ckpt_path:
            model = POMO.load_from_checkpoint(
                ckpt_path, env=base_env, load_baseline=False
            )
            print(f"Loaded POMO model from checkpoint: {ckpt_path}", flush=True)
        else:
            model = POMO(env=base_env)
        model.eval()
        # Ensure model is on the same device as tensors we generate
        try:
            model.to(torch.device(device))
        except Exception:
            pass

    episodes: List[Dict[str, Any]] = []

    n_remaining = int(num_episodes)
    while n_remaining > 0:
        cur_bs = min(batch_size, n_remaining)
        td0 = base_env.reset(base_env.generator(batch_size=[cur_bs]).to(device))
        if solver == "pomo":
            assert model is not None
            # Safety: align td device with model parameters if needed
            try:
                pdev = next(model.parameters()).device
                if pdev != td0.device:
                    td0 = td0.to(pdev)
            except Exception:
                pass
            with torch.no_grad():
                out = model.policy(
                    td0,
                    base_env,
                    phase="val",
                    return_actions=True,
                    decode_type="greedy",
                )
                actions = out["actions"]  # [B, T]
        else:
            # External exact solvers (Concorde / LKH)
            locs = td0["locs"]  # [B, N, 2]
            # Determine effective number of workers for this batch
            workers = int(concorde_workers)
            if workers <= 0:
                import os

                workers = os.cpu_count() or 1
            workers = max(1, min(workers, cur_bs))
            if solver == "concorde":
                actions = _solve_with_concorde_batch(locs, n_jobs=workers).to(
                    td0.device
                )
            else:  # solver == "lkh"
                actions = _solve_with_lkh_batch(
                    locs, n_jobs=workers, exe=lkh_exe
                ).to(td0.device)

        base_final = base_env.get_reward(td0, actions)  # [B]

        # Roll the DenseRewardTSPEnv dynamics to capture masks and current node per-step
        from rl4co.envs.routing.tsp.env import DenseRewardTSPEnv

        env = DenseRewardTSPEnv(generator=gen)
        td = env.reset(td0)
        B, T = actions.shape
        for b in range(B):
            locs_b = td["locs"][b]  # [N,2]
            first_node_b = int(td["first_node"][b].item())
            acts_b = actions[b]
            # Collect sequences
            curr_nodes: List[int] = []
            masks: List[torch.Tensor] = []
            # Step through using env._step to update masks; keep batch dimension 1
            tdb = td[b:b+1].clone()
            for t in range(T):
                curr_nodes.append(int(tdb["current_node"][0].item()))
                masks.append(tdb["action_mask"][0].clone())
                # set action with batch dim
                act_t = acts_b[t].to(dtype=torch.long, device=tdb.device).unsqueeze(0)
                tdb.set("action", act_t)
                tdb = env._step(tdb)
            # Base step rewards (negative edge length)
            base_steps = _negative_edge_lengths(locs_b, acts_b)
            # Per-step undiscounted returns and simple advantages as generic
            # credit signals (independent of any particular Phi or baseline).
            # G_t = sum_{k>=t} r_k, gamma=1
            returns = torch.flip(torch.cumsum(torch.flip(base_steps, dims=[0]), dim=0), dims=[0])
            # Simple per-episode baseline: mean return over time
            baseline = returns.mean()
            advantages = returns - baseline
            episodes.append(
                {
                    "locs": locs_b.cpu(),
                    "first_node": first_node_b,
                    "actions": acts_b.cpu(),
                    "current_nodes": torch.tensor(curr_nodes, dtype=torch.long),
                    "action_masks": torch.stack(masks, dim=0).cpu(),
                    "base_step_reward": base_steps.cpu(),
                    "returns": returns.cpu(),
                    "advantages": advantages.cpu(),
                    "final_reward": float(base_final[b].item()),
                }
            )
        n_remaining -= B

    payload = {
        "episodes": episodes,
        "meta": {"num_loc": int(num_loc), "seed": int(seed)},
    }
    torch.save(payload, out_path)
    return out_path


def load_offline_trajectories(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out-path", type=str, required=True)
    p.add_argument("--num-episodes", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num-loc", type=int, default=None)
    p.add_argument("--ckpt", type=str, default=None, help="Optional POMO checkpoint to load for rollout")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--solver",
        type=str,
        default="pomo",
        choices=["pomo", "concorde", "lkh"],
        help=(
            "Trajectory generator: 'pomo' (policy rollout, optionally with --ckpt), "
            "'concorde' (optimal tours via pyconcorde; ignores --ckpt), "
            "or 'lkh' (external LKH solver; ignores --ckpt)."
        ),
    )
    p.add_argument(
        "--concorde-workers",
        type=int,
        default=0,
        help=(
            "Number of parallel worker processes when --solver is 'concorde' or 'lkh'. "
            "0 or <=0 means auto-detect (use up to all CPU cores)."
        ),
    )
    p.add_argument(
        "--lkh-exe",
        type=str,
        default="LKH",
        help=(
            "LKH executable name or path when --solver=lkh (default: 'LKH' in PATH)."
        ),
    )
    return p.parse_args()


def main():
    args = _parse_args()
    path = collect_tsp20_trajectories(
        out_path=args.out_path,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        seed=args.seed,
        num_loc=args.num_loc,
        ckpt_path=args.ckpt,
        device=args.device,
        solver=args.solver,
        concorde_workers=args.concorde_workers,
        lkh_exe=args.lkh_exe,
    )
    print(f"Saved offline trajectories to: {path}")


if __name__ == "__main__":
    main()
