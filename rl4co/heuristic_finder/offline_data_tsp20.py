from __future__ import annotations

"""
Offline trajectory collection for TSP to support cheap Level-1 PBRS potential evaluation.

This module collects trajectories using a baseline POMO model without PBRS and stores
the minimal state needed to evaluate arbitrary potential functions phi(s) offline.

Usage:
  python -m rl4co.heuristic_finder.offline_data_tsp20 \
    --out-path data/tsp20_offline_trajs.pt \
    --num-episodes 512 --batch-size 128 --seed 1234 [--ckpt <path>]

Output file format (torch .pt): a dict with keys:
  episodes: List[dict] where each dict contains
    - locs: Tensor [N,2]
    - first_node: int
    - actions: LongTensor [T]
    - current_nodes: LongTensor [T]   (node before taking action t)
    - action_masks: BoolTensor [T,N]
    - base_step_reward: Tensor [T]    (negative edge length, r(s,a))
    - final_reward: float             (negative tour length)
  meta: { num_loc, seed }

Note: We explicitly store base_step_reward as the negative edge length so it matches
the PBRS shaping convention used in DensePBRSTSPEnv.
"""

import argparse
from typing import List, Dict, Any, Optional

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


def collect_tsp20_trajectories(
    out_path: str,
    num_episodes: int = 512,
    batch_size: int = 128,
    seed: int = 1234,
    num_loc: Optional[int] = None,
    ckpt_path: Optional[str] = None,
    device: str = "cpu",
) -> str:
    """Collect offline trajectories with a baseline POMO policy.

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

    # Build policy and optionally load a checkpoint
    if ckpt_path:
        model = POMO.load_from_checkpoint(ckpt_path, env=base_env, load_baseline=False)
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
        # Safety: align td device with model parameters if needed
        try:
            pdev = next(model.parameters()).device
            if pdev != td0.device:
                td0 = td0.to(pdev)
        except Exception:
            pass
        with torch.no_grad():
            out = model.policy(td0, base_env, phase="val", return_actions=True, decode_type="greedy")
            actions = out["actions"]  # [B, T]
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
            # Step through using env._step to update masks
            tdb = TensorDict({k: v[b].clone() for k, v in td.items()}, batch_size=[])
            for t in range(T):
                curr_nodes.append(int(tdb["current_node"].item()))
                masks.append(tdb["action_mask"].clone())
                tdb.set("action", acts_b[t])
                tdb = env._step(tdb)
            # Base step rewards (negative edge length)
            base_steps = _negative_edge_lengths(locs_b, acts_b)
            episodes.append(
                {
                    "locs": locs_b.cpu(),
                    "first_node": first_node_b,
                    "actions": acts_b.cpu(),
                    "current_nodes": torch.tensor(curr_nodes, dtype=torch.long),
                    "action_masks": torch.stack(masks, dim=0).cpu(),
                    "base_step_reward": base_steps.cpu(),
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
    )
    print(f"Saved offline trajectories to: {path}")


if __name__ == "__main__":
    main()
