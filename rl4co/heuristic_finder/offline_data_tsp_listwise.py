from __future__ import annotations

"""
Listwise offline data collection for TSP with arbitrary num_loc using a
trained POMO solver.

For a given TSP size `num_loc`, this script:
  - Samples `n_instances` random TSP instances via `TSPGenerator(num_loc=...)`;
  - Runs a multi-start POMO policy (all starts kept, no best-start selection);
  - For every instance i, POMO start k, and decoding time step t, records a
    "partial solution state" containing structural information plus listwise
    preference labels derived from cost-to-go within that list.

Field conventions (saved via torch.save as a single dict):

Instance-level:
  - coords: FloatTensor [num_instances, num_nodes, 2]
      Node coordinates for each instance.
  - num_nodes: int (optional convenience; can be inferred from coords.shape[1])

State-level (per partial solution state):
  - visited_seq: LongTensor [num_states, max_seq_len]
      For each state, the sequence of visited node indices from the POMO start
      node up to the current last node. Positions beyond the current prefix are
      filled with -1. Here max_seq_len == num_nodes.
  - visited_mask: BoolTensor [num_states, num_nodes]
      Boolean mask where 1 indicates that the node has been visited in the
      current partial solution, 0 otherwise.
  - last_node: LongTensor [num_states]
      Index of the last visited node in the current partial solution.
  - prefix_length: FloatTensor [num_states]
      Length of the open path from the start node to `last_node`, i.e., the sum
      of edge lengths along the visited sequence so far (no closing edge).

Per-trajectory quantities repeated at state level:
  - total_tour_length: FloatTensor [num_states]
      Total length of the *complete* tour for the trajectory this state lies on,
      including the closing edge from last node back to first node.
  - cost_to_go: FloatTensor [num_states]
      Defined as total_tour_length - prefix_length. This is only used internally
      to derive listwise preference labels; downstream training does not need to
      regress it directly.

Identifiers and grouping:
  - instance_id: LongTensor [num_states]
      Global instance index in [0, num_instances_collected - 1].
  - time_step: LongTensor [num_states]
      Decoding step index for this partial solution. We use 1-based indexing:
      time_step = 1 after the first node is visited, ..., num_nodes at the
      final step when all nodes have been visited (but before closing the tour).
  - pomo_index: LongTensor [num_states]
      Index of the POMO multi-start trajectory for this instance, in
      [0, num_starts - 1].
  - list_id: LongTensor [num_states]
      Integer list identifier. States with the same list_id share the same
      (instance_id, time_step) pair, i.e., they form a list of POMO trajectories
      for the same instance at the same step.

Listwise preference labels (derived only from within-list rankings of cost_to_go):
  - rank_in_list: LongTensor [num_states]
      Rank of this state within its list when sorting by cost_to_go ascending.
      The best (smallest cost_to_go) has rank 0.
  - is_best: BoolTensor [num_states]
      Whether this state has the best (minimal) cost_to_go in its list.
  - preference_tier: LongTensor [num_states]
      Coarse preference tier per list, defined via rank percentiles:
        * tier 0: top 10% within the list (at least 1 state);
        * tier 1: next 30% (up to 40% cumulative; may be empty for small lists);
        * tier 2: remaining states.

The output is saved as:
  data = {
      "coords": coords,                   # [num_instances, num_nodes, 2]
      "visited_seq": visited_seq,         # [num_states, num_nodes]
      "visited_mask": visited_mask,       # [num_states, num_nodes]
      "last_node": last_node,             # [num_states]
      "prefix_length": prefix_length,     # [num_states]
      "total_tour_length": total_tour_length,
      "cost_to_go": cost_to_go,
      "instance_id": instance_id,
      "time_step": time_step,
      "pomo_index": pomo_index,
      "list_id": list_id,
      "rank_in_list": rank_in_list,
      "is_best": is_best,
      "preference_tier": preference_tier,
      "meta": {"num_loc": num_nodes, "seed": seed, "decode_mode": "..."},
  }

Usage:
  python -m rl4co.heuristic_finder.offline_data_tsp_listwise \
    --num-loc 20 \
    --n-instances 1000 \
    --batch-size 128 \
    --device cuda \
    --seed 0 \
    --checkpoint checkpoints/pomo_tsp20.ckpt \
    --out-dir data/tsp_listwise
"""

import argparse
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torch

from rl4co.envs.routing.tsp.env import TSPEnv, TSPGenerator
from rl4co.models.zoo.pomo import POMO
from rl4co.utils.ops import get_tour_length


def _compute_tour_length(locs: torch.Tensor, tour: torch.Tensor) -> float:
    """Compute total tour length including closing edge for a single instance.

    Args:
        locs: Tensor [num_nodes, 2]
        tour: LongTensor [num_nodes] with node indices in visit order
    Returns:
        Scalar float tour length.
    """
    ordered = locs[tour]  # [N, 2]
    length = get_tour_length(ordered.unsqueeze(0))[0]
    return float(length.item())


def _compute_prefix_lengths(locs: torch.Tensor, tour: torch.Tensor) -> torch.Tensor:
    """Compute per-step prefix path length for a single tour.

    Prefix length at step t (1-based) is the length of the open path from the
    start node to the node visited at position t, excluding any closing edge.
    At t=1, prefix_length=0 (no edges traversed yet).

    Args:
        locs: Tensor [num_nodes, 2]
        tour: LongTensor [num_nodes]
    Returns:
        Tensor [num_nodes] of prefix lengths (float32).
    """
    coords_seq = locs[tour]  # [T, 2]
    T = coords_seq.size(0)
    prefix = torch.zeros(T, dtype=torch.float32)
    if T <= 1:
        return prefix

    # Edge lengths between consecutive visited nodes (open path, no closing edge)
    edges = torch.linalg.norm(coords_seq[1:] - coords_seq[:-1], dim=-1, ord=2)  # [T-1]
    prefix[1:] = torch.cumsum(edges, dim=0)
    return prefix


def _decode_type_from_mode(mode: str) -> str:
    """Map a human read decode mode ('greedy'/'sampling') to a multistart decode_type."""
    m = mode.lower()
    if m in {"greedy", "multistart_greedy"}:
        return "multistart_greedy"
    if m in {"sampling", "multistart_sampling"}:
        return "multistart_sampling"
    raise ValueError(f"Unsupported decode_mode '{mode}', expected 'greedy' or 'sampling'.")


def collect_tsp_listwise(
    num_loc: int,
    n_instances: int,
    batch_size: int,
    device: str,
    seed: int,
    ckpt_path: str,
    out_dir: str,
    max_states: Optional[int] = None,
    step_stride: int = 1,
    decode_mode: str = "greedy",
    shard_id: int = 0,
) -> str:
    """Collect listwise offline data for TSP with arbitrary num_loc using POMO.

    Args:
        num_loc: Number of nodes in each TSP instance.
        n_instances: Number of TSP instances to generate.
        batch_size: Number of instances to process per batch.
        device: Device string, e.g., "cuda" or "cpu".
        seed: Global random seed.
        ckpt_path: Path to a trained POMO checkpoint compatible with num_loc.
        out_dir: Directory to write the output .pt file into.
        max_states: Optional cap on total number of partial states to store.
        step_stride: Keep only states where (time_step % step_stride == 0),
            plus always keep the final step time_step == num_loc. Must be >=1.
        decode_mode: "greedy" or "sampling" (internally mapped to
            "multistart_greedy" / "multistart_sampling").
        shard_id: Shard index for naming; purely cosmetic.

    Returns:
        The output file path written to.
    """
    if step_stride < 1:
        raise ValueError(f"step_stride must be >=1, got {step_stride}")

    L.seed_everything(seed, workers=True)

    # Environment and generator
    gen = TSPGenerator(num_loc=num_loc)
    base_env = TSPEnv(generator=gen, seed=seed)

    # Load trained POMO model
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"POMO checkpoint not found: {ckpt_path}")

    model: POMO = POMO.load_from_checkpoint(
        ckpt_path, env=base_env, load_baseline=False
    )
    model.eval()

    try:
        model.to(torch.device(device))
    except Exception:
        # If the requested device is not available, keep default device
        pass

    # Storage: instance-level
    coords_batches: List[torch.Tensor] = []

    # Storage: state-level
    visited_seq_list: List[torch.Tensor] = []
    visited_mask_list: List[torch.Tensor] = []
    last_node_list: List[int] = []
    prefix_len_list: List[float] = []
    total_tour_len_list: List[float] = []
    cost_to_go_list: List[float] = []
    instance_id_list: List[int] = []
    time_step_list: List[int] = []
    pomo_index_list: List[int] = []
    list_id_list: List[int] = []

    # List grouping and ranking helpers
    list_key_to_id: Dict[Tuple[int, int], int] = {}
    list_to_state_indices: Dict[int, List[int]] = defaultdict(list)
    next_list_id = 0

    # Global counters
    instances_generated = 0
    state_index = 0
    decode_type = _decode_type_from_mode(decode_mode)

    # Main instance generation loop
    while instances_generated < n_instances:
        remaining = n_instances - instances_generated
        cur_bs = min(batch_size, remaining)
        if cur_bs <= 0:
            break

        # Sample a batch of instances and reset env on the chosen device
        td_init = gen(batch_size=[cur_bs]).to(device)
        td0 = base_env.reset(td_init)

        # Align td device with model parameters if needed
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
                decode_type=decode_type,
                select_best=False,  # keep all POMO starts
            )

        actions = out["actions"]  # [B * num_starts, T]
        if actions is None:
            raise RuntimeError("POMO policy did not return actions.")

        # Move actions / locs to CPU for cheaper post-processing
        actions = actions.detach().cpu()
        locs_batch = td0["locs"].detach().cpu()  # [B, N, 2]

        B, num_nodes, _ = locs_batch.shape
        BS, T = actions.shape
        if T != num_nodes:
            raise RuntimeError(
                f"Decoded tour length T={T} does not match num_nodes={num_nodes}."
            )
        if BS % B != 0:
            raise RuntimeError(
                f"Decoded batch size {BS} is not a multiple of instance batch size {B}."
            )

        num_starts = BS // B

        # Append coords for these instances
        coords_batches.append(locs_batch)

        # Process each (instance, POMO start) trajectory
        for pomo_idx in range(num_starts):
            for b_local in range(B):
                flat_idx = pomo_idx * B + b_local
                tour = actions[flat_idx].long()  # [num_nodes]
                locs = locs_batch[b_local]  # [num_nodes, 2]

                global_instance_id = instances_generated + b_local

                # Trajectory-level quantities
                total_len = _compute_tour_length(locs, tour)
                prefix_vec = _compute_prefix_lengths(locs, tour)  # [num_nodes]

                # Build visited sequence and mask incrementally
                visited = torch.zeros(num_nodes, dtype=torch.bool)
                seq = torch.full(
                    (num_nodes,), -1, dtype=torch.long
                )  # pad with -1 beyond prefix

                for t in range(num_nodes):
                    node = int(tour[t].item())
                    seq[t] = node
                    visited[node] = True

                    time_step = t + 1  # 1-based convention

                    # Apply step stride: keep every k-th step plus the final step
                    if step_stride > 1 and (time_step % step_stride != 0) and (
                        time_step != num_nodes
                    ):
                        continue

                    if max_states is not None and state_index >= max_states:
                        break

                    prefix_len_val = float(prefix_vec[t].item())
                    cost_to_go_val = float(total_len - prefix_len_val)

                    visited_seq_list.append(seq.clone())
                    visited_mask_list.append(visited.clone())
                    last_node_list.append(node)
                    prefix_len_list.append(prefix_len_val)
                    total_tour_len_list.append(total_len)
                    cost_to_go_list.append(cost_to_go_val)
                    instance_id_list.append(global_instance_id)
                    time_step_list.append(time_step)
                    pomo_index_list.append(pomo_idx)

                    # Assign list_id for (instance_id, time_step)
                    key = (global_instance_id, time_step)
                    lid = list_key_to_id.get(key)
                    if lid is None:
                        lid = next_list_id
                        list_key_to_id[key] = lid
                        next_list_id += 1
                    list_id_list.append(lid)

                    list_to_state_indices[lid].append(state_index)
                    state_index += 1

                if max_states is not None and state_index >= max_states:
                    break

            if max_states is not None and state_index >= max_states:
                break

        instances_generated += B

        # Lightweight textual progress indicator (instances)
        try:
            print(
                f"[INFO] Collecting listwise TSP data: "
                f"{instances_generated}/{n_instances} instances",
                end="\r",
                flush=True,
            )
        except Exception:
            pass

        if max_states is not None and state_index >= max_states:
            break

    if not coords_batches:
        raise RuntimeError("No instances were generated; nothing to save.")

    coords = torch.cat(coords_batches, dim=0)  # [num_instances_collected, num_nodes, 2]
    num_instances_collected = coords.size(0)

    num_states = state_index
    if num_states == 0:
        raise RuntimeError("No states were collected; check max_states / step_stride.")

    # Convert lists to tensors
    visited_seq = torch.stack(visited_seq_list, dim=0)  # [S, N]
    visited_mask = torch.stack(visited_mask_list, dim=0)  # [S, N]
    last_node = torch.tensor(last_node_list, dtype=torch.long)
    prefix_length = torch.tensor(prefix_len_list, dtype=torch.float32)
    total_tour_length = torch.tensor(total_tour_len_list, dtype=torch.float32)
    cost_to_go = torch.tensor(cost_to_go_list, dtype=torch.float32)
    instance_id = torch.tensor(instance_id_list, dtype=torch.long)
    time_step = torch.tensor(time_step_list, dtype=torch.long)
    pomo_index = torch.tensor(pomo_index_list, dtype=torch.long)
    list_id = torch.tensor(list_id_list, dtype=torch.long)

    assert (
        visited_seq.size(0)
        == visited_mask.size(0)
        == last_node.size(0)
        == prefix_length.size(0)
        == total_tour_length.size(0)
        == cost_to_go.size(0)
        == instance_id.size(0)
        == time_step.size(0)
        == pomo_index.size(0)
        == list_id.size(0)
        == num_states
    ), "Inconsistent number of collected states."

    # Rank within each list (by cost_to_go ascending) and derive preference tiers
    rank_in_list = torch.empty(num_states, dtype=torch.long)
    is_best = torch.zeros(num_states, dtype=torch.bool)
    preference_tier = torch.full(num_states, 2, dtype=torch.long)  # default tier 2

    for lid, indices in list_to_state_indices.items():
        idx = torch.tensor(indices, dtype=torch.long)
        costs = cost_to_go[idx]
        order = torch.argsort(costs, dim=0)  # ascending

        # ranks within this list (0 is best)
        local_ranks = torch.empty_like(order)
        local_ranks[order] = torch.arange(order.numel(), dtype=torch.long)
        rank_in_list[idx] = local_ranks

        # best state(s) in this list
        if order.numel() > 0:
            best_global = idx[order[0]]
            is_best[best_global] = True

        # preference tiers via simple percentiles
        n = order.numel()
        if n == 0:
            continue
        # ensure at least one element in tier 0
        k0 = max(1, int(math.ceil(0.10 * n)))
        k1 = max(k0, int(math.ceil(0.40 * n)))  # up to 40% cumulative

        tier0_idx = idx[order[:k0]]
        preference_tier[tier0_idx] = 0

        if k1 > k0:
            tier1_idx = idx[order[k0:k1]]
            preference_tier[tier1_idx] = 1
        # remaining indices stay at tier 2

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"tsp_numloc{num_loc}_seed{seed}_shard{shard_id}.pt"
    )

    data: Dict[str, Any] = {
        "coords": coords,  # [num_instances, num_nodes, 2]
        "num_nodes": torch.tensor(int(num_nodes), dtype=torch.long),
        "visited_seq": visited_seq,  # [num_states, num_nodes]
        "visited_mask": visited_mask,
        "last_node": last_node,
        "prefix_length": prefix_length,
        "total_tour_length": total_tour_length,
        "cost_to_go": cost_to_go,
        "instance_id": instance_id,
        "time_step": time_step,
        "pomo_index": pomo_index,
        "list_id": list_id,
        "rank_in_list": rank_in_list,
        "is_best": is_best,
        "preference_tier": preference_tier,
        "meta": {
            "num_loc": int(num_loc),
            "num_nodes": int(num_nodes),
            "num_instances": int(num_instances_collected),
            "num_states": int(num_states),
            "seed": int(seed),
            "decode_mode": decode_type,
            "step_stride": int(step_stride),
            "max_states": None if max_states is None else int(max_states),
        },
    }

    torch.save(data, out_path)

    # Simple log summary
    avg_states_per_list = float(num_states) / max(1, len(list_key_to_id))
    print(
        "[INFO] Saved TSP listwise offline data:",
        f"path={out_path}, num_loc={num_loc}, "
        f"instances={num_instances_collected}, states={num_states}, "
        f"lists={len(list_key_to_id)}, avg_states_per_list={avg_states_per_list:.2f}",
    )

    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Collect listwise offline TSP data using a trained POMO solver "
            "for arbitrary num_loc."
        )
    )
    p.add_argument("--num-loc", type=int, required=True, help="TSP node count.")
    p.add_argument(
        "--n-instances",
        type=int,
        required=True,
        help="Number of TSP instances to generate.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of instances per environment batch.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device for POMO rollout (e.g., "cuda" or "cpu").',
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained POMO checkpoint compatible with num_loc.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to store the resulting .pt file.",
    )
    p.add_argument(
        "--max-states",
        type=int,
        default=None,
        help=(
            "Optional maximum number of partial states to collect. "
            "If set, collection stops once this cap is reached."
        ),
    )
    p.add_argument(
        "--step-stride",
        type=int,
        default=1,
        help=(
            "Keep only every k-th decoding step (1-based), plus the final step. "
            "E.g., step_stride=2 keeps time_step in {2,4,6,...,num_loc}."
        ),
    )
    p.add_argument(
        "--decode-mode",
        type=str,
        default="greedy",
        choices=["greedy", "sampling", "multistart_greedy", "multistart_sampling"],
        help=(
            "Decoding mode for POMO. 'greedy'/'sampling' are mapped to "
            "multistart variants internally."
        ),
    )
    p.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Shard index used only in the output filename.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    collect_tsp_listwise(
        num_loc=args.num_loc,
        n_instances=args.n_instances,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        ckpt_path=args.checkpoint,
        out_dir=args.out_dir,
        max_states=args.max_states,
        step_stride=args.step_stride,
        decode_mode=args.decode_mode,
        shard_id=args.shard_id,
    )


if __name__ == "__main__":
    main()
