from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Callable, List, Tuple

import torch

from rl4co.envs.routing.tsp.generator import TSPGenerator
from rl4co.envs.routing.tsp.pbrs_env import TSPStateView, InvariantTSPStateView


PhiFn = Callable[[InvariantTSPStateView], torch.Tensor]


def _load_phi(phi_path: Path) -> PhiFn:
    """Load `phi(state)` from a Python file, ensuring `torch` is available."""
    if not phi_path.is_file():
        raise FileNotFoundError(f"Phi file not found: {phi_path}")

    spec = importlib.util.spec_from_file_location("phi_module", str(phi_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {phi_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "phi"):
        raise AttributeError(f"{phi_path} does not define a `phi(state)` function")

    # Ensure global `torch` is visible inside phi
    if not hasattr(module, "torch"):
        module.torch = torch  # type: ignore[attr-defined]

    phi_fn = getattr(module, "phi")
    if not callable(phi_fn):
        raise TypeError("Loaded `phi` is not callable")
    return phi_fn


def _build_state_view_single(
    locs: torch.Tensor,
    tour_prefix: List[int],
    current_node: int,
    first_node: int,
    action_mask: torch.Tensor,
) -> InvariantTSPStateView:
    """Construct an InvariantTSPStateView for a single TSP instance."""
    device = locs.device
    N = locs.size(0)

    # Batch dimension = 1
    locs_b = locs.unsqueeze(0)  # [1, N, 2]

    k = len(tour_prefix)
    path_prefix = torch.full((1, N), -1, dtype=torch.long, device=device)
    if k > 0:
        path_prefix[0, :k] = torch.tensor(tour_prefix, dtype=torch.long, device=device)
    path_length = torch.tensor([[k]], dtype=torch.long, device=device)

    state = TSPStateView(
        locs=locs_b,
        i=torch.tensor([[k]], dtype=torch.int64, device=device),
        current_node=torch.tensor([current_node], dtype=torch.int64, device=device),
        first_node=torch.tensor([first_node], dtype=torch.int64, device=device),
        action_mask=action_mask.unsqueeze(0).to(device=device, dtype=torch.bool),
        path_prefix=path_prefix,
        path_length=path_length,
    )
    return InvariantTSPStateView(state)


def _phi_value(phi_fn: PhiFn, state: InvariantTSPStateView) -> float:
    """Evaluate phi on a single-instance state and return a scalar."""
    with torch.no_grad():
        val = phi_fn(state)
    if not isinstance(val, torch.Tensor):
        val = torch.as_tensor(val, dtype=torch.float32)
    val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
    if val.dim() > 1:
        val = val.view(-1)
    return float(val.flatten()[0].item())


def greedy_tour_single(
    phi_fn: PhiFn,
    locs: torch.Tensor,
    start_node: int = 0,
) -> Tuple[float, torch.Tensor]:
    """Run phi-greedy decoding on a single TSP instance.

    At each step, among all unvisited nodes j, choose the one minimizing:
        immediate_distance(last, j) + phi(state_after_visiting_j)

    Returns:
        (total_tour_length, tour_indices_tensor [N])
    """
    device = locs.device
    N = locs.size(0)

    # Visited path prefix (as list of node indices)
    tour: List[int] = [start_node]
    # Unvisited mask
    action_mask = torch.ones(N, dtype=torch.bool, device=device)
    action_mask[start_node] = False

    total_cost = 0.0

    while len(tour) < N:
        last = tour[-1]
        unvisited = torch.nonzero(action_mask, as_tuple=False).view(-1)
        if unvisited.numel() == 0:
            break

        best_node = int(unvisited[0].item())
        best_score: float | None = None

        for j_t in unvisited:
            j = int(j_t.item())

            # Hypothetical next tour prefix and mask
            cand_tour = tour + [j]
            cand_mask = action_mask.clone()
            cand_mask[j] = False

            inv_state = _build_state_view_single(
                locs=locs,
                tour_prefix=cand_tour,
                current_node=j,
                first_node=start_node,
                action_mask=cand_mask,
            )
            future = _phi_value(phi_fn, inv_state)

            # Immediate Euclidean distance last -> j
            edge = torch.linalg.norm(locs[last] - locs[j], ord=2).item()
            score = edge + future

            if best_score is None or score < best_score:
                best_score = score
                best_node = j

        # Apply best move
        edge = torch.linalg.norm(locs[tour[-1]] - locs[best_node], ord=2).item()
        total_cost += edge
        tour.append(best_node)
        action_mask[best_node] = False

    # Close tour: last -> start
    last = tour[-1]
    total_cost += torch.linalg.norm(locs[last] - locs[start_node], ord=2).item()

    return total_cost, torch.tensor(tour, dtype=torch.long, device=device)


def evaluate_phi_greedy(
    phi_fn: PhiFn,
    num_instances: int = 1000,
    num_loc: int = 20,
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 1234,
) -> None:
    """Evaluate phi by greedy decoding on random TSP instances."""
    torch.manual_seed(seed)
    device_t = torch.device(device)

    generator = TSPGenerator(num_loc=num_loc)

    lengths: List[float] = []
    remaining = num_instances

    while remaining > 0:
        cur_bs = min(batch_size, remaining)
        td = generator(cur_bs)
        locs_batch = td["locs"].to(device_t)  # [B, N, 2]

        for b in range(cur_bs):
            cost, _ = greedy_tour_single(phi_fn, locs_batch[b])
            lengths.append(cost)

        remaining -= cur_bs

    lengths_t = torch.tensor(lengths, dtype=torch.float32)
    mean = float(lengths_t.mean().item())
    std = float(lengths_t.std(unbiased=False).item())
    min_v = float(lengths_t.min().item())
    max_v = float(lengths_t.max().item())

    print(f"Evaluated {len(lengths)} TSP instances (N={num_loc}).")
    print(f"Greedy-phi tour length: mean={mean:.4f}, std={std:.4f}, "
          f"min={min_v:.4f}, max={max_v:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate phi_best.py via greedy decoding on random TSP instances.",
    )
    parser.add_argument(
        "--phi-file",
        type=str,
        default="phi_best.py",
        help="Path to a Python file defining `phi(state)` (default: phi_best.py at project root).",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=1000,
        help="Number of random TSP instances to evaluate.",
    )
    parser.add_argument(
        "--num-loc",
        type=int,
        default=20,
        help="Number of nodes per TSP instance.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for generating random instances.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (e.g., cpu or cuda).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve phi path relative to project root if needed
    phi_path = Path(args.phi_file)
    if not phi_path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        phi_path = project_root / phi_path

    phi_fn = _load_phi(phi_path)
    evaluate_phi_greedy(
        phi_fn=phi_fn,
        num_instances=args.num_instances,
        num_loc=args.num_loc,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

