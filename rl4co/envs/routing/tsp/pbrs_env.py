from dataclasses import dataclass
from typing import Callable
import os

import torch

from tensordict.tensordict import TensorDict

from rl4co.envs.routing.tsp.env import DenseRewardTSPEnv
from rl4co.utils.ops import gather_by_index


@dataclass
class TSPStateView:
    """
    Lightweight feature view of TSP environment state for potential functions.

    All tensors are shaped as [batch, ...] on the same device as the env state.
    """

    # Static
    locs: torch.Tensor  # [batch, N, 2]
    # Dynamics
    i: torch.Tensor  # [batch, 1], current step index
    current_node: torch.Tensor  # [batch]
    first_node: torch.Tensor  # [batch]
    action_mask: torch.Tensor  # [batch, N], True if not yet visited

    def num_nodes(self) -> int:
        return self.locs.shape[-2]

    def num_remaining(self) -> torch.Tensor:
        return self.action_mask.sum(dim=-1)  # [batch]

    def remaining_ratio(self) -> torch.Tensor:
        """Fraction of nodes remaining to visit in [0,1], shape [batch, 1]."""
        n = float(self.num_nodes())
        return (self.num_remaining().float() / n).unsqueeze(-1)

    def step_ratio(self) -> torch.Tensor:
        """Normalized step index i/N in [0,1], shape [batch, 1]."""
        n = float(self.num_nodes())
        return (self.i.float() / n)

    def current_loc(self) -> torch.Tensor:
        return gather_by_index(self.locs, self.current_node)  # [batch, 2]

    def start_loc(self) -> torch.Tensor:
        """Coordinates of the first node in the tour [batch, 2]."""
        return gather_by_index(self.locs, self.first_node)

    def unvisited_locs(self) -> torch.Tensor:
        mask = self.action_mask
        # NaN-out visited nodes for convenience
        masked = torch.where(mask.unsqueeze(-1), self.locs, torch.nan)
        return masked  # [batch, N, 2] with NaNs for visited

    # -------- Node-count-invariant helpers (fixed-size scalars/vectors) --------
    def graph_scale(self) -> torch.Tensor:
        """A global length scale: diagonal of the bounding box per instance [batch, 1]."""
        # range over nodes for x and y
        x = self.locs[..., 0]
        y = self.locs[..., 1]
        dx = (x.max(dim=-1).values - x.min(dim=-1).values)
        dy = (y.max(dim=-1).values - y.min(dim=-1).values)
        scale = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-6).unsqueeze(-1)
        return scale  # [batch, 1]

    def distances_to_unvisited(self, normalize: bool = True) -> torch.Tensor:
        """Distances from current node to all unvisited nodes [batch, N].

        Visited positions are NaN. When `normalize=True`, divide by `graph_scale()`.
        """
        cur = self.current_loc()  # [B,2]
        dif = self.locs - cur.unsqueeze(1)  # [B,N,2]
        d = torch.linalg.norm(dif, dim=-1, ord=2)  # [B,N]
        if normalize:
            d = d / self.graph_scale()
        # NaN-out visited
        d = torch.where(self.action_mask, d, torch.nan)
        return d

    def nearest_unvisited_distance(self, normalize: bool = True) -> torch.Tensor:
        """Nearest distance to any unvisited node [batch, 1]; 0 if none remain."""
        d = self.distances_to_unvisited(normalize=normalize)  # [B,N] with NaNs
        # replace NaNs with +inf to take min
        d_inf = torch.nan_to_num(d, nan=float("inf"))
        mn = d_inf.min(dim=-1).values  # [B]
        mn = torch.where(torch.isinf(mn), torch.zeros_like(mn), mn)
        return mn.unsqueeze(-1)

    def k_nearest_unvisited(self, k: int = 3, normalize: bool = True) -> torch.Tensor:
        """Return k smallest distances to unvisited nodes sorted ascending [batch, k].

        If fewer than k nodes remain, pad with zeros at the end.
        """
        d = self.distances_to_unvisited(normalize=normalize)  # [B,N]
        d_inf = torch.nan_to_num(d, nan=float("inf"))
        vals, _ = torch.sort(d_inf, dim=-1)  # ascending
        topk = vals[..., :k]
        # replace inf (no available) with 0 for stability
        topk = torch.where(torch.isinf(topk), torch.zeros_like(topk), topk)
        # pad if N < k (should not happen for typical TSP), keep interface robust
        if topk.shape[-1] < k:
            pad = k - topk.shape[-1]
            topk = torch.nn.functional.pad(topk, (0, pad), value=0.0)
        return topk  # [B,k]

    def centroid_unvisited(self) -> torch.Tensor:
        """Centroid of unvisited nodes [batch, 2]; if none remain, returns current_loc()."""
        mask = self.action_mask.float()  # [B,N]
        w = mask.unsqueeze(-1)  # [B,N,1]
        sum_w = mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # [B,1]
        mean = (self.locs * w).nan_to_num(0.0).sum(dim=-2) / sum_w  # [B,2]
        # if none remain, fall back to current_loc
        none_left = (mask.sum(dim=-1) == 0).unsqueeze(-1)
        return torch.where(none_left, self.current_loc(), mean)

    def distance_to_centroid(self, normalize: bool = True) -> torch.Tensor:
        """Distance from current node to centroid of unvisited [batch, 1]."""
        cur = self.current_loc()
        cen = self.centroid_unvisited()
        d = torch.linalg.norm(cur - cen, dim=-1, ord=2).unsqueeze(-1)
        if normalize:
            d = d / self.graph_scale()
        return d

    def distance_to_start(self, normalize: bool = True) -> torch.Tensor:
        """Distance from current node to start node [batch, 1]."""
        cur = self.current_loc()
        st = self.start_loc()
        d = torch.linalg.norm(cur - st, dim=-1, ord=2).unsqueeze(-1)
        if normalize:
            d = d / self.graph_scale()
        return d


class DensePBRSTSPEnv(DenseRewardTSPEnv):
    """
    Dense step-reward TSP environment with PBRS shaping.

    Adds a potential-based reward shaping term to the dense per-step reward:
        r'_t = r_t + gamma * (Phi(s_{t+1}) - Phi(s_t))

    Important: here the base per-step reward r_t is the NEGATIVE edge length so that
    summing step rewards aligns with the original objective (reward = -tour length).
    get_reward(td, actions) remains the original objective (negative tour length),
    ensuring evaluation stays unbiased; shaping only affects step rewards exposed during stepping.
    """

    def __init__(
        self,
        potential_fn: Callable[[TSPStateView], torch.Tensor],
        gamma: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._potential_fn = potential_fn
        self._gamma = gamma
        # Optional logging controls via env vars
        # PBRS_LOG_PHI=1 enables logging; modes: first|stats|all; PBRS_LOG_PHI_EVERY for throttling
        self._log_phi_enabled = os.environ.get("PBRS_LOG_PHI", "0") not in ("0", "", "false", "False")
        self._log_phi_mode = os.environ.get("PBRS_LOG_PHI_MODE", "first")
        try:
            self._log_phi_every = max(1, int(os.environ.get("PBRS_LOG_PHI_EVERY", "1")))
        except Exception:
            self._log_phi_every = 1

    @staticmethod
    def _build_state_view(td: TensorDict) -> TSPStateView:
        # td keys as defined in DenseRewardTSPEnv
        return TSPStateView(
            locs=td["locs"],
            i=td["i"],
            current_node=td["current_node"],
            first_node=td["first_node"],
            action_mask=td["action_mask"],
        )

    def _step(self, td: TensorDict) -> TensorDict:
        # state view BEFORE transition
        sv_before = self._build_state_view(td)
        phi_before = self._safe_phi(sv_before)
        # Optional Phi logging (before transition)
        if self._log_phi_enabled:
            # step index before transition
            try:
                step_idx = int(td["i"].view(-1)[0].item())
            except Exception:
                step_idx = 0
            # Throttle by PBRS_LOG_PHI_EVERY
            if (step_idx % self._log_phi_every) == 0:
                if self._log_phi_mode == "all":
                    vals = phi_before.squeeze(-1).detach().cpu().tolist()
                    print(f"[PBRS] step {step_idx}: Phi(s)={vals}")
                elif self._log_phi_mode == "stats":
                    pb = phi_before.squeeze(-1)
                    m = float(pb.mean().item())
                    s = float(pb.std(unbiased=False).item())
                    mn = float(pb.min().item())
                    mx = float(pb.max().item())
                    print(f"[PBRS] step {step_idx}: Phi(s) stats mean={m:.6f} std={s:.6f} min={mn:.6f} max={mx:.6f}")
                else:  # first
                    val0 = float(phi_before.view(-1)[0].item())
                    print(f"[PBRS] step {step_idx}: Phi(s)={val0:.6f}")

        # base dense step update from parent (computes last->current edge length)
        last_node = td["current_node"].clone()
        current_node = td["action"]
        first_node = current_node if td["i"].all() == 0 else td["first_node"]

        available = td["action_mask"].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )
        done = torch.sum(available, dim=-1) == 0

        last_node_loc = gather_by_index(td["locs"], last_node)
        curr_node_loc = gather_by_index(td["locs"], current_node)
        # Use negative edge length as base step reward to align with the
        # environment's objective convention (reward = -tour length)
        base_reward = -torch.linalg.norm(
            last_node_loc - curr_node_loc, dim=-1, ord=2
        )[:, None]

        td_next = td.clone()
        td_next.update(
            {
                "first_node": first_node,
                "current_node": current_node,
                "i": td["i"] + 1,
                "action_mask": available,
            }
        )

        # state view AFTER transition
        sv_after = self._build_state_view(td_next)
        phi_after = self._safe_phi(sv_after)

        shaped = base_reward + self._gamma * (phi_after - phi_before)

        td_next.set("reward", shaped)
        td_next.set("done", done)
        return td_next

    def _safe_phi(self, state: TSPStateView) -> torch.Tensor:
        """Evaluate potential with basic safety: return zeros on failure.

        Returns a column tensor [batch, 1]
        """
        try:
            inv = InvariantTSPStateView(state)
            val = self._potential_fn(inv)
            if not isinstance(val, torch.Tensor):
                val = torch.as_tensor(val, device=state.locs.device, dtype=torch.float32)
            if val.dim() == 1:
                val = val.unsqueeze(-1)
            val = val.to(dtype=torch.float32, device=state.locs.device)
            # sanitize numerical issues from user code
            val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
            val = torch.clamp(val, min=-1e3, max=1e3)
            return val
        except Exception:
            # Fallback to zero potential
            b = state.locs.shape[0]
        return torch.zeros((b, 1), device=state.locs.device, dtype=torch.float32)


class InvariantTSPStateView:
    """
    Restricted state view exposing only node-count-invariant helpers to potential functions.

    Methods mirror a subset of TSPStateView that return fixed-size tensors independent of N.
    """

    def __init__(self, base: TSPStateView):
        self._base = base

    # Progress scalars
    def remaining_ratio(self) -> torch.Tensor:
        return self._base.remaining_ratio()

    def visited_ratio(self) -> torch.Tensor:
        return self._base.visited_ratio()

    def step_ratio(self) -> torch.Tensor:
        return self._base.step_ratio()

    # Geometry scalars/vectors (fixed size)
    def graph_scale(self) -> torch.Tensor:
        return self._base.graph_scale()

    def nearest_unvisited_distance(self, normalize: bool = True) -> torch.Tensor:
        return self._base.nearest_unvisited_distance(normalize=normalize)

    def k_nearest_unvisited(self, k: int = 3, normalize: bool = True) -> torch.Tensor:
        return self._base.k_nearest_unvisited(k=k, normalize=normalize)

    def k_farthest_unvisited(self, k: int = 3, normalize: bool = True) -> torch.Tensor:
        return self._base.k_farthest_unvisited(k=k, normalize=normalize)

    def mean_unvisited_distance(self, normalize: bool = True) -> torch.Tensor:
        return self._base.mean_unvisited_distance(normalize=normalize)

    def max_unvisited_distance(self, normalize: bool = True) -> torch.Tensor:
        return self._base.max_unvisited_distance(normalize=normalize)

    def std_unvisited_distance(self, normalize: bool = True) -> torch.Tensor:
        return self._base.std_unvisited_distance(normalize=normalize)

    def centroid_unvisited(self) -> torch.Tensor:
        return self._base.centroid_unvisited()

    def distance_to_centroid(self, normalize: bool = True) -> torch.Tensor:
        return self._base.distance_to_centroid(normalize=normalize)

    def distance_to_start(self, normalize: bool = True) -> torch.Tensor:
        return self._base.distance_to_start(normalize=normalize)

    # Optional fixed-size references
    def current_loc(self) -> torch.Tensor:
        return self._base.current_loc()

    def start_loc(self) -> torch.Tensor:
        return self._base.start_loc()
