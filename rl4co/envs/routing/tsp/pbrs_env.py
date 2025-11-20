from dataclasses import dataclass
from typing import Callable, Optional
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
    # Optional: prefix of visited path indices and its current length
    path_prefix: Optional[torch.Tensor] = None  # [batch, N] long, -1 padded
    path_length: Optional[torch.Tensor] = None  # [batch, 1] long

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

    def visited_ratio(self) -> torch.Tensor:
        """Fraction of nodes already visited in [0,1], shape [batch, 1]."""
        n = float(self.num_nodes())
        visited = (~self.action_mask).sum(dim=-1).float()
        return (visited / n).unsqueeze(-1)

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

    def visited_mask(self) -> torch.Tensor:
        """Boolean mask of visited nodes [batch, N]."""
        return ~self.action_mask

    def unvisited_mask(self) -> torch.Tensor:
        """Boolean mask of unvisited nodes [batch, N] (alias of action_mask)."""
        return self.action_mask

    def current_node_index(self) -> torch.Tensor:
        """Index of current node [batch]."""
        return self.current_node

    def first_node_index(self) -> torch.Tensor:
        """Index of first node [batch]."""
        return self.first_node

    # New helpers
    def all_node_coords(self) -> torch.Tensor:
        """Return coordinates of all nodes [batch, N, 2]."""
        return self.locs

    def partial_path_indices(self) -> torch.Tensor:
        """Return the indices of the already visited path.

        If a running path buffer exists, return shape [batch, N] long, filled by visit order
        and padded with -1. Otherwise, derive from visited_mask (unordered) and pad.
        """
        device = self.locs.device
        B = self.locs.shape[0]
        N = self.locs.shape[-2]
        out = torch.full((B, N), -1, dtype=torch.long, device=device)
        if self.path_prefix is not None:
            try:
                src = self.path_prefix.to(device=device, dtype=torch.long)
                # ensure shape [B, N]
                if src.dim() == 1:
                    src = src.unsqueeze(0)
                K = min(src.shape[-1], N)
                out[:, :K] = src[:, :K]
                return out
            except Exception:
                pass
        vm = self.visited_mask()
        for b in range(B):
            idx = torch.nonzero(vm[b], as_tuple=False).flatten()
            k = min(idx.numel(), N)
            if k > 0:
                out[b, :k] = idx[:k]
        return out

    # -------- Node-count-invariant helpers (fixed-size scalars/vectors) --------
    # def graph_scale(self) -> torch.Tensor:
    #     """A global length scale: diagonal of the bounding box per instance [batch, 1]."""
    #     # range over nodes for x and y
    #     x = self.locs[..., 0]
    #     y = self.locs[..., 1]
    #     dx = (x.max(dim=-1).values - x.min(dim=-1).values)
    #     dy = (y.max(dim=-1).values - y.min(dim=-1).values)
    #     scale = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-6).unsqueeze(-1)
    #     return scale  # [batch, 1]

    # def distances_to_unvisited(self, normalize: bool = True) -> torch.Tensor:
    #     """Distances from current node to all unvisited nodes [batch, N].

    #     Visited positions are NaN. When `normalize=True`, divide by `graph_scale()`.
    #     """
    #     cur = self.current_loc()  # [B,2]
    #     dif = self.locs - cur.unsqueeze(1)  # [B,N,2]
    #     d = torch.linalg.norm(dif, dim=-1, ord=2)  # [B,N]
    #     if normalize:
    #         d = d / self.graph_scale()
    #     # NaN-out visited
    #     d = torch.where(self.action_mask, d, torch.nan)
    #     return d

    # def distances_from_current(self, normalize: bool = True) -> torch.Tensor:
    #     """Distances from current node to all nodes [batch, N] (no NaNs)."""
    #     cur = self.current_loc()  # [B,2]
    #     dif = self.locs - cur.unsqueeze(1)  # [B,N,2]
    #     d = torch.linalg.norm(dif, dim=-1, ord=2)  # [B,N]
    #     if normalize:
    #         d = d / self.graph_scale()
    #     return d

    def distance_matrix(self, normalize: bool = False) -> torch.Tensor:
        """Full pairwise distance matrix [batch, N, N]. Diagonal is 0.

        The `normalize` argument is accepted for backward compatibility with
        some heuristic code, but is currently ignored (no rescaling).
        """
        locs = self.locs  # [B,N,2]
        dif = locs.unsqueeze(-3) - locs.unsqueeze(-2)  # [B,N,N,2]
        d = torch.linalg.norm(dif, dim=-1, ord=2)  # [B,N,N]
        # ensure exact zeros on diagonal (numerical stability)
        ii = torch.arange(locs.shape[-2], device=locs.device)
        d[..., ii, ii] = 0.0
        return d

    # def nearest_unvisited_distance(self, normalize: bool = True) -> torch.Tensor:
    #     """Nearest distance to any unvisited node [batch, 1]; 0 if none remain."""
    #     d = self.distances_to_unvisited(normalize=normalize)  # [B,N] with NaNs
    #     # replace NaNs with +inf to take min
    #     d_inf = torch.nan_to_num(d, nan=float("inf"))
    #     mn = d_inf.min(dim=-1).values  # [B]
    #     mn = torch.where(torch.isinf(mn), torch.zeros_like(mn), mn)
    #     return mn.unsqueeze(-1)

    # def k_nearest_unvisited(self, k: int = 3, normalize: bool = True) -> torch.Tensor:
    #     """Return k smallest distances to unvisited nodes sorted ascending [batch, k].

    #     If fewer than k nodes remain, pad with zeros at the end.
    #     """
    #     d = self.distances_to_unvisited(normalize=normalize)  # [B,N]
    #     d_inf = torch.nan_to_num(d, nan=float("inf"))
    #     vals, _ = torch.sort(d_inf, dim=-1)  # ascending
    #     topk = vals[..., :k]
    #     # replace inf (no available) with 0 for stability
    #     topk = torch.where(torch.isinf(topk), torch.zeros_like(topk), topk)
    #     # pad if N < k (should not happen for typical TSP), keep interface robust
    #     if topk.shape[-1] < k:
    #         pad = k - topk.shape[-1]
    #         topk = torch.nn.functional.pad(topk, (0, pad), value=0.0)
    #     return topk  # [B,k]

    # def centroid_unvisited(self) -> torch.Tensor:
    #     """Centroid of unvisited nodes [batch, 2]; if none remain, returns current_loc()."""
    #     mask = self.action_mask.float()  # [B,N]
    #     w = mask.unsqueeze(-1)  # [B,N,1]
    #     sum_w = mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)  # [B,1]
    #     mean = (self.locs * w).nan_to_num(0.0).sum(dim=-2) / sum_w  # [B,2]
    #     # if none remain, fall back to current_loc
    #     none_left = (mask.sum(dim=-1) == 0).unsqueeze(-1)
    #     return torch.where(none_left, self.current_loc(), mean)

    # def distance_to_centroid(self, normalize: bool = True) -> torch.Tensor:
    #     """Distance from current node to centroid of unvisited [batch, 1]."""
    #     cur = self.current_loc()
    #     cen = self.centroid_unvisited()
    #     d = torch.linalg.norm(cur - cen, dim=-1, ord=2).unsqueeze(-1)
    #     if normalize:
    #         d = d / self.graph_scale()
    #     return d

    # def distance_to_start(self, normalize: bool = True) -> torch.Tensor:
    #     """Distance from current node to start node [batch, 1]."""
    #     cur = self.current_loc()
    #     st = self.start_loc()
    #     d = torch.linalg.norm(cur - st, dim=-1, ord=2).unsqueeze(-1)
    #     if normalize:
    #         d = d / self.graph_scale()
    #     return d


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
        pure_shaping_terminal: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._potential_fn = potential_fn
        self._gamma = gamma
        # If True, use pure potential-based shaping on intermediate steps and
        # add the accumulated base reward only at terminal step. This keeps the
        # endpoint reward aligned with the original objective while making
        # intermediate rewards purely Delta-Phi.
        self._pure_shaping_terminal = bool(pure_shaping_terminal)
        # Optional Delta-Phi normalization / clipping controls via env vars
        # PBRS_CENTER_DPHI=1 centers Delta-Phi within batch
        # PBRS_NORM_DPHI=1 normalizes Delta-Phi std within batch
        # PBRS_DPHI_CLIP sets |Delta-Phi| clip (default 5.0)
        # PBRS_LAMBDA scales shaping strength (default 0.1)
        self._center_dphi = os.environ.get("PBRS_CENTER_DPHI", "0") not in (
            "0",
            "",
            "false",
            "False",
        )
        self._norm_dphi = os.environ.get("PBRS_NORM_DPHI", "0") not in (
            "0",
            "",
            "false",
            "False",
        )
        try:
            self._dphi_clip = float(os.environ.get("PBRS_DPHI_CLIP", "5.0"))
        except Exception:
            self._dphi_clip = 5.0
        try:
            self._lambda = float(os.environ.get("PBRS_LAMBDA", "0.1"))
        except Exception:
            self._lambda = 0.1
        self._lambda = max(0.0, float(self._lambda))
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
            path_prefix=td.get("path_prefix", None),
            path_length=td.get("path_length", None),
        )

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        td_out = super()._reset(td, batch_size)
        # Initialize path buffer
        device = td_out["locs"].device
        B = td_out["locs"].shape[0]
        N = td_out["locs"].shape[-2]
        path_prefix = torch.full((B, N), -1, dtype=torch.long, device=device)
        path_length = torch.zeros((B, 1), dtype=torch.long, device=device)
        td_out.set("path_prefix", path_prefix)
        td_out.set("path_length", path_length)
        if self._pure_shaping_terminal:
            # Track accumulated base reward so it can be added only at terminal step
            base_return = torch.zeros((B, 1), dtype=torch.float32, device=device)
            td_out.set("base_return", base_return)
        return td_out

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

        # Update path buffer (record chosen action at position equal to previous step index)
        try:
            B = td["locs"].shape[0]
            N = td["locs"].shape[-2]
            device = td.device
            # previous step index per instance
            k = td["i"].view(-1).to(torch.long)
            pp = td.get("path_prefix", torch.full((B, N), -1, dtype=torch.long, device=device)).clone()
            b_idx = torch.arange(pp.shape[0], device=pp.device)
            cur = current_node.to(torch.long)
            # clamp positions to valid range
            k_clamped = torch.clamp(k, 0, N - 1)
            pp[b_idx, k_clamped] = cur
            td_next.set("path_prefix", pp)
            td_next.set("path_length", torch.clamp(k_clamped + 1, max=N).view(-1, 1))
        except Exception:
            pass

        # state view AFTER transition
        sv_after = self._build_state_view(td_next)
        phi_after = self._safe_phi(sv_after)

        # Raw Delta-Phi
        dphi = phi_after - phi_before  # [B,1]

        # Batch-wise stabilization of Delta-Phi: center, normalize, and clip
        dphi_norm = dphi.view(-1)
        if self._center_dphi or self._norm_dphi:
            mean = dphi_norm.mean()
            if self._center_dphi:
                dphi_norm = dphi_norm - mean
            if self._norm_dphi:
                std = dphi_norm.std(unbiased=False)
                dphi_norm = dphi_norm / (std + 1e-6)
        if self._dphi_clip is not None and self._dphi_clip > 0.0:
            dphi_norm = torch.clamp(dphi_norm, -self._dphi_clip, self._dphi_clip)
        dphi_shaped = dphi_norm.view_as(dphi)

        # Effective shaping coefficient (small lambda for safety)
        gamma_eff = float(self._gamma) * float(self._lambda)

        if self._pure_shaping_terminal:
            # Accumulate base reward but only add it to the shaped reward at terminal step.
            base_return_prev = td.get("base_return", torch.zeros_like(base_reward))
            base_return_next = base_return_prev + base_reward
            td_next.set("base_return", base_return_next)

            shaped = gamma_eff * dphi_shaped
            if done.any():
                # Add endpoint (original) reward when trajectory finishes.
                done_f = done.unsqueeze(-1).to(base_reward.dtype)
                shaped = shaped + base_return_next * done_f
        else:
            shaped = base_reward + gamma_eff * dphi_shaped

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
    Restricted state view exposing helpers to potential functions.

    Primary goal is to encourage node-count-invariant designs; however, it also
    exposes some raw N-dependent features (e.g., masks and distance matrix) for
    expert aggregation. Ensure the final output remains broadcastable to [B,1]
    and numerically stable.
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
    # def graph_scale(self) -> torch.Tensor:
    #     return self._base.graph_scale()

    # def nearest_unvisited_distance(self, normalize: bool = True) -> torch.Tensor:
    #     return self._base.nearest_unvisited_distance(normalize=normalize)

    # def k_nearest_unvisited(self, k: int = 3, normalize: bool = True) -> torch.Tensor:
    #     return self._base.k_nearest_unvisited(k=k, normalize=normalize)

    # def k_farthest_unvisited(self, k: int = 3, normalize: bool = True) -> torch.Tensor:
    #     return self._base.k_farthest_unvisited(k=k, normalize=normalize)

    # def mean_unvisited_distance(self, normalize: bool = True) -> torch.Tensor:
    #     return self._base.mean_unvisited_distance(normalize=normalize)

    # def max_unvisited_distance(self, normalize: bool = True) -> torch.Tensor:
    #     return self._base.max_unvisited_distance(normalize=normalize)

    # def std_unvisited_distance(self, normalize: bool = True) -> torch.Tensor:
    #     return self._base.std_unvisited_distance(normalize=normalize)

    # def centroid_unvisited(self) -> torch.Tensor:
    #     return self._base.centroid_unvisited()

    # def distance_to_centroid(self, normalize: bool = True) -> torch.Tensor:
    #     return self._base.distance_to_centroid(normalize=normalize)

    # def distance_to_start(self, normalize: bool = True) -> torch.Tensor:
    #     return self._base.distance_to_start(normalize=normalize)

    # Optional fixed-size references
    def current_loc(self) -> torch.Tensor:
        return self._base.current_loc()

    def start_loc(self) -> torch.Tensor:
        return self._base.start_loc()

    # Raw N-dependent helpers (use with reductions to keep invariance)
    def action_mask(self) -> torch.Tensor:
        return self._base.action_mask

    def unvisited_mask(self) -> torch.Tensor:
        return self._base.unvisited_mask()

    def visited_mask(self) -> torch.Tensor:
        return self._base.visited_mask()

    def current_node_index(self) -> torch.Tensor:
        return self._base.current_node_index()

    def first_node_index(self) -> torch.Tensor:
        return self._base.first_node_index()

    # def distances_from_current(self, normalize: bool = True) -> torch.Tensor:
    #     return self._base.distances_from_current(normalize=normalize)

    def distance_matrix(self) -> torch.Tensor:
        return self._base.distance_matrix()

    # New helpers pass-through
    def all_node_coords(self) -> torch.Tensor:
        return self._base.all_node_coords()

    def partial_path_indices(self) -> torch.Tensor:
        return self._base.partial_path_indices()
