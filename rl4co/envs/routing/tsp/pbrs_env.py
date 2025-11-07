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
        n = float(self.num_nodes())
        return (self.num_remaining().float() / n).unsqueeze(-1)

    def visited_ratio(self) -> torch.Tensor:
        return 1.0 - self.remaining_ratio()

    def step_ratio(self) -> torch.Tensor:
        n = float(self.num_nodes())
        return (self.i.float() / n)

    def current_loc(self) -> torch.Tensor:
        return gather_by_index(self.locs, self.current_node)  # [batch, 2]

    def start_loc(self) -> torch.Tensor:
        return gather_by_index(self.locs, self.first_node)  # [batch, 2]

    def unvisited_locs(self) -> torch.Tensor:
        mask = self.action_mask
        return torch.where(mask.unsqueeze(-1), self.locs, torch.nan)

    # -------- Node-count-invariant helpers --------
    def graph_scale(self) -> torch.Tensor:
        x = self.locs[..., 0]
        y = self.locs[..., 1]
        dx = (x.max(dim=-1).values - x.min(dim=-1).values)
        dy = (y.max(dim=-1).values - y.min(dim=-1).values)
        return torch.sqrt(dx * dx + dy * dy).clamp_min(1e-6).unsqueeze(-1)

    def graph_aspect_ratio(self) -> torch.Tensor:
        x = self.locs[..., 0]
        y = self.locs[..., 1]
        dx = (x.max(dim=-1).values - x.min(dim=-1).values)
        dy = (y.max(dim=-1).values - y.min(dim=-1).values)
        eps = torch.finfo(self.locs.dtype).eps
        return (dx / (dy + eps)).unsqueeze(-1)

    def distances_to_unvisited(self, normalize: bool = True) -> torch.Tensor:
        """Distances from current node to all unvisited nodes [batch, N].

        Visited positions are NaN. When `normalize=True`, divide by `graph_scale()`.
        """
        cur = self.current_loc()  # [B,2]
        dif = self.locs - cur.unsqueeze(1)  # [B,N,2]
        d = torch.linalg.norm(dif, dim=-1, ord=2)  # [B,N]
        if normalize:
            d = d / self.graph_scale()
        return torch.where(self.action_mask, d, torch.nan)

    def nearest_unvisited_distance(self, normalize: bool = True) -> torch.Tensor:
        d = self.distances_to_unvisited(normalize=normalize)
        d_inf = torch.nan_to_num(d, nan=float("inf"))
        mn = d_inf.min(dim=-1).values
        mn = torch.where(torch.isinf(mn), torch.zeros_like(mn), mn)
        return mn.unsqueeze(-1)

    def k_nearest_unvisited(self, k: int = 3, normalize: bool = True) -> torch.Tensor:
        d = self.distances_to_unvisited(normalize=normalize)
        d_inf = torch.nan_to_num(d, nan=float("inf"))
        vals, _ = torch.sort(d_inf, dim=-1)
        topk = vals[..., :k]
        topk = torch.where(torch.isinf(topk), torch.zeros_like(topk), topk)
        if topk.shape[-1] < k:
            pad = k - topk.shape[-1]
            topk = torch.nn.functional.pad(topk, (0, pad), value=0.0)
        return topk

    def k_farthest_unvisited(self, k: int = 3, normalize: bool = True) -> torch.Tensor:
        d = self.distances_to_unvisited(normalize=normalize)
        d_ninf = torch.nan_to_num(d, nan=float("-inf"))
        vals, _ = torch.sort(d_ninf, dim=-1, descending=True)
        topk = vals[..., :k]
        topk = torch.where(torch.isinf(topk), torch.zeros_like(topk), topk)
        if topk.shape[-1] < k:
            pad = k - topk.shape[-1]
            topk = torch.nn.functional.pad(topk, (0, pad), value=0.0)
        return topk

    def mean_unvisited_distance(self, normalize: bool = True) -> torch.Tensor:
        d = self.distances_to_unvisited(normalize=normalize)
        m = torch.nanmean(d, dim=-1)
        return torch.nan_to_num(m, nan=0.0).unsqueeze(-1)

    def max_unvisited_distance(self, normalize: bool = True) -> torch.Tensor:
        d = self.distances_to_unvisited(normalize=normalize)
        d_ninf = torch.nan_to_num(d, nan=float("-inf"))
        mx = d_ninf.max(dim=-1).values
        mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
        return mx.unsqueeze(-1)

    def std_unvisited_distance(self, normalize: bool = True) -> torch.Tensor:
        d = self.distances_to_unvisited(normalize=normalize)
        mean = torch.nanmean(d, dim=-1)
        mean2 = torch.nanmean(d * d, dim=-1)
        var = torch.clamp(mean2 - mean * mean, min=0.0)
        return torch.sqrt(var).unsqueeze(-1)

    def centroid_unvisited(self) -> torch.Tensor:
        mask = self.action_mask.float()
        w = mask.unsqueeze(-1)
        sum_w = mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        mean = (self.locs * w).nan_to_num(0.0).sum(dim=-2) / sum_w
        none_left = (mask.sum(dim=-1) == 0).unsqueeze(-1)
        return torch.where(none_left, self.current_loc(), mean)

    def vector_to_centroid(self) -> torch.Tensor:
        cur = self.current_loc()
        cen = self.centroid_unvisited()
        vec = cen - cur
        norm = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(1e-6)
        return vec / norm

    def distance_to_centroid(self, normalize: bool = True) -> torch.Tensor:
        cur = self.current_loc()
        cen = self.centroid_unvisited()
        d = torch.linalg.norm(cur - cen, dim=-1, ord=2).unsqueeze(-1)
        return d / self.graph_scale() if normalize else d

    def distance_to_start(self, normalize: bool = True) -> torch.Tensor:
        cur = self.current_loc()
        st = self.start_loc()
        d = torch.linalg.norm(cur - st, dim=-1, ord=2).unsqueeze(-1)
        return d / self.graph_scale() if normalize else d


class DensePBRSTSPEnv(DenseRewardTSPEnv):
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
        self._log_phi_enabled = os.environ.get("PBRS_LOG_PHI", "0") not in ("0", "", "false", "False")
        self._log_phi_mode = os.environ.get("PBRS_LOG_PHI_MODE", "first")
        try:
            self._log_phi_every = max(1, int(os.environ.get("PBRS_LOG_PHI_EVERY", "1")))
        except Exception:
            self._log_phi_every = 1
        # Delta-Phi normalization/centering (to stabilize shaping magnitude)
        self._center_dphi = os.environ.get("PBRS_CENTER_DPHI", "0") not in ("0", "", "false", "False")
        self._norm_dphi = os.environ.get("PBRS_NORM_DPHI", "0") not in ("0", "", "false", "False")
        # Optional Delta-Phi logging
        self._log_dphi_enabled = os.environ.get("PBRS_LOG_DPHI", "0") not in ("0", "", "false", "False")
        self._log_dphi_mode = os.environ.get("PBRS_LOG_DPHI_MODE", "stats")
        try:
            self._log_dphi_every = max(1, int(os.environ.get("PBRS_LOG_DPHI_EVERY", "1")))
        except Exception:
            self._log_dphi_every = 1

    @staticmethod
    def _build_state_view(td: TensorDict) -> TSPStateView:
        return TSPStateView(
            locs=td["locs"],
            i=td["i"],
            current_node=td["current_node"],
            first_node=td["first_node"],
            action_mask=td["action_mask"],
        )

    def _step(self, td: TensorDict) -> TensorDict:
        # BEFORE transition
        sv_before = self._build_state_view(td)
        phi_before = self._safe_phi(sv_before)

        last_node = td["current_node"].clone()
        current_node = td["action"]
        first_node = current_node if td["i"].all() == 0 else td["first_node"]

        available = td["action_mask"].scatter(
            -1, current_node.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )
        done = torch.sum(available, dim=-1) == 0

        last_node_loc = gather_by_index(td["locs"], last_node)
        curr_node_loc = gather_by_index(td["locs"], current_node)
        base_reward = -torch.linalg.norm(last_node_loc - curr_node_loc, dim=-1, ord=2)[:, None]

        td_next = td.clone()
        td_next.update(
            {
                "first_node": first_node,
                "current_node": current_node,
                "i": td["i"] + 1,
                "action_mask": available,
            }
        )

        # AFTER transition
        sv_after = self._build_state_view(td_next)
        phi_after = self._safe_phi(sv_after)

        dphi = (phi_after - phi_before)
        if self._center_dphi:
            dphi = dphi - dphi.mean(dim=0, keepdim=True)
        if self._norm_dphi:
            std = dphi.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            dphi = dphi / std

        if self._log_phi_enabled:
            try:
                step_idx = int(td["i"].view(-1)[0].item())
            except Exception:
                step_idx = 0
            if (step_idx % self._log_phi_every) == 0:
                if self._log_phi_mode == "stats":
                    pb = phi_before.squeeze(-1)
                    m = float(pb.mean().item())
                    s = float(pb.std(unbiased=False).item())
                    mn = float(pb.min().item())
                    mx = float(pb.max().item())
                    print(f"[PBRS] step {step_idx}: Phi(s) stats mean={m:.6f} std={s:.6f} min={mn:.6f} max={mx:.6f}")
                elif self._log_phi_mode == "all":
                    vals = phi_before.squeeze(-1).detach().cpu().tolist()
                    print(f"[PBRS] step {step_idx}: Phi(s)={vals}")
                else:
                    val0 = float(phi_before.view(-1)[0].item())
                    print(f"[PBRS] step {step_idx}: Phi(s)={val0:.6f}")

        if self._log_dphi_enabled:
            try:
                step_idx = int(td["i"].view(-1)[0].item())
            except Exception:
                step_idx = 0
            if (step_idx % self._log_dphi_every) == 0:
                dp = dphi.squeeze(-1)
                m = float(dp.mean().item())
                s = float(dp.std(unbiased=False).item())
                mn = float(dp.min().item())
                mx = float(dp.max().item())
                print(f"[PBRS] step {step_idx}: dPhi stats mean={m:.6f} std={s:.6f} min={mn:.6f} max={mx:.6f}")

        shaped = base_reward + self._gamma * dphi
        td_next.set("reward", shaped)
        td_next.set("done", done)
        return td_next

    def _safe_phi(self, state: TSPStateView) -> torch.Tensor:
        try:
            inv = InvariantTSPStateView(state)
            val = self._potential_fn(inv)
            if not isinstance(val, torch.Tensor):
                val = torch.as_tensor(val, device=state.locs.device, dtype=torch.float32)
            if val.dim() == 1:
                val = val.unsqueeze(-1)
            val = val.to(dtype=torch.float32, device=state.locs.device)
            val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
            val = torch.clamp(val, min=-1e3, max=1e3)
            return val
        except Exception:
            b = state.locs.shape[0]
            return torch.zeros((b, 1), device=state.locs.device, dtype=torch.float32)


class InvariantTSPStateView:
    """Restricted view exposing only node-count-invariant helpers to potential functions."""
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

    def graph_aspect_ratio(self) -> torch.Tensor:
        return self._base.graph_aspect_ratio()

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

    def vector_to_centroid(self) -> torch.Tensor:
        return self._base.vector_to_centroid()

    def distance_to_centroid(self, normalize: bool = True) -> torch.Tensor:
        return self._base.distance_to_centroid(normalize=normalize)

    def distance_to_start(self, normalize: bool = True) -> torch.Tensor:
        return self._base.distance_to_start(normalize=normalize)

    # Optional fixed-size references
    def current_loc(self) -> torch.Tensor:
        return self._base.current_loc()

    def start_loc(self) -> torch.Tensor:
        return self._base.start_loc()