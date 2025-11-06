from dataclasses import dataclass
from typing import Callable

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

    def current_loc(self) -> torch.Tensor:
        return gather_by_index(self.locs, self.current_node)  # [batch, 2]

    def unvisited_locs(self) -> torch.Tensor:
        mask = self.action_mask
        # NaN-out visited nodes for convenience
        masked = torch.where(mask.unsqueeze(-1), self.locs, torch.nan)
        return masked  # [batch, N, 2] with NaNs for visited


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
            val = self._potential_fn(state)
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
