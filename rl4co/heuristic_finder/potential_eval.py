from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import math
import re

import torch

from rl4co.envs.routing.tsp.pbrs_env import TSPStateView, InvariantTSPStateView


@dataclass
class PhiStats:
    mean_dphi: float
    std_dphi: float
    abs_dphi_q95: float
    step_shaping_ratio: float
    episode_shaping_ratio: float
    var_ratio_shaped_vs_base: float
    corr_dphi_future_cost: Optional[float]
    corr_phi0_final_reward: Optional[float]
    corr_sum_dphi_final_reward: Optional[float]


def _pearsonr(x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    if x.numel() < 2 or y.numel() < 2:
        return None
    x = x.flatten()
    y = y.flatten()
    xm = x - x.mean()
    ym = y - y.mean()
    xs = xm.std(unbiased=False)
    ys = ym.std(unbiased=False)
    if xs.item() == 0 or ys.item() == 0:
        return None
    r = (xm * ym).mean() / (xs * ys)
    return float(r.item())


def _build_state_view(locs: torch.Tensor, i: int, current_node: int, first_node: int, action_mask: torch.Tensor) -> TSPStateView:
    # Shapes match pbrs_env.TSPStateView expectations
    B = 1
    return TSPStateView(
        locs=locs.unsqueeze(0),
        i=torch.tensor([[i]], dtype=torch.int64),
        current_node=torch.tensor([current_node], dtype=torch.int64),
        first_node=torch.tensor([first_node], dtype=torch.int64),
        action_mask=action_mask.unsqueeze(0).to(torch.bool),
    )


def compute_phi_stats(
    phi_fn: Callable[[InvariantTSPStateView], torch.Tensor],
    trajectories: Dict[str, Any],
    gamma: float,
) -> PhiStats:
    episodes: List[Dict[str, Any]] = trajectories["episodes"]
    dphi_vals: List[float] = []
    r_vals: List[float] = []
    rp_vals: List[float] = []
    # For correlations
    dphi_all: List[float] = []
    future_cost_all: List[float] = []
    sum_dphi_per_ep: List[float] = []
    final_reward_per_ep: List[float] = []

    for ep in episodes:
        locs = ep["locs"].to(torch.float32)
        first_node = int(ep["first_node"])
        actions = ep["actions"].to(torch.long)
        current_nodes = ep["current_nodes"].to(torch.long)
        masks = ep["action_masks"].to(torch.bool)
        r_base = ep["base_step_reward"].to(torch.float32)  # negative edge length
        final_reward = float(ep["final_reward"])  # negative tour length
        T = actions.shape[0]

        # Precompute cumulative future cost (positive cost): we use -r_base as cost per step
        # future_cost[t] = sum_{k=t}^{T-1} (-r_k)
        cost_steps = (-r_base).flatten()
        future_cost = torch.flip(torch.cumsum(torch.flip(cost_steps, dims=[0]), dim=0), dims=[0])

        sum_dphi = 0.0
        for t in range(T):
            # before state
            sv_before = _build_state_view(locs, t, int(current_nodes[t].item()), first_node, masks[t])
            inv_before = InvariantTSPStateView(sv_before)
            phi_b = phi_fn(inv_before)
            if not isinstance(phi_b, torch.Tensor):
                phi_b = torch.as_tensor(phi_b, dtype=torch.float32)
            phi_b = torch.nan_to_num(phi_b, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
            if phi_b.dim() > 0:
                phi_b = phi_b.view(-1)[0]

            # after state (advance one step)
            next_i = t + 1
            next_curr = int(actions[t].item())
            # update mask: mark the chosen node as visited
            mask_next = masks[t].clone()
            mask_next[next_curr] = False
            sv_after = _build_state_view(locs, next_i, next_curr, first_node, mask_next)
            inv_after = InvariantTSPStateView(sv_after)
            phi_a = phi_fn(inv_after)
            if not isinstance(phi_a, torch.Tensor):
                phi_a = torch.as_tensor(phi_a, dtype=torch.float32)
            phi_a = torch.nan_to_num(phi_a, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
            if phi_a.dim() > 0:
                phi_a = phi_a.view(-1)[0]

            dphi = float((phi_a - phi_b).item())
            r = float(r_base[t].item())
            r_prime = r + float(gamma) * dphi
            dphi_vals.append(dphi)
            r_vals.append(r)
            rp_vals.append(r_prime)

            dphi_all.append(dphi)
            future_cost_all.append(float(future_cost[t].item()))
            sum_dphi += dphi

        sum_dphi_per_ep.append(sum_dphi)
        final_reward_per_ep.append(final_reward)

    if len(dphi_vals) == 0:
        return PhiStats(0.0, 0.0, 0.0, math.inf, math.inf, math.inf, None, None, None)

    dphi_t = torch.as_tensor(dphi_vals, dtype=torch.float32)
    r_t = torch.as_tensor(r_vals, dtype=torch.float32)
    rp_t = torch.as_tensor(rp_vals, dtype=torch.float32)

    mean_dphi = float(dphi_t.mean().item())
    std_dphi = float(dphi_t.std(unbiased=False).item())
    abs_dphi_q95 = float(torch.quantile(dphi_t.abs(), torch.tensor(0.95)).item())

    mean_abs_r = float(r_t.abs().mean().item()) if r_t.numel() > 0 else 0.0
    mean_abs_dphi = float(dphi_t.abs().mean().item())
    step_shaping_ratio = (abs(float(gamma)) * mean_abs_dphi / mean_abs_r) if mean_abs_r > 0 else math.inf

    # Episode-level ratios
    sum_dphi_t = torch.as_tensor(sum_dphi_per_ep, dtype=torch.float32)
    final_reward_t = torch.as_tensor(final_reward_per_ep, dtype=torch.float32)
    mean_abs_sum_dphi = float(sum_dphi_t.abs().mean().item())
    mean_abs_final = float(final_reward_t.abs().mean().item()) if final_reward_t.numel() > 0 else 0.0
    episode_shaping_ratio = (abs(float(gamma)) * mean_abs_sum_dphi / mean_abs_final) if mean_abs_final > 0 else math.inf

    # Variance ratio
    base_var = float(r_t.var(unbiased=False).item()) if r_t.numel() > 1 else 0.0
    shaped_var = float(rp_t.var(unbiased=False).item()) if rp_t.numel() > 1 else 0.0
    var_ratio_shaped_vs_base = shaped_var / base_var if base_var > 0 else math.inf

    # Correlations
    corr_dphi_future_cost = None
    if len(dphi_all) > 1 and len(future_cost_all) > 1:
        corr_dphi_future_cost = _pearsonr(torch.tensor(dphi_all), torch.tensor(future_cost_all))
    corr_phi0_final_reward = _pearsonr(torch.tensor([0.0]*len(final_reward_per_ep)), torch.tensor(final_reward_per_ep)) if final_reward_per_ep else None
    corr_sum_dphi_final_reward = _pearsonr(sum_dphi_t, final_reward_t) if final_reward_t.numel() > 1 else None

    return PhiStats(
        mean_dphi=mean_dphi,
        std_dphi=std_dphi,
        abs_dphi_q95=abs_dphi_q95,
        step_shaping_ratio=step_shaping_ratio,
        episode_shaping_ratio=episode_shaping_ratio,
        var_ratio_shaped_vs_base=var_ratio_shaped_vs_base,
        corr_dphi_future_cost=corr_dphi_future_cost,
        corr_phi0_final_reward=corr_phi0_final_reward,
        corr_sum_dphi_final_reward=corr_sum_dphi_final_reward,
    )


def _complexity_of_code(code: str) -> float:
    # Simple heuristic: count non-empty, non-comment lines and operators
    lines = [ln for ln in code.replace("\r\n", "\n").split("\n") if ln.strip() and not ln.strip().startswith("#")]
    n_lines = len(lines)
    body = "\n".join(lines)
    # Count occurrences of common ops and torch functions
    op_tokens = re.findall(r"[+\-*/]", body)
    torch_calls = re.findall(r"torch\.[a-zA-Z_]+", body)
    return float(n_lines + 0.5 * len(op_tokens) + 0.25 * len(torch_calls))


def cheap_score_phi(
    phi_fn: Callable[[InvariantTSPStateView], torch.Tensor],
    trajectories: Dict[str, Any],
    gamma: float,
    config: Dict[str, Any],
    code: Optional[str] = None,
) -> Tuple[float, PhiStats, float]:
    """Compute a cheap scalar score from offline stats with light regularization.

    Returns (score, stats, complexity).
    """
    stats = compute_phi_stats(phi_fn, trajectories, gamma)

    # Hard filters
    max_step = float(config.get("max_step_shaping_ratio", 10.0))
    max_ep = float(config.get("max_episode_shaping_ratio", 10.0))
    max_var = float(config.get("max_var_ratio_shaped_vs_base", 10.0))
    min_q95 = float(config.get("min_abs_dphi_q95", 1e-4))

    valid = True
    if not math.isfinite(stats.step_shaping_ratio) or stats.step_shaping_ratio > max_step:
        valid = False
    if not math.isfinite(stats.episode_shaping_ratio) or stats.episode_shaping_ratio > max_ep:
        valid = False
    if not math.isfinite(stats.var_ratio_shaped_vs_base) or stats.var_ratio_shaped_vs_base > max_var:
        valid = False
    if stats.abs_dphi_q95 < min_q95:
        valid = False

    # Base score components
    score = 0.0
    if not valid:
        score = -1e6
    else:
        # Rewarding reasonable correlation: prefer negative correlation with future cost
        corr_fc = stats.corr_dphi_future_cost
        if corr_fc is not None and math.isfinite(corr_fc):
            score += float(-corr_fc)  # more negative corr -> higher score

        # Penalize high variance ratio but softly within bounds
        if math.isfinite(stats.var_ratio_shaped_vs_base):
            score -= 0.1 * max(0.0, stats.var_ratio_shaped_vs_base - 1.0)

        # Penalize overly large step shaping ratio within allowed range
        if math.isfinite(stats.step_shaping_ratio):
            score -= 0.05 * max(0.0, stats.step_shaping_ratio - 1.0)

    # Complexity penalty
    complexity = _complexity_of_code(code or "") if code is not None else 0.0
    score -= float(config.get("complexity_penalty_alpha", 0.001)) * complexity

    return float(score), stats, float(complexity)

