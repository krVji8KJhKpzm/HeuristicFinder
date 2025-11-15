from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import math
import re
import sys

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


def _build_state_view(
    locs: torch.Tensor,
    i: int,
    current_node: int,
    first_node: int,
    action_mask: torch.Tensor,
    device: torch.device,
    ) -> TSPStateView:
    # Shapes match pbrs_env.TSPStateView expectations
    B = 1
    return TSPStateView(
        locs=locs.unsqueeze(0).to(device),
        i=torch.tensor([[i]], dtype=torch.int64, device=device),
        current_node=torch.tensor([current_node], dtype=torch.int64, device=device),
        first_node=torch.tensor([first_node], dtype=torch.int64, device=device),
        action_mask=action_mask.unsqueeze(0).to(torch.bool).to(device),
    )


def _build_batched_state_view(
    locs: torch.Tensor,           # [B, N, 2]
    i: torch.Tensor,              # [B]
    current_node: torch.Tensor,   # [B]
    first_node: torch.Tensor,     # [B]
    action_mask: torch.Tensor,    # [B, N]
    device: torch.device,
) -> TSPStateView:
    return TSPStateView(
        locs=locs.to(device=device),
        i=i.view(-1, 1).to(device=device, dtype=torch.int64),
        current_node=current_node.to(device=device, dtype=torch.int64),
        first_node=first_node.to(device=device, dtype=torch.int64),
        action_mask=action_mask.to(device=device, dtype=torch.bool),
    )


def compute_phi_stats(
    phi_fn: Callable[[InvariantTSPStateView], torch.Tensor],
    trajectories: Dict[str, Any],
    gamma: float,
    device: str | torch.device = "cpu",
    batch_states: Optional[int] = None,
) -> PhiStats:
    episodes: List[Dict[str, Any]] = trajectories["episodes"]
    device = torch.device(device)
    bs = int(batch_states) if batch_states else 4096

    # Accumulators
    dphi_chunks: List[torch.Tensor] = []  # step-level
    r_chunks: List[torch.Tensor] = []
    rp_chunks: List[torch.Tensor] = []
    future_cost_chunks: List[torch.Tensor] = []
    final_reward_per_ep: List[float] = []
    sum_dphi_per_ep: List[float] = [0.0 for _ in range(len(episodes))]
    phi0_vals: List[float] = []

    # State buffers for batched phi evaluation
    b_locs: List[torch.Tensor] = []
    b_i: List[int] = []
    b_curr: List[int] = []
    b_first: List[int] = []
    b_mask: List[torch.Tensor] = []
    a_locs: List[torch.Tensor] = []
    a_i: List[int] = []
    a_curr: List[int] = []
    a_first: List[int] = []
    a_mask: List[torch.Tensor] = []
    step_r: List[float] = []
    step_ep_idx: List[int] = []
    step_future_cost: List[float] = []

    def _flush():
        if not b_locs:
            return
        with torch.no_grad():
            locs_b = torch.stack(b_locs, dim=0).to(device)
            i_b = torch.tensor(b_i, dtype=torch.int64, device=device)
            cur_b = torch.tensor(b_curr, dtype=torch.int64, device=device)
            fir_b = torch.tensor(b_first, dtype=torch.int64, device=device)
            m_b = torch.stack(b_mask, dim=0).to(torch.bool).to(device)

            locs_a = torch.stack(a_locs, dim=0).to(device)
            i_a = torch.tensor(a_i, dtype=torch.int64, device=device)
            cur_a = torch.tensor(a_curr, dtype=torch.int64, device=device)
            fir_a = torch.tensor(a_first, dtype=torch.int64, device=device)
            m_a = torch.stack(a_mask, dim=0).to(torch.bool).to(device)

            sv_b = _build_batched_state_view(locs_b, i_b, cur_b, fir_b, m_b, device)
            sv_a = _build_batched_state_view(locs_a, i_a, cur_a, fir_a, m_a, device)
            inv_b = InvariantTSPStateView(sv_b)
            inv_a = InvariantTSPStateView(sv_a)
            try:
                phi_b = phi_fn(inv_b)
                phi_a = phi_fn(inv_a)
            except Exception as e:
                src = getattr(phi_fn, "_source_code", None)
                if src is not None:
                    print(
                        "[HeuristicFinder] Runtime error in phi(state) during compute_phi_stats; source code follows:",
                        file=sys.stderr,
                        flush=True,
                    )
                    try:
                        # Print a separator to make logs easier to scan
                        print("=" * 80, file=sys.stderr, flush=True)
                        print(src, file=sys.stderr, flush=True)
                        print("=" * 80, file=sys.stderr, flush=True)
                    except Exception:
                        pass
                raise
            if not isinstance(phi_b, torch.Tensor):
                phi_b = torch.as_tensor(phi_b, dtype=torch.float32, device=device)
            if not isinstance(phi_a, torch.Tensor):
                phi_a = torch.as_tensor(phi_a, dtype=torch.float32, device=device)
            phi_b = torch.nan_to_num(phi_b, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
            phi_a = torch.nan_to_num(phi_a, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
            if phi_b.dim() > 1:
                phi_b = phi_b.squeeze(-1)
            if phi_a.dim() > 1:
                phi_a = phi_a.squeeze(-1)
            dphi = (phi_a - phi_b).view(-1)

            r_batch = torch.tensor(step_r, dtype=torch.float32, device=device)
            rp_batch = r_batch + float(gamma) * dphi

            # Accumulate per-episode sums
            for idx, dv in zip(step_ep_idx, dphi.tolist()):
                sum_dphi_per_ep[idx] += float(dv)

            # Store chunks on CPU
            dphi_chunks.append(dphi.detach().to("cpu"))
            r_chunks.append(r_batch.detach().to("cpu"))
            rp_chunks.append(rp_batch.detach().to("cpu"))
            future_cost_chunks.append(torch.tensor(step_future_cost, dtype=torch.float32))

        # Clear buffers
        b_locs.clear(); b_i.clear(); b_curr.clear(); b_first.clear(); b_mask.clear()
        a_locs.clear(); a_i.clear(); a_curr.clear(); a_first.clear(); a_mask.clear()
        step_r.clear(); step_ep_idx.clear(); step_future_cost.clear()

    # Compute phi0 per episode (batched once)
    with torch.no_grad():
        if len(episodes) > 0:
            locs0 = torch.stack([ep["locs"].to(torch.float32) for ep in episodes], dim=0).to(device)
            i0 = torch.zeros((len(episodes),), dtype=torch.int64, device=device)
            curr0 = torch.stack([ep["current_nodes"][0].to(torch.long) for ep in episodes]).to(device)
            first0 = torch.tensor([int(ep["first_node"]) for ep in episodes], dtype=torch.int64, device=device)
            mask0 = torch.stack([ep["action_masks"][0].to(torch.bool) for ep in episodes], dim=0).to(device)
            sv0 = _build_batched_state_view(locs0, i0, curr0, first0, mask0, device)
            inv0 = InvariantTSPStateView(sv0)
            phi0 = phi_fn(inv0)
            if not isinstance(phi0, torch.Tensor):
                phi0 = torch.as_tensor(phi0, dtype=torch.float32, device=device)
            phi0 = torch.nan_to_num(phi0, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
            if phi0.dim() > 1:
                phi0 = phi0.squeeze(-1)
            phi0_vals = phi0.detach().to("cpu").view(-1).tolist()

    # Iterate episodes, fill buffers and flush in chunks
    for ep_idx, ep in enumerate(episodes):
        final_reward_per_ep.append(float(ep["final_reward"]))
        locs = ep["locs"].to(torch.float32)
        actions = ep["actions"].to(torch.long)
        current_nodes = ep["current_nodes"].to(torch.long)
        masks = ep["action_masks"].to(torch.bool)
        r_base = ep["base_step_reward"].to(torch.float32)
        T = actions.shape[0]
        # future cost per-step on CPU
        cost_steps = (-r_base).flatten()
        future_cost = torch.flip(torch.cumsum(torch.flip(cost_steps, dims=[0]), dim=0), dims=[0])

        for t in range(T):
            # before
            b_locs.append(locs)
            b_i.append(t)
            b_curr.append(int(current_nodes[t].item()))
            b_first.append(int(ep["first_node"]))
            b_mask.append(masks[t])
            # after
            next_i = t + 1
            next_curr = int(actions[t].item())
            mask_next = masks[t].clone()
            mask_next[next_curr] = False
            a_locs.append(locs)
            a_i.append(next_i)
            a_curr.append(next_curr)
            a_first.append(int(ep["first_node"]))
            a_mask.append(mask_next)
            # meta
            step_r.append(float(r_base[t].item()))
            step_ep_idx.append(ep_idx)
            step_future_cost.append(float(future_cost[t].item()))

            if len(b_locs) >= bs:
                _flush()

    # flush remainder
    _flush()

    if len(dphi_chunks) == 0:
        return PhiStats(0.0, 0.0, 0.0, math.inf, math.inf, math.inf, None, None, None)

    dphi_t = torch.cat(dphi_chunks, dim=0)
    r_t = torch.cat(r_chunks, dim=0)
    rp_t = torch.cat(rp_chunks, dim=0)
    future_cost_all = torch.cat(future_cost_chunks, dim=0)

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
    corr_dphi_future_cost = _pearsonr(dphi_t, future_cost_all) if dphi_t.numel() > 1 and future_cost_all.numel() > 1 else None
    corr_phi0_final_reward = None
    if len(phi0_vals) == len(final_reward_per_ep) and len(phi0_vals) > 1:
        corr_phi0_final_reward = _pearsonr(
            torch.tensor(phi0_vals, dtype=torch.float32),
            torch.tensor(final_reward_per_ep, dtype=torch.float32),
        )
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
    # Device selection for cheap eval
    dev = config.get("cheap_eval_device", "cpu")
    bs = config.get("cheap_eval_batch_states", None)
    stats = compute_phi_stats(phi_fn, trajectories, gamma, device=dev, batch_states=bs)

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


def mse_phi_vs_value(
    phi_fn: Callable[[InvariantTSPStateView], torch.Tensor],
    trajectories: Dict[str, Any],
    device: str | torch.device = "cpu",
    batch_states: Optional[int] = None,
    target: str = "future_cost",
) -> float:
    """Compute mean-squared error between Phi(s) and Monte Carlo value V(s).

    V(s_t) is taken as the future tour length from state s_t, i.e. the
    return-to-go of edge lengths along the rollout induced by the baseline
    policy used in `offline_data_tsp20.collect_tsp20_trajectories`.
    """
    episodes: List[Dict[str, Any]] = trajectories["episodes"]
    device = torch.device(device)
    bs = int(batch_states) if batch_states else 4096

    # Accumulators for batched evaluation
    b_locs: List[torch.Tensor] = []
    b_i: List[int] = []
    b_curr: List[int] = []
    b_first: List[int] = []
    b_mask: List[torch.Tensor] = []
    b_targets: List[float] = []

    mse_sum = 0.0
    n_total = 0

    def _flush() -> None:
        nonlocal mse_sum, n_total
        if not b_locs:
            return
        with torch.no_grad():
            locs_b = torch.stack(b_locs, dim=0).to(device)
            i_b = torch.tensor(b_i, dtype=torch.int64, device=device)
            curr_b = torch.tensor(b_curr, dtype=torch.int64, device=device)
            first_b = torch.tensor(b_first, dtype=torch.int64, device=device)
            mask_b = torch.stack(b_mask, dim=0).to(torch.bool).to(device)
            tgt_b = torch.tensor(b_targets, dtype=torch.float32, device=device)

            sv = _build_batched_state_view(locs_b, i_b, curr_b, first_b, mask_b, device)
            inv = InvariantTSPStateView(sv)
            try:
                phi = phi_fn(inv)
            except Exception as e:
                src = getattr(phi_fn, "_source_code", None)
                if src is not None:
                    print(
                        "[HeuristicFinder] Runtime error in phi(state) during mse_phi_vs_value; source code follows:",
                        file=sys.stderr,
                        flush=True,
                    )
                    try:
                        print("=" * 80, file=sys.stderr, flush=True)
                        print(src, file=sys.stderr, flush=True)
                        print("=" * 80, file=sys.stderr, flush=True)
                    except Exception:
                        pass
                # Re-raise so callers (e.g., evolutionary loop) can handle/penalize this candidate
                raise
            if not isinstance(phi, torch.Tensor):
                phi = torch.as_tensor(phi, dtype=torch.float32, device=device)
            phi = torch.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
            if phi.dim() > 1:
                phi = phi.squeeze(-1)
            phi_flat = phi.view(-1)

            # Align lengths defensively
            L = min(phi_flat.numel(), tgt_b.numel())
            if L == 0:
                return
            err = phi_flat[:L] - tgt_b[:L]
            mse_sum += float((err * err).sum().item())
            n_total += int(L)

        b_locs.clear()
        b_i.clear()
        b_curr.clear()
        b_first.clear()
        b_mask.clear()
        b_targets.clear()

    for ep in episodes:
        locs = ep["locs"].to(torch.float32)
        current_nodes = ep["current_nodes"].to(torch.long)
        masks = ep["action_masks"].to(torch.bool)
        r_base = ep["base_step_reward"].to(torch.float32)
        T = current_nodes.shape[0]

        # Future tour length (sum of edge lengths from t onward)
        cost_steps = (-r_base).flatten()
        future_cost = torch.flip(torch.cumsum(torch.flip(cost_steps, dims=[0]), dim=0), dims=[0])

        # Optional alternative targets (kept for completeness)
        returns = ep.get("returns", None)
        if returns is not None:
            returns = returns.to(torch.float32)

        for t in range(T):
            b_locs.append(locs)
            b_i.append(t)
            b_curr.append(int(current_nodes[t].item()))
            b_first.append(int(ep["first_node"]))
            b_mask.append(masks[t])
            if target == "future_cost":
                tgt_val = float(future_cost[t].item())
            elif target == "return":
                if returns is None:
                    raise ValueError("returns not present in trajectories for target='return'")
                tgt_val = float(returns[t].item())
            else:
                raise ValueError(f"Unknown target type '{target}'")
            b_targets.append(tgt_val)

            if len(b_locs) >= bs:
                _flush()

    _flush()

    if n_total == 0:
        # No valid states; treat as infinitely bad
        return float("inf")
    return float(mse_sum / float(n_total))
