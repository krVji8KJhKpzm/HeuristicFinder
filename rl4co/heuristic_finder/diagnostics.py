from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import torch

from tensordict.tensordict import TensorDict

from rl4co.envs.routing.tsp.env import TSPEnv, TSPGenerator
from rl4co.envs.routing.tsp.pbrs_env import DensePBRSTSPEnv
from rl4co.heuristic_finder.potential import PotentialSpec, compile_potential
from rl4co.models.zoo.pomo import POMO


@dataclass
class PhiScaleStats:
    # basic across-all-steps stats
    phi_mean: float
    phi_std: float
    phi_min: float
    phi_max: float
    dphi_mean: float
    dphi_std: float
    dphi_min: float
    dphi_max: float
    base_step_reward_mean: float
    base_step_reward_std: float
    # correlation info
    corr_dphi_base_reward: Optional[float]
    # recommended gamma to roughly match scales (std-wise)
    gamma_match_std: Optional[float]
    # clamp diagnostics
    frac_phi_clamped: float
    # time profile (means per step)
    step_means_phi: List[float]
    step_means_dphi: List[float]
    # quantiles for quick distribution glimpse
    phi_q10: float
    phi_q50: float
    phi_q90: float
    dphi_q10: float
    dphi_q50: float
    dphi_q90: float


@dataclass
class PhiProfile:
    num_loc: int
    batches: int
    batch_size: int
    device: str
    stats: PhiScaleStats


def _to_float(x: torch.Tensor) -> float:
    try:
        return float(x.item())
    except Exception:
        return float(x)


def _safe_corr(x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    x = x.detach().flatten()
    y = y.detach().flatten()
    if x.numel() < 2 or y.numel() < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    xs = x.std(unbiased=False)
    ys = y.std(unbiased=False)
    if xs <= 0 or ys <= 0:
        return None
    r = (x * y).mean() / (xs * ys)
    return _to_float(r)


def _profile_on_batch(
    pbrs_env: DensePBRSTSPEnv,
    base_env: TSPEnv,
    policy: POMO,
    td0: TensorDict,
) -> Dict[str, torch.Tensor]:
    """Run one batch rollout under greedy POMO, collecting phi, dphi, and base rewards.

    Returns:
      dict of tensors with shapes [B, T] for step-wise arrays and [B] for per-episode if any.
    """
    device = td0.device
    with torch.no_grad():
        out = policy.policy(td0, base_env, phase="val", return_actions=True, decode_type="greedy")
        actions = out["actions"]  # [B, T]
        base_reward = base_env.get_reward(td0, actions)  # [B]

    # Step through PBRS env to compute phi/dphi/base per step
    td = pbrs_env.reset(td0.clone())
    B, T = actions.shape
    phis_b = []  # [B]
    dphis = []  # [B]
    base_steps = []  # [B]

    for t in range(T):
        # Build state view BEFORE
        sv_before = pbrs_env._build_state_view(td)
        phi_before = pbrs_env._safe_phi(sv_before).squeeze(-1)  # [B]

        # action and step
        td.set("action", actions[:, t])
        td_next = pbrs_env.step(td)["next"]

        # Build AFTER
        sv_after = pbrs_env._build_state_view(td_next)
        phi_after = pbrs_env._safe_phi(sv_after).squeeze(-1)  # [B]

        # Base step reward is negative edge length (mirror of env logic)
        last_node_loc = td["locs"].gather(-2, td["current_node"].view(B, 1, 1).expand(-1, -1, 2)).squeeze(-2)
        curr_node_loc = td["locs"].gather(-2, td["action"].view(B, 1, 1).expand(-1, -1, 2)).squeeze(-2)
        base_step = -torch.linalg.norm(last_node_loc - curr_node_loc, dim=-1, ord=2)  # [B]

        phis_b.append(phi_before)
        dphis.append(phi_after - phi_before)
        base_steps.append(base_step)

        td = td_next
        if td["done"].all():
            break

    # pad T if early stop
    T_actual = len(phis_b)
    if T_actual < T:
        pad = T - T_actual
        zero = torch.zeros((B,), device=device)
        phis_b += [zero.clone() for _ in range(pad)]
        dphis += [zero.clone() for _ in range(pad)]
        base_steps += [zero.clone() for _ in range(pad)]

    phis_b = torch.stack(phis_b, dim=1)  # [B,T]
    dphis = torch.stack(dphis, dim=1)  # [B,T]
    base_steps = torch.stack(base_steps, dim=1)  # [B,T]

    # clamp fraction (phi is clamped in _safe_phi to [-1e3,1e3])
    clamped = (phis_b >= 1e3) | (phis_b <= -1e3)
    frac_phi_clamped = clamped.float().mean()

    return {
        "phis": phis_b,
        "dphis": dphis,
        "base_steps": base_steps,
        "base_reward": base_reward,
        "frac_phi_clamped": frac_phi_clamped,
    }


def profile_phi_on_tsp(
    phi: PotentialSpec,
    num_loc_list: Sequence[int] = (20, 50, 100),
    batches_per_n: int = 4,
    batch_size: int = 128,
    device: str = "cpu",
    seed: Optional[int] = None,
    gamma: float = 1.0,
) -> Dict[str, object]:
    """Profile a potential on TSP across sizes, summarizing scale and distribution.

    Returns a JSON-serializable dict with a list of PhiProfile entries under key 'profiles'.
    """
    if seed is not None:
        torch.manual_seed(seed)

    profiles: List[Dict[str, object]] = []
    for n in num_loc_list:
        gen = TSPGenerator(num_loc=n)
        base_env = TSPEnv(generator=gen)
        pbrs_env = DensePBRSTSPEnv(potential_fn=phi.fn, generator=gen, gamma=gamma)

        # policy with random-initialized AM inside POMO; eval/greedy
        policy = POMO(env=base_env)
        policy.eval()

        # accumulators across batches
        all_phis = []
        all_dphis = []
        all_base_steps = []
        clamped_fracs = []

        for _ in range(batches_per_n):
            td0 = base_env.reset(base_env.generator(batch_size=[batch_size]).to(device))
            out = _profile_on_batch(pbrs_env, base_env, policy, td0)
            all_phis.append(out["phis"])  # [B,T]
            all_dphis.append(out["dphis"])  # [B,T]
            all_base_steps.append(out["base_steps"])  # [B,T]
            clamped_fracs.append(out["frac_phi_clamped"])  # scalar

        phis = torch.cat(all_phis, dim=0)
        dphis = torch.cat(all_dphis, dim=0)
        base_steps = torch.cat(all_base_steps, dim=0)
        frac_phi_clamped = torch.stack(clamped_fracs).mean()

        # flatten across steps and batch for overall stats
        phif = phis.flatten()
        dphif = dphis.flatten()
        brf = base_steps.flatten()

        # stats
        phi_mean = _to_float(phif.mean())
        phi_std = _to_float(phif.std(unbiased=False))
        phi_min = _to_float(phif.min())
        phi_max = _to_float(phif.max())
        dphi_mean = _to_float(dphif.mean())
        dphi_std = _to_float(dphif.std(unbiased=False))
        dphi_min = _to_float(dphif.min())
        dphi_max = _to_float(dphif.max())
        base_mean = _to_float(brf.mean())
        base_std = _to_float(brf.std(unbiased=False))

        # quantiles
        phi_q = torch.quantile(phif, torch.tensor([0.1, 0.5, 0.9], device=phif.device))
        dphi_q = torch.quantile(dphif, torch.tensor([0.1, 0.5, 0.9], device=phif.device))

        # correlations
        corr_db = _safe_corr(dphif, brf)

        # scale matching suggestion
        gamma_match = None
        if dphi_std > 0:
            gamma_match = _to_float(abs(base_std) / (abs(dphi_std) + 1e-12))

        # stepwise means
        step_means_phi = [
            _to_float(phis[:, t].mean()) for t in range(phis.shape[1])
        ]
        step_means_dphi = [
            _to_float(dphis[:, t].mean()) for t in range(dphis.shape[1])
        ]

        stats = PhiScaleStats(
            phi_mean=phi_mean,
            phi_std=phi_std,
            phi_min=phi_min,
            phi_max=phi_max,
            dphi_mean=dphi_mean,
            dphi_std=dphi_std,
            dphi_min=dphi_min,
            dphi_max=dphi_max,
            base_step_reward_mean=base_mean,
            base_step_reward_std=base_std,
            corr_dphi_base_reward=corr_db,
            gamma_match_std=gamma_match,
            frac_phi_clamped=_to_float(frac_phi_clamped),
            step_means_phi=step_means_phi,
            step_means_dphi=step_means_dphi,
            phi_q10=_to_float(phi_q[0]),
            phi_q50=_to_float(phi_q[1]),
            phi_q90=_to_float(phi_q[2]),
            dphi_q10=_to_float(dphi_q[0]),
            dphi_q50=_to_float(dphi_q[1]),
            dphi_q90=_to_float(dphi_q[2]),
        )

        profiles.append(
            asdict(
                PhiProfile(
                    num_loc=n,
                    batches=batches_per_n,
                    batch_size=batch_size,
                    device=device,
                    stats=stats,
                )
            )
        )

    return {
        "profiles": profiles,
    }


def save_profile_json(profile: Dict[str, object], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def summarize_profile_for_prompt(profile: Dict[str, object]) -> str:
    """Make a compact textual summary suitable to embed into an LLM prompt.

    Focus on scale, distribution, and suggested gamma.
    """
    lines: List[str] = []
    for p in profile.get("profiles", []):
        num_loc = p.get("num_loc")
        s = p.get("stats", {})
        lines.append(
            (
                f"TSP-{num_loc}: phi mean={s.get('phi_mean'):.4f}, std={s.get('phi_std'):.4f}, "
                f"q10/50/90={s.get('phi_q10'):.4f}/{s.get('phi_q50'):.4f}/{s.get('phi_q90'):.4f}; "
                f"dphi std={s.get('dphi_std'):.4f}, base_step std={s.get('base_step_reward_std'):.4f}, "
                f"corr(dphi, base)={s.get('corr_dphi_base_reward')}; "
                f"suggested gamma≈{s.get('gamma_match_std')}"
            )
        )
    return "\n".join(lines)


def load_phi_code(path: str) -> PotentialSpec:
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()
    fn = compile_potential(code)
    name = os.path.basename(path)
    return PotentialSpec(name=name, code=code, fn=fn)


def _main_cli():
    import argparse

    ap = argparse.ArgumentParser(description="Profile PBRS phi(state) scale/distribution on TSP")
    ap.add_argument("--code", type=str, required=True, help="Path to a Python file containing def phi(state): ...")
    ap.add_argument("--num-loc", type=str, default="20,50,100", help="Comma-separated TSP sizes to test")
    ap.add_argument("--batches", type=int, default=4, help="Batches per size")
    ap.add_argument("--batch-size", type=int, default=128, help="Batch size per batch")
    ap.add_argument("--device", type=str, default="cpu", help="Device (cpu|cuda)")
    ap.add_argument("--gamma", type=float, default=1.0, help="PBRS gamma used for profiling")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument("--out", type=str, default=None, help="Optional path to save JSON profile")
    ap.add_argument("--print-summary", action="store_true", help="Print a short summary line per size")
    args = ap.parse_args()

    spec = load_phi_code(args.code)
    sizes = [int(x.strip()) for x in args.num_loc.split(",") if x.strip()]
    prof = profile_phi_on_tsp(
        spec,
        num_loc_list=sizes,
        batches_per_n=args.batches,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        gamma=args.gamma,
    )
    if args.out:
        save_profile_json(prof, args.out)
    if args.print_summary:
        print(summarize_profile_for_prompt(prof))
    if not args.out and not args.print_summary:
        # Default: print full JSON to stdout
        print(json.dumps(prof, indent=2))


if __name__ == "__main__":
    _main_cli()

