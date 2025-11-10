from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import os

import torch
import lightning as L

from tensordict.tensordict import TensorDict

from rl4co.envs.routing.tsp.env import TSPEnv, TSPGenerator
from rl4co.envs.routing.tsp.pbrs_env import DensePBRSTSPEnv, TSPStateView
from rl4co.heuristic_finder.potential import PotentialSpec
from rl4co.models.zoo.pomo import POMO
from rl4co.models.zoo.pomo.pbrs_model import POMOPBRS
from rl4co.utils.trainer import RL4COTrainer


@dataclass
class EvalResult:
    avg_base_reward: float
    steps: int
    shaped_step_reward_mean: float
    shaped_step_reward_std: float


def rollout_shaped_rewards(
    env: DensePBRSTSPEnv, td0: TensorDict, actions: torch.Tensor
) -> torch.Tensor:
    """Rollout env deterministically under provided `actions` collecting shaped step rewards.

    Returns tensor of shape [batch, decode_len]
    """
    td = td0.clone()
    B, L = actions.shape
    rewards = []
    for t in range(L):
        td.set("action", actions[:, t])
        td = env.step(td)["next"]
        rewards.append(td["reward"].squeeze(-1))
        if td["done"].all():
            break
    return torch.stack(rewards, dim=1)


def evaluate_phi_on_tsp20(
    phi: PotentialSpec,
    batch_size: int = 128,
    device: str = "cpu",
    ) -> EvalResult:
    """Simple evaluation: sample TSP20 batch, get POMO actions, compute shaped step rewards.

    Note: this does not train the policy; it provides quick diagnostics about the shaped signal.
    """
    # Base env for policy action generation
    # Switch evaluation to TSP-50
    num_loc = int(os.environ.get("TSP_NUM_LOC", "50"))
    gen = TSPGenerator(num_loc=num_loc)
    base_env = TSPEnv(generator=gen)

    # Sample batch
    td = base_env.reset(base_env.generator(batch_size=[batch_size]).to(device))

    # Policy (randomly initialized AM defaults inside POMO)
    model = POMO(env=base_env)
    model.eval()

    with torch.no_grad():
        # Disable multistart for simple evaluation
        out = model.policy(
            td, base_env, phase="val", return_actions=True, decode_type="greedy"
        )
        actions = out["actions"]
        base_reward = base_env.get_reward(td, actions)

    # PBRS env for shaped step rewards (evaluation of potential only)
    pbrs_env = DensePBRSTSPEnv(potential_fn=phi.fn, generator=gen)
    td0 = pbrs_env.reset(td)
    shaped_steps = rollout_shaped_rewards(pbrs_env, td0, actions)

    return EvalResult(
        avg_base_reward=base_reward.mean().item(),
        steps=shaped_steps.shape[1],
        shaped_step_reward_mean=shaped_steps.mean().item(),
        shaped_step_reward_std=shaped_steps.std(unbiased=False).item(),
    )



def train_fitness_phi_on_tsp20(
    phi: PotentialSpec,
    epochs: int = 1,
    batch_size: int = 64,
    train_data_size: int = 1_000,
    val_data_size: int = 256,
    num_starts: int = 8,
    device: str = "cpu",
    accelerator: str = "cpu",
    devices: int = 1,
    seed: Optional[int] = None,
    # PBRS shaping controls
    pbrs_gamma: float = 1.0,
    reward_scale: Optional[str] = None,  # None | "scale" | "norm" | int
    center_dphi: bool = False,
    norm_dphi: bool = False,
) -> float:
    """Short POMOPBRS training as fitness; returns validation reward (higher is better).

    Keeps budgets small for CPU; increase as needed for stronger signals.
    """
    if seed is not None:
        L.seed_everything(seed, workers=True)

    # Optional env flags for PBRS stability
    if center_dphi:
        os.environ["PBRS_CENTER_DPHI"] = "1"
    if norm_dphi:
        os.environ["PBRS_NORM_DPHI"] = "1"

    # Switch fitness short-training to TSP-50
    num_loc = int(os.environ.get("TSP_NUM_LOC", "50"))
    gen = TSPGenerator(num_loc=num_loc)
    env = TSPEnv(generator=gen, seed=seed)

    model = POMOPBRS(
        env=env,
        potential_fn=phi.fn,
        pbrs_gamma=pbrs_gamma,
        num_starts=num_starts,
        batch_size=batch_size,
        train_data_size=train_data_size,
        val_data_size=val_data_size,
        optimizer_kwargs={"lr": 1e-4},
        metrics={"train": ["loss", "reward"], "val": ["reward"]},
        reward_scale=reward_scale,
    )

    trainer = RL4COTrainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=epochs,
        precision="32-true",
        logger=None,
        callbacks=[],
        reload_dataloaders_every_n_epochs=1,
        enable_progress_bar=False,
    )

    trainer.fit(model)

    # Prefer logged metric; fallback to manual evaluation if missing
    key = "val/reward"
    if key in trainer.callback_metrics:
        val_reward = trainer.callback_metrics[key]
        val_reward = val_reward.item() if hasattr(val_reward, "item") else float(val_reward)
        return val_reward

    # Fallback: evaluate on a small batch
    td = env.reset(env.generator(batch_size=[val_data_size]))
    with torch.no_grad():
        out = model.policy(td, env, phase="val", return_actions=False)
    return out["reward"].mean().item()
