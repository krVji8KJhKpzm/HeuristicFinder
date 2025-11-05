from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from tensordict.tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.routing.tsp.pbrs_env import DensePBRSTSPEnv
from rl4co.heuristic_finder.potential import PotentialSpec
from rl4co.models.rl.reinforce.reinforce import REINFORCE
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.zoo.pomo.model import POMO
from rl4co.utils.ops import batchify, unbatchify
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class POMOPBRS(POMO):
    """POMO variant that uses PBRS-shaped step rewards for training loss.

    - Uses the standard POMO policy and evaluation reward (original objective).
    - Computes per-step shaped rewards via a Dense PBRS TSP env with provided potential.
    - The loss is sum_t -( (r'_t - baseline) * log p(a_t|s_t) ).
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        potential_fn: Callable,
        policy: nn.Module = None,
        policy_kwargs={},
        baseline: str = "shared",
        num_augment: int = 8,
        augment_fn: str | Callable = "dihedral8",
        first_aug_identity: bool = True,
        feats: list = None,
        num_starts: int = None,
        **kwargs,
    ):
        super().__init__(
            env=env,
            policy=policy,
            policy_kwargs=policy_kwargs,
            baseline=baseline,
            num_augment=num_augment,
            augment_fn=augment_fn,
            first_aug_identity=first_aug_identity,
            feats=feats,
            num_starts=num_starts,
            **kwargs,
        )

        # a PBRS env instance for stepwise shaped rewards (same generator to align state)
        try:
            gen = getattr(env, "generator", None)
            self._pbrs_env = DensePBRSTSPEnv(potential_fn=potential_fn, generator=gen)
        except Exception as e:
            log.error(f"Failed to set up PBRS env: {e}")
            raise
        # avoid pickling warnings for lightning hparams
        try:
            self.save_hyperparameters(logger=False, ignore=["env", "policy", "potential_fn"])
        except Exception:
            pass

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        td = self.env.reset(batch)

        # Use multistart also during training (POMO behavior)
        n_aug, n_start = self.num_augment, self.num_starts
        n_start = self.env.get_num_starts(td) if n_start is None else n_start

        # During training, we do not augment the data
        if phase == "train":
            n_aug = 0
        elif n_aug > 1:
            td = self.augment(td)

        # Request per-step log likelihoods and actions
        out = self.policy(
            td,
            self.env,
            phase=phase,
            return_actions=True,
            return_sum_log_likelihood=False,
            num_starts=n_start,
        )

        # Unbatchify reward & log_likelihood to [B, A, S, ...] when applicable
        log_likelihood = out["log_likelihood"]  # [batch, decode_len]

        if phase == "train":
            # Compute shaped per-step rewards for provided multistart actions
            actions = out["actions"]  # [B*S, L]

            # Replicate initial td across starts for PBRS rollout
            td_ms = batchify(td, n_start)  # [B*S]
            td_roll = self._pbrs_env.reset(td_ms)

            shaped_steps = []
            for t in range(actions.shape[1]):
                td_roll.set("action", actions[:, t])  # [B*S]
                td_roll = self._pbrs_env.step(td_roll)["next"]
                shaped_steps.append(td_roll["reward"].squeeze(-1))  # [B*S]
                if td_roll["done"].all():
                    break
            shaped_flat = torch.stack(shaped_steps, dim=1)  # [B*S, L']

            # Align shapes with log_likelihood
            L = min(shaped_flat.shape[1], log_likelihood.shape[1])
            shaped_flat = shaped_flat[:, :L]
            log_likelihood = log_likelihood[:, :L]

            # Reshape to [B, S, L]
            shaped = unbatchify(shaped_flat, n_start)
            log_likelihood = unbatchify(log_likelihood, n_start)

            # Baseline per start (on [B,S]); broadcast over time
            shaped_total = shaped.sum(dim=-1)  # [B, S]
            bl_val, bl_loss = self.baseline.eval(td, shaped_total, self.env)  # [B,1]
            advantage = shaped - bl_val.unsqueeze(-1)  # [B, S, L]
            reinforce_loss = -(advantage * log_likelihood).mean()
            loss = reinforce_loss + bl_loss
            out.update({
                "loss": loss,
                "reinforce_loss": reinforce_loss,
                "bl_loss": bl_loss,
                "bl_val": bl_val,
            })

        # For val/test, compute standard POMO metrics via superclass logic
        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}
