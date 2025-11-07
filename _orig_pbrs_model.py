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

    def post_setup_hook(self):
        """After Lightning setup, print a quick estimate of Phi's scale.

        We evaluate Phi(state) on a small fresh batch and also estimate a one-step
        Delta Phi using a random valid action to help diagnose shaping magnitude.
        """
        try:
            # Sample a small batch from the training generator
            B = 128
            td0 = self.env.generator(batch_size=[B])
            td0 = td0.to(self.device)

            # Reset both base and PBRS envs on the same batch
            td_base = self.env.reset(td0)
            td_pbrs = self._pbrs_env.reset(td_base.clone())

            # Evaluate Phi(state) on initial states
            sv0 = self._pbrs_env._build_state_view(td_pbrs)
            phi0 = self._pbrs_env._safe_phi(sv0).squeeze(-1)  # [B]
            m, s = phi0.mean().item(), phi0.std(unbiased=False).item()
            am = phi0.abs().mean().item()
            mn, mx = phi0.min().item(), phi0.max().item()

            # Take one random valid action per instance and compute Delta Phi
            mask = td_pbrs["action_mask"]  # [B, N]
            rand = torch.rand_like(mask.float())
            rand = rand.masked_fill(~mask, -1e9)
            actions = rand.argmax(dim=-1)  # [B]

            td_step = td_pbrs.clone()
            td_step.set("action", actions)
            sv_before = self._pbrs_env._build_state_view(td_step)
            phi_before = self._pbrs_env._safe_phi(sv_before).squeeze(-1)
            td_step = self._pbrs_env.step(td_step)["next"]
            sv_after = self._pbrs_env._build_state_view(td_step)
            phi_after = self._pbrs_env._safe_phi(sv_after).squeeze(-1)
            dphi = (phi_after - phi_before)
            dm, ds = dphi.mean().item(), dphi.std(unbiased=False).item()
            dam = dphi.abs().mean().item()

            log.info(
                (
                    f"[PBRS] Phi(state) scale (B={B}): "
                    f"mean={m:.6f} std={s:.6f} abs_mean={am:.6f} min={mn:.6f} max={mx:.6f}"
                )
            )
            log.info(
                (
                    f"[PBRS] Delta Phi (one random step): "
                    f"mean={dm:.6f} std={ds:.6f} abs_mean={dam:.6f}"
                )
            )
        except Exception as e:
            log.error(f"[PBRS] Failed to evaluate Phi scale: {e}")

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
