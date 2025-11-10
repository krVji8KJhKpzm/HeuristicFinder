from __future__ import annotations

from lightning import Callback, Trainer


class PrintValObjective(Callback):
    """Print the validation objective (tour length) at the end of each epoch.

    Note: In RL4CO, reward = -objective for TSP. We log objective as -(val/reward).
    This prints to stdout so it appears in your .log when redirecting outputs.
    """

    def on_validation_epoch_end(self, trainer: Trainer, pl_module) -> None:
        metrics = trainer.callback_metrics
        if metrics is None:
            return
        key = "val/reward"
        if key in metrics and metrics[key] is not None:
            try:
                val_reward = metrics[key].item() if hasattr(metrics[key], "item") else float(metrics[key])
                objective = -val_reward
                print(f"[epoch {trainer.current_epoch}] val_objective={objective:.6f}", flush=True)
            except Exception:
                pass

