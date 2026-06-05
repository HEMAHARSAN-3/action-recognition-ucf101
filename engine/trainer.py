"""
engine/trainer.py

Training Infrastructure

Part A:
- EMA
- Checkpoint Manager
- Trainer Skeleton
"""

from pathlib import Path
from copy import deepcopy

import torch


class ModelEMA:
    """
    Exponential Moving Average of model weights.
    """

    def __init__(
        self,
        model,
        decay=0.9999,
    ):
        self.ema = deepcopy(model).eval()

        self.decay = decay

        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(
        self,
        model,
    ):
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()

        for key in ema_state.keys():

            if ema_state[key].dtype.is_floating_point:

                ema_state[key].mul_(
                    self.decay
                ).add_(
                    model_state[key],
                    alpha=1.0 - self.decay,
                )

    def state_dict(self):
        return self.ema.state_dict()


class CheckpointManager:
    """
    Save and load checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir,
    ):
        self.checkpoint_dir = Path(
            checkpoint_dir
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        model,
        optimizer,
        epoch,
        best_metric,
        filename,
    ):

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
        }

        save_path = (
            self.checkpoint_dir
            / filename
        )

        torch.save(
            checkpoint,
            save_path,
        )

        print(
            f"[INFO] Saved checkpoint: {save_path}"
        )

    def load(
        self,
        model,
        optimizer,
        checkpoint_path,
    ):

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        return (
            checkpoint["epoch"],
            checkpoint["best_metric"],
        )


class Trainer:
    """
    Trainer Skeleton

    Part A:
    Infrastructure only.
    """

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        config,
    ):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        self.device = device

        self.config = config

        self.model.to(
            self.device
        )

        self.ema = ModelEMA(
            model,
            decay=config.get(
                "ema_decay",
                0.9999,
            ),
        )

        self.checkpoint_manager = (
            CheckpointManager(
                config.get(
                    "checkpoint_dir",
                    "checkpoints",
                )
            )
        )

        self.best_metric = 0.0

        print(
            "[INFO] Trainer initialized"
        )