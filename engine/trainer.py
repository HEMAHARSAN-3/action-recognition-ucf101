"""
engine/trainer.py

Training Engine

Features:
- AMP Support
- Gradient Accumulation
- EMA
- Checkpoint Saving
- Validation
- Scheduler Integration
- Progress Bars
"""

from pathlib import Path
from copy import deepcopy

import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from utils.metrics import (
    AverageMeter,
    accuracy,
)


class ModelEMA:
    """
    Exponential Moving Average
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
    def update(self, model):

        ema_state = self.ema.state_dict()
        model_state = model.state_dict()

        for k, v in ema_state.items():

            if v.dtype.is_floating_point:

                v.mul_(self.decay).add_(
                    model_state[k],
                    alpha=(1.0 - self.decay),
                )

    def state_dict(self):
        return self.ema.state_dict()


class CheckpointManager:
    """
    Save / Load Checkpoints
    """

    def __init__(
        self,
        checkpoint_dir="checkpoints",
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
        scheduler,
        epoch,
        best_metric,
        filename,
    ):

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),
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
    scheduler,
    checkpoint_path,
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    if (
        scheduler is not None
        and checkpoint.get(
            "scheduler_state_dict"
        ) is not None
    ):
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    return (
        checkpoint["epoch"],
        checkpoint["best_metric"],
    )


class Trainer:

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

        self.device = torch.device(device)

        self.config = config

        self.model.to(self.device)

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
        self.start_epoch = 1

        self.scaler = GradScaler(
            enabled=(
                torch.cuda.is_available()
                and config.get(
                    "amp",
                    True,
                )
            )
        )

        print(
            "[INFO] Trainer initialized"
        )

    def train_one_epoch(
        self,
        train_loader,
        epoch,
    ):

        self.model.train()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        accumulation_steps = (
            self.config.get(
                "accumulation_steps",
                1,
            )
        )

        use_amp = (
            torch.cuda.is_available()
            and self.config.get(
                "amp",
                True,
            )
        )

        progress_bar = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch}",
        )

        self.optimizer.zero_grad()

        for step, (
            inputs,
            targets,
        ) in enumerate(progress_bar):

            inputs = inputs.to(
                self.device,
                non_blocking=True,
            )

            targets = targets.to(
                self.device,
                non_blocking=True,
            )

            with autocast(
                enabled=use_amp
            ):

                outputs = self.model(
                    inputs
                )

                loss = self.criterion(
                    outputs,
                    targets,
                )

                loss = (
                    loss
                    / accumulation_steps
                )

            self.scaler.scale(
                loss
            ).backward()

            if (
                (step + 1)
                % accumulation_steps
                == 0
            ):

                self.scaler.unscale_(
                    self.optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get(
                        "gradient_clip",
                        1.0,
                    ),
                )

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

                self.optimizer.zero_grad()

                self.ema.update(
                    self.model
                )

                if (
                    self.scheduler
                    is not None
                ):
                    self.scheduler.step()

            acc = accuracy(
                outputs.detach(),
                targets,
            )

            loss_meter.update(
                loss.item()
                * accumulation_steps,
                targets.size(0),
            )

            acc_meter.update(
                acc,
                targets.size(0),
            )

            progress_bar.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                acc=f"{acc_meter.avg:.2f}",
            )

        return {
            "loss": loss_meter.avg,
            "acc": acc_meter.avg,
        }
    
    def resume(
        self,
        checkpoint_path,
    ):
        """
        Resume training from checkpoint.
        """

        epoch, best_metric = (
            self.checkpoint_manager.load(
                self.model,
                self.optimizer,
                self.scheduler,
                checkpoint_path,
            )
        )

        self.start_epoch = epoch + 1
        self.best_metric = best_metric

        print(
            f"[INFO] Resumed from epoch {epoch}"
        )

    @torch.no_grad()
    def validate(
        self,
        val_loader,
    ):

        self.model.eval()

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()

        for (
            inputs,
            targets,
        ) in tqdm(
            val_loader,
            desc="Validation",
        ):

            inputs = inputs.to(
                self.device
            )

            targets = targets.to(
                self.device
            )

            outputs = self.model(
                inputs
            )

            loss = self.criterion(
                outputs,
                targets,
            )

            acc = accuracy(
                outputs,
                targets,
            )

            loss_meter.update(
                loss.item(),
                targets.size(0),
            )

            acc_meter.update(
                acc,
                targets.size(0),
            )

        return {
            "loss": loss_meter.avg,
            "acc": acc_meter.avg,
        }

    def fit(
        self,
        train_loader,
        val_loader,
    ):

        epochs = self.config.get(
            "epochs",
            1,
        )

        for epoch in range(
            self.start_epoch,
            epochs + 1,
        ):

            train_metrics = (
                self.train_one_epoch(
                    train_loader,
                    epoch,
                )
            )

            val_metrics = (
                self.validate(
                    val_loader
                )
            )

            print(
                f"\nEpoch {epoch}"
            )

            print(
                f"Train Loss: {train_metrics['loss']:.4f}"
            )

            print(
                f"Train Acc : {train_metrics['acc']:.2f}"
            )

            print(
                f"Val Loss  : {val_metrics['loss']:.4f}"
            )

            print(
                f"Val Acc   : {val_metrics['acc']:.2f}"
            )

            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                best_metric=self.best_metric,
                filename="last_model.pth",
            )

            if (
                val_metrics["acc"]
                > self.best_metric
            ):

                self.best_metric = (
                    val_metrics["acc"]
                )

                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    best_metric=self.best_metric,
                    filename="best_model.pth",
                )

                print(
                    "[INFO] New Best Model Saved"
                )