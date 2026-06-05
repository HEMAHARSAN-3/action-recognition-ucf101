"""
scripts/train.py

Training Entry Point
"""

import sys
from pathlib import Path

# --------------------------------------------------
# Add Project Root to Python Path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------
# Standard Imports
# --------------------------------------------------

import argparse
import yaml
import torch

from torch.utils.data import DataLoader

# --------------------------------------------------
# Project Imports
# --------------------------------------------------

from utils.seed import set_seed

from data.dataset import VideoDataset

from models.model_factory import get_model

from engine.losses import build_loss
from engine.scheduler import build_scheduler
from engine.trainer import Trainer


def load_config(config_path):
    """
    Load YAML configuration.
    """

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:

        config = yaml.safe_load(f)

    return config


def build_dataloaders(dataset_cfg, train_cfg):
    """
    Build Train and Validation Dataloaders.
    """

    print("[INFO] Loading Dataset...")

    train_dataset = VideoDataset(
        root_dir=dataset_cfg["root"],
        num_frames=dataset_cfg["num_frames"],
        train=True,
    )

    val_dataset = VideoDataset(
        root_dir=dataset_cfg["root"],
        num_frames=dataset_cfg["num_frames"],
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"[INFO] Train Samples: {len(train_dataset)}"
    )

    print(
        f"[INFO] Val Samples: {len(val_dataset)}"
    )

    return train_loader, val_loader


def build_optimizer(model, train_cfg):
    """
    AdamW Optimizer.
    """

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    return optimizer


def main(args):

    # ---------------------------------------------
    # Load Config
    # ---------------------------------------------

    cfg = load_config(
        args.config
    )

    set_seed(
        cfg.get(
            "seed",
            42,
        )
    )

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    # ---------------------------------------------
    # Device
    # ---------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[INFO] Device: {device}"
    )

    # ---------------------------------------------
    # Dataset
    # ---------------------------------------------

    train_loader, val_loader = (
        build_dataloaders(
            dataset_cfg,
            train_cfg,
        )
    )

    # ---------------------------------------------
    # Model
    # ---------------------------------------------

    print(
        "[INFO] Building Model..."
    )

    model = get_model(
        {
            "arch": model_cfg["arch"],
            "num_classes": dataset_cfg[
                "num_classes"
            ],
            "pretrained": model_cfg.get(
                "pretrained",
                True,
            ),
        }
    )

    # ---------------------------------------------
    # Loss
    # ---------------------------------------------

    criterion = build_loss(
        smoothing=train_cfg.get(
            "label_smoothing",
            0.1,
        )
    )

    # ---------------------------------------------
    # Optimizer
    # ---------------------------------------------

    optimizer = build_optimizer(
        model,
        train_cfg,
    )

    # ---------------------------------------------
    # Scheduler
    # ---------------------------------------------

    scheduler = build_scheduler(
        optimizer=optimizer,
        config=train_cfg,
        steps_per_epoch=len(
            train_loader
        ),
    )

    # ---------------------------------------------
    # Trainer
    # ---------------------------------------------

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=train_cfg,
    )

    # ---------------------------------------------
    # Resume Checkpoint (Future)
    # ---------------------------------------------

    if args.resume:

        print(
            f"[INFO] Resume support will be added later: {args.resume}"
        )

    # ---------------------------------------------
    # Training
    # ---------------------------------------------

    trainer.fit(
        train_loader,
        val_loader,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train Action Recognition Model"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    main(args)