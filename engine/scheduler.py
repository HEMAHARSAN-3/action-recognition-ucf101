"""
engine/scheduler.py

Learning Rate Schedulers
"""

from torch.optim.lr_scheduler import (
    OneCycleLR,
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
)


def build_scheduler(
    optimizer,
    config,
    steps_per_epoch,
):
    """
    Build scheduler from config.
    """

    epochs = config["epochs"]

    max_lr = config["lr"]

    warmup_epochs = max(
        1,
        int(0.1 * epochs),
    )

    scheduler_type = config.get(
        "scheduler",
        "onecycle",
    )

    if scheduler_type == "onecycle":

        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
        )

        return scheduler

    elif scheduler_type == "cosine":

        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs,
        )

        cosine = CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                warmup,
                cosine,
            ],
            milestones=[
                warmup_epochs,
            ],
        )

        return scheduler

    else:

        raise ValueError(
            f"Unsupported scheduler: {scheduler_type}"
        )