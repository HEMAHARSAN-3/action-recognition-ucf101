"""
engine/losses.py

Loss Functions for Action Recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy with Label Smoothing.
    """

    def __init__(
        self,
        smoothing=0.1,
    ):
        super().__init__()

        self.smoothing = smoothing

    def forward(
        self,
        logits,
        target,
    ):

        num_classes = logits.size(1)

        log_probs = F.log_softmax(
            logits,
            dim=1,
        )

        with torch.no_grad():

            true_dist = torch.zeros_like(
                log_probs
            )

            true_dist.fill_(
                self.smoothing
                / (num_classes - 1)
            )

            true_dist.scatter_(
                1,
                target.unsqueeze(1),
                1.0 - self.smoothing,
            )

        loss = torch.mean(
            torch.sum(
                -true_dist * log_probs,
                dim=1,
            )
        )

        return loss


def mixup_criterion(
    criterion,
    pred,
    target_a,
    target_b,
    lam,
):
    """
    MixUp loss.
    """

    return (
        lam
        * criterion(
            pred,
            target_a,
        )
        + (1 - lam)
        * criterion(
            pred,
            target_b,
        )
    )


def cutmix_criterion(
    criterion,
    pred,
    target_a,
    target_b,
    lam,
):
    """
    CutMix loss.
    """

    return (
        lam
        * criterion(
            pred,
            target_a,
        )
        + (1 - lam)
        * criterion(
            pred,
            target_b,
        )
    )


def build_loss(
    smoothing=0.1,
):
    """
    Factory function.
    """

    return LabelSmoothingCrossEntropy(
        smoothing=smoothing
    )