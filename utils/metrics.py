"""
utils/metrics.py

Metrics for Action Recognition.
"""

import torch


class AverageMeter:
    """
    Computes and stores
    average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(
        self,
        val,
        n=1,
    ):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(
    output,
    target,
):
    """
    Top-1 Accuracy

    Args:
        output: (B, num_classes)
        target: (B,)

    Returns:
        float
    """

    with torch.no_grad():

        pred = torch.argmax(
            output,
            dim=1,
        )

        correct = (
            pred == target
        ).sum()

        acc = (
            correct.float()
            / target.size(0)
        ) * 100.0

    return acc.item()


def top_k_accuracy(
    output,
    target,
    topk=(1, 5),
):
    """
    Compute Top-K Accuracy.

    Automatically handles cases where
    K > num_classes.

    Args:
        output: (B, num_classes)
        target: (B,)
        topk: tuple

    Returns:
        List[float]
    """

    with torch.no_grad():

        num_classes = output.size(1)

        maxk = min(
            max(topk),
            num_classes,
        )

        batch_size = target.size(0)

        _, pred = output.topk(
            maxk,
            dim=1,
            largest=True,
            sorted=True,
        )

        pred = pred.t()

        correct = pred.eq(
            target.view(1, -1).expand_as(pred)
        )

        results = []

        for k in topk:

            k = min(
                k,
                num_classes,
            )

            correct_k = (
                correct[:k]
                .reshape(-1)
                .float()
                .sum(0)
            )

            results.append(
                (
                    correct_k
                    * (100.0 / batch_size)
                ).item()
            )

        return results