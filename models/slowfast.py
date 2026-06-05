"""
models/slowfast.py

SlowFast R50 Wrapper

Input:
[
    slow_pathway,
    fast_pathway
]

Output:
(B, num_classes)
"""

import torch
import torch.nn as nn

from pytorchvideo.models.hub import (
    slowfast_r50,
)


class SlowFastModel(nn.Module):

    def __init__(
        self,
        num_classes=101,
        pretrained=False,
    ):
        super().__init__()

        self.model = slowfast_r50(
            pretrained=pretrained
        )

        in_features = (
            self.model.blocks[-1]
            .proj
            .in_features
        )

        self.model.blocks[-1].proj = nn.Linear(
            in_features,
            num_classes,
        )

    def forward(
        self,
        x,
    ):
        return self.model(x)


def build_slowfast(
    num_classes=101,
    pretrained=False,
):
    return SlowFastModel(
        num_classes=num_classes,
        pretrained=pretrained,
    )