"""
models/i3d.py

I3D-style model implemented using
TorchVision S3D backbone.

Input:
    (B, C, T, H, W)

Output:
    (B, num_classes)
"""

import torch
import torch.nn as nn

from torchvision.models.video import (
    s3d,
    S3D_Weights,
)


class I3DModel(nn.Module):
    """
    S3D-based Action Recognition Model
    """

    def __init__(
        self,
        num_classes=101,
        pretrained=True,
        dropout=0.5,
    ):
        super().__init__()

        if pretrained:

            weights = (
                S3D_Weights.DEFAULT
            )

            self.backbone = s3d(
                weights=weights
            )

        else:

            self.backbone = s3d(
                weights=None
            )

        in_features = (
            self.backbone.classifier[
                1
            ].in_channels
        )

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(
                p=dropout
            ),
            nn.Conv3d(
                in_features,
                num_classes,
                kernel_size=1,
                stride=1,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.backbone(x)


def build_i3d(
    num_classes=101,
    pretrained=True,
):
    return I3DModel(
        num_classes=num_classes,
        pretrained=pretrained,
    )