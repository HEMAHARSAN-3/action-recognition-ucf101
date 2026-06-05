"""
models/model_factory.py

Central Model Factory

Supported Models:
- i3d (S3D Wrapper)
- slowfast
"""

from models.i3d import build_i3d
from models.slowfast import build_slowfast


def get_model(config):
    """
    Build model from config.

    Expected config:

    {
        "arch": "i3d",
        "num_classes": 101,
        "pretrained": True
    }
    """

    arch = config["arch"].lower()

    num_classes = config.get(
        "num_classes",
        101,
    )

    pretrained = config.get(
        "pretrained",
        True,
    )

    if arch == "i3d":

        return build_i3d(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    elif arch == "slowfast":

        return build_slowfast(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    else:

        raise ValueError(
            f"Unsupported model: {arch}"
        )