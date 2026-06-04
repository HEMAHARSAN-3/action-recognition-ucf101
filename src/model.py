import torch
import torch.nn as nn

from torchvision.models.video import (
    r3d_18,
    R3D_18_Weights
)


def build_model(num_classes=101):

    model = r3d_18(
        weights=R3D_18_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


if __name__ == "__main__":

    model = build_model()

    dummy_input = torch.randn(
        1,
        3,
        16,
        112,
        112
    )

    output = model(dummy_input)

    print("Output Shape:")
    print(output.shape)