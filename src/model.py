import torch
import torch.nn as nn
from torchvision.models.video import r3d_18


def build_model(num_classes=101):
    """
    Build R3D-18 model for UCF101 action recognition
    """

    model = r3d_18(weights=None)

    # Replace final classification layer
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes
    )

    return model


if __name__ == "__main__":

    # Create model
    model = build_model()

    # Create dummy batch
    dummy_input = torch.randn(
        4,   # Batch Size
        3,   # RGB Channels
        16,  # Frames
        112, # Height
        112  # Width
    )

    # Forward Pass
    output = model(dummy_input)

    print("=" * 50)
    print("R3D-18 MODEL TEST")
    print("=" * 50)

    print("\nInput Shape:")
    print(dummy_input.shape)

    print("\nOutput Shape:")
    print(output.shape)

    print("\nNumber of Classes:")
    print(output.shape[1])

    print("\nModel Test Successful!")