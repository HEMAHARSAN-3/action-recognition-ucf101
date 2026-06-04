import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from dataset_v2 import UCF101Dataset
from model import build_model


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (videos, labels) in enumerate(loader):

        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(videos)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 5 == 0:
            print(
                f"Batch [{batch_idx+1}/{len(loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / len(loader)

    accuracy = 100 * correct / total

    return epoch_loss, accuracy


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 50)
    print("TRAINING PIPELINE TEST")
    print("=" * 50)

    print(f"\nDevice: {device}")

    # Load full dataset
    full_dataset = UCF101Dataset(
        dataset_root="data/raw/UCF-101",
        split_file="data/splits/trainlist01.txt"
    )

    print(
        f"\nTotal Dataset Size: "
        f"{len(full_dataset)}"
    )

    # Use only first 100 videos
    dataset = Subset(
        full_dataset,
        range(100)
    )

    print(
        f"Training Subset Size: "
        f"{len(dataset)}"
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    print(
        f"Number of Batches: "
        f"{len(loader)}"
    )

    model = build_model()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=1e-4
    )

    print("\nStarting Training...\n")

    loss, acc = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device
    )

    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)

    print(f"\nLoss     : {loss:.4f}")
    print(f"Accuracy : {acc:.2f}%")

    print("\nTraining Pipeline Working Successfully!")


if __name__ == "__main__":
    main()