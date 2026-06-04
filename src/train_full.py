import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from dataset_v2 import UCF101Dataset
from model import build_model


CHECKPOINT_DIR = "outputs/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


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

    for videos, labels in loader:

        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(videos)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    loss = running_loss / len(loader)

    accuracy = (
        100 * correct / total
    )

    return loss, accuracy


def validate(
    model,
    loader,
    criterion,
    device
):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for videos, labels in loader:

            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    loss = running_loss / len(loader)

    accuracy = (
        100 * correct / total
    )

    return loss, accuracy


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using Device: {device}")

    full_dataset = UCF101Dataset(
        dataset_root="data/raw/UCF-101",
        split_file="data/splits/trainlist01.txt"
    )

    # TEMPORARY SMALL DATASET
    train_dataset = Subset(
        full_dataset,
        range(0, 200)
    )

    val_dataset = Subset(
        full_dataset,
        range(200, 250)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    model = build_model()

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=1e-4
    )

    best_accuracy = 0.0

    EPOCHS = 3

    for epoch in range(EPOCHS):

        print(
            f"\nEpoch [{epoch+1}/{EPOCHS}]"
        )

        train_loss, train_acc = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )
        )

        val_loss, val_acc = (
            validate(
                model,
                val_loader,
                criterion,
                device
            )
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Acc : {train_acc:.2f}%"
        )

        print(
            f"Val Loss  : {val_loss:.4f}"
        )

        print(
            f"Val Acc   : {val_acc:.2f}%"
        )

        if val_acc > best_accuracy:

            best_accuracy = val_acc

            checkpoint_path = (
                f"{CHECKPOINT_DIR}/best_model.pth"
            )

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

            print(
                f"Saved Best Model -> "
                f"{checkpoint_path}"
            )

    print("\nTraining Finished")
    print(
        f"Best Validation Accuracy: "
        f"{best_accuracy:.2f}%"
    )


if __name__ == "__main__":
    main()