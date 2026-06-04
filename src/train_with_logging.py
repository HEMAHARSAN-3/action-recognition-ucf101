import os
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from dataset_v3 import UCF101Dataset
from model import build_model


# ==================================================
# CREATE OUTPUT DIRECTORIES
# ==================================================

os.makedirs("outputs/checkpoints", exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)


# ==================================================
# TRAIN FUNCTION
# ==================================================

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

    epoch_loss = (
        running_loss / len(loader)
    )

    accuracy = (
        100 * correct / total
    )

    return epoch_loss, accuracy


# ==================================================
# VALIDATION FUNCTION
# ==================================================

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

    epoch_loss = (
        running_loss / len(loader)
    )

    accuracy = (
        100 * correct / total
    )

    return epoch_loss, accuracy


# ==================================================
# SAVE TRAINING CURVES
# ==================================================

def save_plots(
    train_losses,
    val_losses,
    train_accs,
    val_accs
):

    epochs = range(
        1,
        len(train_losses) + 1
    )

    # Loss Curve
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_losses,
        label="Train Loss"
    )

    plt.plot(
        epochs,
        val_losses,
        label="Val Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()

    plt.savefig(
        "outputs/plots/loss_curve.png"
    )

    plt.close()

    # Accuracy Curve
    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_accs,
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        val_accs,
        label="Val Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.savefig(
        "outputs/plots/accuracy_curve.png"
    )

    plt.close()


# ==================================================
# MAIN
# ==================================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 50)
    print("TRAINING WITH LOGGING")
    print("=" * 50)

    print(f"\nDevice: {device}")

    full_dataset = UCF101Dataset(
        dataset_root="data/raw/UCF-101",
        split_file="data/splits/trainlist01.txt"
    )

    # SMALL DATASET FOR TESTING

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

    EPOCHS = 5

    best_accuracy = 0

    train_losses = []
    val_losses = []

    train_accs = []
    val_accs = []

    csv_file = (
        "outputs/logs/training_log.csv"
    )

    with open(
        csv_file,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Epoch",
                "Train Loss",
                "Train Accuracy",
                "Val Loss",
                "Val Accuracy"
            ]
        )

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

            train_losses.append(
                train_loss
            )

            val_losses.append(
                val_loss
            )

            train_accs.append(
                train_acc
            )

            val_accs.append(
                val_acc
            )

            writer.writerow(
                [
                    epoch + 1,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc
                ]
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

                torch.save(
                    model.state_dict(),
                    "outputs/checkpoints/best_model.pth"
                )

                print(
                    "Best Model Saved!"
                )

    save_plots(
        train_losses,
        val_losses,
        train_accs,
        val_accs
    )

    print("\nTraining Finished")

    print(
        f"Best Validation Accuracy: "
        f"{best_accuracy:.2f}%"
    )

    print(
        "\nGenerated Files:"
    )

    print(
        "outputs/logs/training_log.csv"
    )

    print(
        "outputs/plots/loss_curve.png"
    )

    print(
        "outputs/plots/accuracy_curve.png"
    )

    print(
        "outputs/checkpoints/best_model.pth"
    )


if __name__ == "__main__":
    main()