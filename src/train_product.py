import os
import csv
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from dataset_v3 import UCF101Dataset
from model import build_model


# ==========================================
# CONFIG
# ==========================================

TRAIN_SIZE = 2000
VAL_SIZE = 500

BATCH_SIZE = 2
EPOCHS = 5

LR = 1e-4

BEST_MODEL_PATH = (
    "outputs/checkpoints/best_model_product.pth"
)

LAST_MODEL_PATH = (
    "outputs/checkpoints/last_checkpoint.pth"
)

LOG_FILE = (
    "outputs/logs/product_training_log.csv"
)

os.makedirs(
    "outputs/checkpoints",
    exist_ok=True
)

os.makedirs(
    "outputs/logs",
    exist_ok=True
)


# ==========================================
# TRAIN
# ==========================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device
):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for videos, labels in loader:

        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():

            outputs = model(videos)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        preds = outputs.argmax(1)

        total += labels.size(0)

        correct += (
            preds == labels
        ).sum().item()

    loss = running_loss / len(loader)

    acc = (
        correct / total
    ) * 100

    return loss, acc


# ==========================================
# VALIDATE
# ==========================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0
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

            preds = outputs.argmax(1)

            total += labels.size(0)

            correct += (
                preds == labels
            ).sum().item()

    loss = running_loss / len(loader)

    acc = (
        correct / total
    ) * 100

    return loss, acc


# ==========================================
# MAIN
# ==========================================

def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nUsing Device: {device}"
    )

    dataset = UCF101Dataset(
        dataset_root="data/raw/UCF-101",
        split_file="data/splits/trainlist01.txt"
    )

    train_dataset = Subset(
        dataset,
        range(0, TRAIN_SIZE)
    )

    val_dataset = Subset(
        dataset,
        range(
            TRAIN_SIZE,
            TRAIN_SIZE + VAL_SIZE
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = build_model()

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LR
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=2,
        factor=0.5
    )

    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0

    with open(
        LOG_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Epoch",
                "Train Loss",
                "Train Acc",
                "Val Loss",
                "Val Acc"
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
                    scaler,
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

            writer.writerow(
                [
                    epoch + 1,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc
                ]
            )

            scheduler.step(val_acc)

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

            if val_acc > best_acc:

                best_acc = val_acc

                torch.save(
                    model.state_dict(),
                    BEST_MODEL_PATH
                )

                print(
                    "Best Model Saved"
                )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict()
                },
                LAST_MODEL_PATH
            )

    print(
        "\nTraining Complete"
    )

    print(
        f"Best Accuracy: "
        f"{best_acc:.2f}%"
    )


if __name__ == "__main__":
    main()