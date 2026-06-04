import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from torch.utils.data import DataLoader, Subset

from dataset_v3 import UCF101Dataset
from model import build_model


os.makedirs(
    "outputs/evaluation",
    exist_ok=True
)


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    dataset = UCF101Dataset(
        dataset_root="data/raw/UCF-101",
        split_file="data/splits/trainlist01.txt"
    )

    # Small validation subset
    dataset = Subset(
        dataset,
        range(200, 250)
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    model = build_model()

    model.load_state_dict(
        torch.load(
            "outputs/checkpoints/best_model.pth",
            map_location=device
        )
    )

    model.to(device)

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for videos, labels in loader:

            videos = videos.to(device)

            outputs = model(videos)

            predictions = outputs.argmax(
                dim=1
            )

            y_true.extend(
                labels.numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

    # ==================================
    # CSV RESULTS
    # ==================================

    df = pd.DataFrame({
        "True Label": y_true,
        "Predicted Label": y_pred
    })

    df.to_csv(
        "outputs/evaluation/evaluation_results.csv",
        index=False
    )

    # ==================================
    # CLASSIFICATION REPORT
    # ==================================

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0
    )

    with open(
        "outputs/evaluation/classification_report.txt",
        "w"
    ) as f:

        f.write(report)

    # ==================================
    # CONFUSION MATRIX
    # ==================================

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot(
        ax=ax,
        colorbar=False
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.savefig(
        "outputs/evaluation/confusion_matrix.png"
    )

    plt.close()

    print("\nEvaluation Complete")

    print(
        "Saved:"
    )

    print(
        "outputs/evaluation/evaluation_results.csv"
    )

    print(
        "outputs/evaluation/classification_report.txt"
    )

    print(
        "outputs/evaluation/confusion_matrix.png"
    )


if __name__ == "__main__":
    main()