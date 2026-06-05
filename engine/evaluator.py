"""
engine/evaluator.py

Evaluation utilities for Action Recognition.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from utils.metrics import (
    accuracy,
    top_k_accuracy,
)


class Evaluator:

    def __init__(
        self,
        model,
        device="cpu",
    ):

        self.model = model
        self.device = torch.device(
            device
        )

        self.model.to(
            self.device
        )

    @torch.no_grad()
    def evaluate(
        self,
        dataloader,
        class_names=None,
    ):

        self.model.eval()

        all_preds = []
        all_targets = []

        top1_scores = []
        top5_scores = []

        for inputs, targets in dataloader:

            inputs = inputs.to(
                self.device
            )

            targets = targets.to(
                self.device
            )

            outputs = self.model(
                inputs
            )

            top1 = accuracy(
                outputs,
                targets,
            )

            top1_top5 = top_k_accuracy(
                outputs,
                targets,
                topk=(1, 5),
            )

            top1_scores.append(
                top1
            )

            top5_scores.append(
                top1_top5[1]
            )

            preds = torch.argmax(
                outputs,
                dim=1,
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_targets.extend(
                targets.cpu().numpy()
            )

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                all_targets,
                all_preds,
                average="weighted",
                zero_division=0,
            )
        )

        report = classification_report(
            all_targets,
            all_preds,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        )

        cm = confusion_matrix(
            all_targets,
            all_preds,
        )

        metrics = {
            "top1": float(
                np.mean(top1_scores)
            ),
            "top5": float(
                np.mean(top5_scores)
            ),
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
            "f1": float(f1),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

        return metrics

    def save_metrics(
        self,
        metrics,
        output_file,
    ):

        output_file = Path(
            output_file
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4,
            )

        print(
            f"[INFO] Metrics saved: {output_file}"
        )