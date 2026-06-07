"""
scripts/evaluate.py

Evaluate trained Action Recognition models.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from data.dataset import VideoDataset
from engine.evaluator import Evaluator
from models.model_factory import get_model


def load_config(config_path):

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:

        return yaml.safe_load(f)


def main(args):

    cfg = load_config(
        args.config
    )

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dataset = VideoDataset(
        root_dir=dataset_cfg["root"],
        num_frames=dataset_cfg["num_frames"],
        train=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    model = get_model(
        {
            "arch": model_cfg["arch"],
            "num_classes": dataset_cfg["num_classes"],
            "pretrained": False,
        }
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    evaluator = Evaluator(
        model=model,
        device=device,
    )

    metrics = evaluator.evaluate(
        dataloader,
        class_names=dataset.classes,
    )

    output_dir = Path("outputs")
    output_dir.mkdir(
        exist_ok=True
    )

    evaluator.save_metrics(
        metrics,
        output_dir / "metrics.json",
    )

    evaluator.save_confusion_matrix(
        metrics,
        output_dir / "confusion_matrix.png",
    )

    evaluator.save_classification_report(
        metrics,
        output_dir / "classification_report.json",
    )

    print("\n===== RESULTS =====")

    print(
        f"Top1 Accuracy : {metrics['top1']:.2f}"
    )

    print(
        f"Top5 Accuracy : {metrics['top5']:.2f}"
    )

    print(
        f"Precision     : {metrics['precision']:.4f}"
    )

    print(
        f"Recall        : {metrics['recall']:.4f}"
    )

    print(
        f"F1 Score      : {metrics['f1']:.4f}"
    )

    print(
        f"Latency (ms)  : {metrics['avg_latency_ms']:.2f}"
    )

    print(
        f"Throughput    : {metrics['throughput_fps']:.2f} FPS"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Evaluate Action Recognition Model"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    main(args)