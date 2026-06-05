"""
scripts/inference.py

Single Video Inference
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import yaml
import cv2
import torch
import numpy as np

from models.model_factory import get_model


def load_config(config_path):

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:

        return yaml.safe_load(f)


def load_video(
    video_path,
    num_frames=16,
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError(
            "No frames found."
        )

    indices = np.linspace(
        0,
        len(frames) - 1,
        num_frames,
        dtype=int,
    )

    sampled_frames = [
        frames[i]
        for i in indices
    ]

    sampled_frames = np.stack(
        sampled_frames
    )

    return sampled_frames


def preprocess_frames(
    frames,
):

    processed = []

    for frame in frames:

        frame = cv2.resize(
            frame,
            (224, 224),
        )

        frame = (
            frame.astype(np.float32)
            / 255.0
        )

        processed.append(frame)

    frames = np.stack(
        processed
    )

    frames = torch.tensor(
        frames
    ).permute(
        3,
        0,
        1,
        2,
    )

    return frames.float()


@torch.no_grad()
def predict(
    model,
    clip,
    class_names,
    device,
):

    clip = (
        clip.unsqueeze(0)
        .to(device)
    )

    outputs = model(
        clip
    )

    probs = torch.softmax(
        outputs,
        dim=1,
    )

    confidence, pred = (
        probs.max(dim=1)
    )

    return (
        class_names[
            pred.item()
        ],
        confidence.item()
        * 100,
    )


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

    model = get_model(
        {
            "arch": model_cfg["arch"],
            "num_classes": dataset_cfg[
                "num_classes"
            ],
            "pretrained": False,
        }
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(device)
    model.eval()

    dataset_root = Path(
        dataset_cfg["root"]
    )

    class_names = sorted(
        [
            d.name
            for d in dataset_root.iterdir()
            if d.is_dir()
        ]
    )

    frames = load_video(
        args.video,
        num_frames=dataset_cfg[
            "num_frames"
        ],
    )

    clip = preprocess_frames(
        frames
    )

    label, confidence = predict(
        model,
        clip,
        class_names,
        device,
    )

    print(
        "\n===== PREDICTION ====="
    )

    print(
        f"Class      : {label}"
    )

    print(
        f"Confidence : {confidence:.2f}%"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Video Inference"
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
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