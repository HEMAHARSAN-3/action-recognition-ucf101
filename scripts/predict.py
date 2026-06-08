"""
scripts/predict.py

Inference Script for Action Recognition

Usage:

python scripts/predict.py \
    --config configs/ucf101_i3d.yaml \
    --checkpoint checkpoints/best_model.pth \
    --video sample.mp4
"""

import sys
from pathlib import Path
import argparse
import yaml
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_factory import get_model


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_class_names(dataset_root):
    """
    Load UCF101 class names from dataset folders.
    """

    dataset_root = Path(dataset_root)

    classes = sorted(
        [
            d.name
            for d in dataset_root.iterdir()
            if d.is_dir()
        ]
    )

    return classes


def get_frame_indices(total_frames, num_frames):
    """
    Uniform temporal sampling.
    """

    if total_frames <= 0:
        raise RuntimeError(
            "Video contains no frames."
        )

    indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
    ).astype(int)

    return indices


def load_video(video_path, num_frames=16):
    """
    Load video and prepare tensor.

    Returns:
        Tensor (1, C, T, H, W)
    """

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    frame_indices = get_frame_indices(
        total_frames,
        num_frames,
    )

    frames = []

    for idx in frame_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(idx),
        )

        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        frame = cv2.resize(
            frame,
            (224, 224),
        )

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(
            f"Failed to read video: {video_path}"
        )

    while len(frames) < num_frames:
        frames.append(frames[-1])

    clip = np.stack(frames)

    clip = (
        torch.from_numpy(clip)
        .float()
        / 255.0
    )

    # (T,H,W,C)
    # -> (T,C,H,W)

    clip = clip.permute(
        0,
        3,
        1,
        2,
    )

    # (T,C,H,W)
    # -> (C,T,H,W)

    clip = clip.permute(
        1,
        0,
        2,
        3,
    )

    clip = clip.unsqueeze(0)

    return clip


@torch.no_grad()
def predict(
    model,
    video_tensor,
    class_names,
    device,
):
    """
    Run inference.
    """

    model.eval()

    video_tensor = video_tensor.to(device)

    outputs = model(video_tensor)

    probs = torch.softmax(
        outputs,
        dim=1,
    )

    confidence, pred_idx = torch.max(
        probs,
        dim=1,
    )

    prediction = class_names[
        pred_idx.item()
    ]

    confidence = (
        confidence.item() * 100
    )

    return prediction, confidence


def main(args):

    cfg = load_config(args.config)

    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[INFO] Device: {device}"
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

    model.to(device)

    print(
        "[INFO] Model loaded successfully"
    )

    class_names = load_class_names(
        dataset_cfg["root"]
    )

    video_tensor = load_video(
        args.video,
        dataset_cfg["num_frames"],
    )

    prediction, confidence = predict(
        model,
        video_tensor,
        class_names,
        device,
    )

    print("\n========== RESULT ==========")

    print(
        f"Video      : {args.video}"
    )

    print(
        f"Prediction : {prediction}"
    )

    print(
        f"Confidence : {confidence:.2f}%"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Action Recognition Inference"
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

    parser.add_argument(
        "--video",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    main(args)