"""
data/optical_flow.py

Optical Flow Extraction for Action Recognition.

Uses:
    OpenCV Farneback Dense Optical Flow

Output:
    flow.npy

Shape:
    (T-1, H, W, 2)
"""

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def extract_optical_flow(video_path):
    """
    Extract dense optical flow.

    Args:
        video_path (str)

    Returns:
        np.ndarray
    """

    cap = cv2.VideoCapture(video_path)

    success, prev_frame = cap.read()

    if not success:
        raise RuntimeError(
            f"Cannot open {video_path}"
        )

    prev_gray = cv2.cvtColor(
        prev_frame,
        cv2.COLOR_BGR2GRAY,
    )

    flows = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        flows.append(flow)

        prev_gray = gray

    cap.release()

    return np.array(
        flows,
        dtype=np.float32,
    )


def save_flow(
    video_path,
    output_path,
):
    """
    Extract and save flow.
    """

    flow = extract_optical_flow(
        video_path
    )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        flow,
    )

    print(
        f"[INFO] Saved: {output_path}"
    )


def process_dataset(
    dataset_root,
    output_root,
):
    """
    Process all videos.

    Args:
        dataset_root
        output_root
    """

    dataset_root = Path(
        dataset_root
    )

    output_root = Path(
        output_root
    )

    video_files = []

    for ext in (
        "*.avi",
        "*.mp4",
        "*.mov",
        "*.mkv",
    ):
        video_files.extend(
            dataset_root.rglob(ext)
        )

    print(
        f"[INFO] Found {len(video_files)} videos"
    )

    for video_path in tqdm(
        video_files,
        desc="Extracting Flow",
    ):

        relative = video_path.relative_to(
            dataset_root
        )

        output_file = (
            output_root
            / relative.parent
            / f"{video_path.stem}.npy"
        )

        if output_file.exists():
            continue

        try:

            save_flow(
                str(video_path),
                output_file,
            )

        except Exception as e:

            print(
                f"[ERROR] {video_path}"
            )

            print(e)