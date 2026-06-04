import cv2
import numpy as np
from pathlib import Path


def load_train_split(split_file):
    samples = []

    with open(split_file, "r") as f:
        for line in f:
            video_path, label = line.strip().split()
            samples.append((video_path, int(label) - 1))

    return samples


def sample_frames(video_path, num_frames=16):

    cap = cv2.VideoCapture(str(video_path))

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int
    )

    frames = []

    for idx in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        success, frame = cap.read()

        if success:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame = cv2.resize(
                frame,
                (112, 112)
            )

            frames.append(frame)

    cap.release()

    return np.array(frames)


def save_frames(frames, output_dir):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for i, frame in enumerate(frames):

        frame_bgr = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        cv2.imwrite(
            str(output_dir / f"frame_{i:02d}.jpg"),
            frame_bgr
        )


if __name__ == "__main__":

    dataset_root = Path("data/raw/UCF-101")

    train_samples = load_train_split(
        "data/splits/trainlist01.txt"
    )

    first_video = dataset_root / train_samples[0][0]

    frames = sample_frames(first_video)

    save_frames(
        frames,
        Path("outputs/sample_frames")
    )

    print(
        f"Saved {len(frames)} frames "
        f"to outputs/sample_frames"
    )