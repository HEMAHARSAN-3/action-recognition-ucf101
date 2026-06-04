from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class UCF101Dataset(Dataset):

    def __init__(
        self,
        dataset_root,
        split_file,
        num_frames=16,
        image_size=112
    ):

        self.dataset_root = Path(dataset_root)

        self.num_frames = num_frames

        self.image_size = image_size

        self.samples = []

        with open(split_file, "r") as f:

            for line in f:

                video_path, label = (
                    line.strip().split()
                )

                self.samples.append(
                    (
                        video_path,
                        int(label) - 1
                    )
                )

    def __len__(self):

        return len(self.samples)

    def sample_frames(
        self,
        video_path
    ):

        cap = cv2.VideoCapture(
            str(video_path)
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            self.num_frames,
            dtype=int
        )

        frames = []

        for idx in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                idx
            )

            success, frame = cap.read()

            if success:

                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                frame = cv2.resize(
                    frame,
                    (
                        self.image_size,
                        self.image_size
                    )
                )

                frames.append(frame)

        cap.release()

        frames = np.array(frames)

        return frames

    def __getitem__(
        self,
        idx
    ):

        video_rel_path, label = (
            self.samples[idx]
        )

        video_path = (
            self.dataset_root
            / video_rel_path
        )

        frames = self.sample_frames(
            video_path
        )

        frames = (
            torch.tensor(
                frames,
                dtype=torch.float32
            )
            / 255.0
        )

        frames = frames.permute(
            3,
            0,
            1,
            2
        )

        return frames, label


if __name__ == "__main__":

    dataset = UCF101Dataset(
        dataset_root="data/raw/UCF-101",
        split_file="data/splits/trainlist01.txt"
    )

    video_tensor, label = dataset[0]

    print(
        "Tensor Shape:",
        video_tensor.shape
    )

    print(
        "Label:",
        label
    )