"""
data/dataset.py

Video Dataset for:
- UCF101
- HMDB51

Returns:
(
    clip_tensor,
    label
)

clip shape:
(C, T, H, W)
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


VIDEO_EXTENSIONS = (
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
)


class VideoDataset(Dataset):
    """
    Generic Video Dataset

    Folder Structure:

    root/
    ├── Class1
    │   ├── video1.avi
    │   ├── video2.avi
    │
    ├── Class2
    │   ├── video3.avi
    """

    def __init__(
        self,
        root_dir,
        num_frames=16,
        transform=None,
        train=True,
    ):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.transform = transform
        self.train = train

        self.classes = sorted(
            [
                d.name
                for d in self.root_dir.iterdir()
                if d.is_dir()
            ]
        )

        self.class_to_idx = {
            cls_name: idx
            for idx, cls_name in enumerate(self.classes)
        }

        self.samples = []

        self._build_dataset()

        print(f"[INFO] Found {len(self.classes)} classes")
        print(f"[INFO] Found {len(self.samples)} videos")

    def _build_dataset(self):
        """
        Build dataset index
        """

        for class_name in self.classes:

            class_dir = self.root_dir / class_name

            label = self.class_to_idx[class_name]

            for video_path in class_dir.iterdir():

                if video_path.suffix.lower() in VIDEO_EXTENSIONS:

                    self.samples.append(
                        (
                            str(video_path),
                            label,
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def _get_frame_indices(
        self,
        total_frames,
    ):
        """
        Generate temporal sampling indices
        """

        if total_frames <= 0:
            raise RuntimeError(
                "Video contains no frames."
            )

        # Video shorter than clip length
        if total_frames < self.num_frames:

            indices = np.linspace(
                0,
                total_frames - 1,
                self.num_frames,
            ).astype(int)

        else:

            # Training = random temporal crop
            if self.train:

                start = np.random.randint(
                    0,
                    total_frames - self.num_frames + 1,
                )

            # Validation = center crop
            else:

                start = (
                    total_frames
                    - self.num_frames
                ) // 2

            indices = np.linspace(
                start,
                start + self.num_frames - 1,
                self.num_frames,
            ).astype(int)

        return indices

    def _load_frames(
        self,
        video_path,
    ):
        """
        Read only required frames
        """

        cap = cv2.VideoCapture(video_path)

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        frame_indices = self._get_frame_indices(
            total_frames
        )

        frames = []

        for frame_idx in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_idx),
            )

            success, frame = cap.read()

            if not success:
                continue

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            frames.append(frame)

        cap.release()

        if len(frames) == 0:

            raise RuntimeError(
                f"Failed to load frames from {video_path}"
            )

        # Pad if necessary
        while len(frames) < self.num_frames:
            frames.append(frames[-1])

        return frames

    def __getitem__(
        self,
        index,
    ):
        video_path, label = self.samples[index]

        frames = self._read_video(
            video_path
        )

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

        if self.transform:
            clip = self.transform(clip)

        return clip, label