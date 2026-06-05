"""
data/transforms.py

Video Transform Pipeline for Action Recognition

Tensor Format:
    (C, T, H, W)

Example:
    (3, 16, 224, 224)
"""

from typing import Tuple
import random

import torch
import torchvision.transforms.functional as F


class RandomHorizontalFlipVideo:
    """
    Random horizontal flip for video clips.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """
        clip shape: (C, T, H, W)
        """

        if random.random() < self.p:
            clip = torch.flip(clip, dims=[3])

        return clip


class RandomResizedCropVideo:
    """
    Random resized crop for video clips.
    """

    def __init__(
        self,
        size: Tuple[int, int] = (224, 224),
        scale=(0.8, 1.0),
    ):
        self.size = size
        self.scale = scale

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:

        _, _, h, w = clip.shape

        scale = random.uniform(*self.scale)

        crop_h = int(h * scale)
        crop_w = int(w * scale)

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        clip = clip[
            :,
            :,
            top : top + crop_h,
            left : left + crop_w,
        ]

        clip = torch.nn.functional.interpolate(
            clip.permute(1, 0, 2, 3),
            size=self.size,
            mode="bilinear",
            align_corners=False,
        )

        clip = clip.permute(1, 0, 2, 3)

        return clip


class ColorJitterVideo:
    """
    Apply color jitter frame-wise.
    """

    def __init__(
        self,
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, clip: torch.Tensor):

        frames = []

        for t in range(clip.shape[1]):

            frame = clip[:, t]

            brightness_factor = random.uniform(
                max(0, 1 - self.brightness),
                1 + self.brightness,
            )

            contrast_factor = random.uniform(
                max(0, 1 - self.contrast),
                1 + self.contrast,
            )

            saturation_factor = random.uniform(
                max(0, 1 - self.saturation),
                1 + self.saturation,
            )

            frame = F.adjust_brightness(
                frame,
                brightness_factor,
            )

            frame = F.adjust_contrast(
                frame,
                contrast_factor,
            )

            frame = F.adjust_saturation(
                frame,
                saturation_factor,
            )

            frames.append(frame)

        return torch.stack(frames, dim=1)


class TemporalJitter:
    """
    Randomly shift frame sequence.
    """

    def __init__(self, max_shift: int = 2):
        self.max_shift = max_shift

    def __call__(self, clip: torch.Tensor):

        shift = random.randint(
            -self.max_shift,
            self.max_shift,
        )

        return torch.roll(
            clip,
            shifts=shift,
            dims=1,
        )


class NormalizeVideo:
    """
    Normalize RGB video clip.
    """

    def __init__(
        self,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.mean = torch.tensor(mean).view(3, 1, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1, 1)

    def __call__(self, clip: torch.Tensor):

        return (clip - self.mean) / self.std


class ComposeVideo:
    """
    Compose multiple transforms.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, clip):

        for transform in self.transforms:
            clip = transform(clip)

        return clip


def get_train_transforms():

    return ComposeVideo(
        [
            RandomResizedCropVideo((224, 224)),
            RandomHorizontalFlipVideo(0.5),
            ColorJitterVideo(),
            TemporalJitter(),
            NormalizeVideo(),
        ]
    )


def get_val_transforms():

    return ComposeVideo(
        [
            NormalizeVideo(),
        ]
    )