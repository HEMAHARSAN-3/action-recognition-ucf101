"""
utils/visualize.py

GradCAM Visualization Utilities
"""

from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F

from scipy.ndimage import zoom


class GradCAM:
    """
    Generic GradCAM for Video Models.
    """

    def __init__(
        self,
        model,
        target_layer,
    ):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self._register_hooks()

    def _register_hooks(self):

        def forward_hook(
            module,
            input,
            output,
        ):
            self.activations = output.detach()

        def backward_hook(
            module,
            grad_input,
            grad_output,
        ):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(
            forward_hook
        )

        self.target_layer.register_full_backward_hook(
            backward_hook
        )

    def generate(
        self,
        clip,
        class_idx=None,
    ):
        """
        Args:
            clip: (1,C,T,H,W)

        Returns:
            cam: (T,H,W)
        """

        self.model.zero_grad()

        outputs = self.model(clip)

        if class_idx is None:
            class_idx = outputs.argmax(
                dim=1
            ).item()

        score = outputs[:, class_idx]

        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(
            dim=(2, 3, 4),
            keepdim=True,
        )

        cam = (
            weights * activations
        ).sum(dim=1)

        cam = F.relu(cam)

        cam = cam.squeeze(0)

        cam = cam.cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min + 1e-8
        )

        return cam


def overlay_heatmap(
    frame,
    heatmap,
    alpha=0.5,
):
    """
    Overlay heatmap on frame.

    frame : RGB image
    heatmap : (H,W)
    """

    heatmap = np.uint8(
        heatmap * 255
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET,
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB,
    )

    overlay = cv2.addWeighted(
        frame,
        1 - alpha,
        heatmap,
        alpha,
        0,
    )

    return overlay


def create_gradcam_frames(
    frames,
    cam,
):
    """
    Args:
        frames : list RGB frames
        cam : (T,H,W)

    Returns:
        list overlay frames
    """

    output_frames = []

    num_frames = min(
        len(frames),
        cam.shape[0],
    )

    for idx in range(num_frames):

        frame = frames[idx]

        heatmap = cam[idx]

        heatmap = cv2.resize(
            heatmap,
            (
                frame.shape[1],
                frame.shape[0],
            ),
        )

        overlay = overlay_heatmap(
            frame,
            heatmap,
        )

        output_frames.append(
            overlay
        )

    return output_frames


def save_gif(
    frames,
    output_path,
    fps=5,
):
    """
    Save GIF.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    imageio.mimsave(
        str(output_path),
        frames,
        fps=fps,
    )


def create_side_by_side(
    original_frames,
    gradcam_frames,
):
    """
    Create side-by-side comparison.
    """

    combined = []

    num_frames = min(
        len(original_frames),
        len(gradcam_frames),
    )

    for idx in range(num_frames):

        left = original_frames[idx]
        right = gradcam_frames[idx]

        if left.shape != right.shape:

            right = cv2.resize(
                right,
                (
                    left.shape[1],
                    left.shape[0],
                ),
            )

        merged = np.concatenate(
            [
                left,
                right,
            ],
            axis=1,
        )

        combined.append(
            merged
        )

    return combined


def save_side_by_side_gif(
    original_frames,
    gradcam_frames,
    output_path,
    fps=5,
):
    """
    Save Original vs GradCAM GIF.
    """

    combined = create_side_by_side(
        original_frames,
        gradcam_frames,
    )

    save_gif(
        combined,
        output_path,
        fps=fps,
    )


def upsample_cam(
    cam,
    target_frames,
):
    """
    Upsample temporal CAM.

    cam: (T,H,W)
    """

    scale_t = (
        target_frames
        / cam.shape[0]
    )

    return zoom(
        cam,
        (
            scale_t,
            1,
            1,
        ),
        order=1,
    )