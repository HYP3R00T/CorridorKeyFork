"""Postprocessor stage — tensor-to-numpy conversion and resize back to source resolution.

Takes the raw model output tensors (on device, at model resolution) and
returns float32 numpy arrays at the original source resolution.
"""

from __future__ import annotations

import numpy as np
import torch


def tensor_to_numpy_hwc(t: torch.Tensor) -> np.ndarray:
    """Convert a [1, C, H, W] tensor to a [H, W, C] float32 numpy array.

    Moves to CPU, detaches from the autograd graph, and squeezes the batch dim.

    Args:
        t: Tensor of shape [1, C, H, W].

    Returns:
        float32 numpy array of shape [H, W, C].
    """
    return t.detach().cpu().float().squeeze(0).permute(1, 2, 0).numpy()


def resize_to_source(
    alpha: torch.Tensor,
    fg: torch.Tensor,
    source_h: int,
    source_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize alpha and fg tensors back to source resolution and convert to numpy.

    Uses bilinear interpolation for fg and area/bilinear for alpha to preserve
    soft edges as accurately as possible.

    Args:
        alpha: [1, 1, H, W] tensor, range 0-1.
        fg: [1, 3, H, W] tensor, range 0-1.
        source_h: Target height in pixels.
        source_w: Target width in pixels.

    Returns:
        Tuple of (alpha_np [H, W, 1], fg_np [H, W, 3]), both float32 numpy.
    """
    size = (source_h, source_w)

    alpha_r = torch.nn.functional.interpolate(alpha.float(), size=size, mode="bilinear", align_corners=False)
    fg_r = torch.nn.functional.interpolate(fg.float(), size=size, mode="bilinear", align_corners=False)

    alpha_np = tensor_to_numpy_hwc(alpha_r).clip(0.0, 1.0)
    fg_np = tensor_to_numpy_hwc(fg_r).clip(0.0, 1.0)

    return alpha_np, fg_np
