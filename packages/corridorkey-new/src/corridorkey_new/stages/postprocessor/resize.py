"""Postprocessor stage — tensor-to-numpy conversion and resize back to source resolution.

Takes the raw model output tensors (on device, at model resolution) and
returns float32 numpy arrays at the original source resolution.

Resize strategy:
  - Alpha: cv2.INTER_LANCZOS4 on upscale (sharper edges), INTER_AREA on downscale.
  - FG:    cv2.INTER_LINEAR always (colour accuracy over sharpness).
"""

from __future__ import annotations

import cv2
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

    Alpha uses Lanczos4 on upscale (sharper matte edges) and INTER_AREA on
    downscale (anti-aliased). FG uses bilinear always (colour accuracy).

    Args:
        alpha: [1, 1, H, W] tensor, range 0-1.
        fg: [1, 3, H, W] tensor, range 0-1.
        source_h: Target height in pixels.
        source_w: Target width in pixels.

    Returns:
        Tuple of (alpha_np [H, W, 1], fg_np [H, W, 3]), both float32 numpy.
    """
    alpha_np = tensor_to_numpy_hwc(alpha.float()).clip(0.0, 1.0)  # [H_model, W_model, 1]
    fg_np = tensor_to_numpy_hwc(fg.float()).clip(0.0, 1.0)  # [H_model, W_model, 3]

    model_h, model_w = alpha_np.shape[:2]
    target = (source_w, source_h)  # cv2 uses (width, height)

    if source_h == model_h and source_w == model_w:
        return alpha_np, fg_np

    upscaling = source_h * source_w > model_h * model_w

    # Alpha: Lanczos4 on upscale for sharp matte edges; INTER_AREA on downscale.
    alpha_interp = cv2.INTER_LANCZOS4 if upscaling else cv2.INTER_AREA
    alpha_2d = cv2.resize(alpha_np[:, :, 0], target, interpolation=alpha_interp)
    alpha_out = np.clip(alpha_2d, 0.0, 1.0)[:, :, np.newaxis].astype(np.float32)

    # FG: bilinear always — colour accuracy matters more than sharpness here.
    fg_out = cv2.resize(fg_np, target, interpolation=cv2.INTER_LINEAR).astype(np.float32)

    return alpha_out, fg_out
