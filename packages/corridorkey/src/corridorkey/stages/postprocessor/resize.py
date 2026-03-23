"""Postprocessor stage — tensor-to-numpy conversion and resize back to source resolution.

Takes the raw model output tensors (on device, at model resolution) and
returns float32 numpy arrays at the original source resolution.

Since the preprocessor squishes the frame to a square (no padding), the
postprocessor simply resizes directly back to source resolution — no crop step.

Resize strategy:
  - Downscaling: always INTER_AREA (anti-aliased, no ringing).
  - FG upscaling: configurable — "lanczos4" (default), "bicubic", or "bilinear".
  - Alpha upscaling: configurable — "lanczos4" (default) or "bilinear".
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np
import torch

FgUpsampleMode = Literal["bilinear", "bicubic", "lanczos4"]
AlphaUpsampleMode = Literal["bilinear", "lanczos4"]

_FG_INTERP_MAP: dict[str, int] = {
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos4": cv2.INTER_LANCZOS4,
}
_ALPHA_INTERP_MAP: dict[str, int] = {
    "bilinear": cv2.INTER_LINEAR,
    "lanczos4": cv2.INTER_LANCZOS4,
}


def tensor_to_numpy_hwc(t: torch.Tensor) -> np.ndarray:
    """Convert a [1, C, H, W] tensor to a [H, W, C] float32 numpy array."""
    return t.detach().cpu().float().squeeze(0).permute(1, 2, 0).numpy()


def resize_to_source(
    alpha: torch.Tensor,
    fg: torch.Tensor,
    source_h: int,
    source_w: int,
    fg_upsample_mode: FgUpsampleMode = "lanczos4",
    alpha_upsample_mode: AlphaUpsampleMode = "lanczos4",
) -> tuple[np.ndarray, np.ndarray]:
    """Resize alpha and fg tensors back to source resolution and convert to numpy.

    Downscaling always uses INTER_AREA. Upscaling uses the configured modes.

    Args:
        alpha: [1, 1, H, W] tensor, range 0-1.
        fg: [1, 3, H, W] tensor, range 0-1.
        source_h: Target height in pixels.
        source_w: Target width in pixels.
        fg_upsample_mode: Interpolation for upscaling FG. Default "lanczos4".
        alpha_upsample_mode: Interpolation for upscaling alpha. Default "lanczos4".

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

    alpha_interp = _ALPHA_INTERP_MAP[alpha_upsample_mode] if upscaling else cv2.INTER_AREA
    alpha_2d = cv2.resize(alpha_np[:, :, 0], target, interpolation=alpha_interp)
    alpha_out = np.clip(alpha_2d, 0.0, 1.0)[:, :, np.newaxis].astype(np.float32)

    fg_interp = _FG_INTERP_MAP[fg_upsample_mode] if upscaling else cv2.INTER_AREA
    fg_out = cv2.resize(fg_np, target, interpolation=fg_interp).astype(np.float32)

    return alpha_out, fg_out
