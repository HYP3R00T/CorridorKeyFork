from __future__ import annotations

import cv2
import numpy as np


def sharpen_with_hint(
    alpha: np.ndarray,
    fg: np.ndarray,
    hint: np.ndarray,
    dilation_px: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a hard binary mask derived from the alpha hint to alpha and FG.

    Eliminates the soft edge tails introduced when upscaling model output from
    inference resolution to source resolution. The hint may be at any resolution
    — it is upscaled to match alpha/fg using INTER_NEAREST to preserve hard edges.

    Steps:
      1. Binarise the hint at its native resolution (threshold 0.5).
      2. Dilate by ``dilation_px`` to give breathing room so fine model edge
         detail is not clipped.
      3. Upscale to source resolution with INTER_NEAREST (hard boundary).
      4. Multiply both alpha and FG by the mask.

    Args:
        alpha: Alpha matte [H, W, 1] float32 0-1 at source resolution.
        fg: Foreground image [H, W, 3] float32 0-1 at source resolution.
        hint: Alpha hint [H, W, 1] or [H, W] float32 0-1 at any resolution.
            Upscaled to source resolution when dimensions differ.
        dilation_px: Dilation radius in pixels applied to the binarised hint.
            Default 3.

    Returns:
        Tuple of (alpha, fg) with the hard mask applied, same shapes as input.
    """
    src_h, src_w = alpha.shape[:2]

    # Binarise hint at its native resolution
    hint_2d = hint[:, :, 0] if hint.ndim == 3 else hint
    binary = (hint_2d >= 0.5).astype(np.uint8)

    # Dilate to recover breathing room around edges (kernel size controls radius,
    # iterations=1 is intentional — radius is set via kernel size not iteration count)
    if dilation_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilation_px + 1, 2 * dilation_px + 1),
        )
        binary = cv2.dilate(binary, kernel, iterations=1)

    # Upscale to source resolution with INTER_NEAREST to preserve hard boundary
    if binary.shape[0] != src_h or binary.shape[1] != src_w:
        binary = cv2.resize(binary, (src_w, src_h), interpolation=cv2.INTER_NEAREST)

    mask = binary[:, :, np.newaxis].astype(np.float32)
    alpha_out = alpha * mask
    fg_out = fg * mask

    return alpha_out, fg_out
