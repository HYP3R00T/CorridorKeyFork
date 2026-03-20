"""Preprocessing stage — color space conversion (step 4).

Converts linear light images to sRGB before inference.
The model requires sRGB input — this is an input contract, not an optimisation.

Swap this file to support additional transfer functions (e.g. PQ, HLG, Log-C)
without touching any other part of the preprocessor.
"""

from __future__ import annotations

import numpy as np


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """Convert a linear light float32 image to sRGB gamma.

    Uses the IEC 61966-2-1 piecewise transfer function. Values are clamped
    to [0, 1] before conversion — out-of-range linear values are not valid
    sRGB and would produce undefined results.

    Only call this when ``ClipManifest.is_linear`` is True. sRGB input must
    pass through unchanged.

    Args:
        image: float32 array [H, W, 3], linear light, range 0.0–1.0.

    Returns:
        float32 array [H, W, 3], sRGB gamma-encoded, range 0.0–1.0.
    """
    image = np.clip(image, 0.0, 1.0)
    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)
