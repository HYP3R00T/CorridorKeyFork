"""Preprocessing stage — color space conversion (step 4).

Converts linear light images to sRGB before inference.
The model requires sRGB input — this is an input contract, not an optimisation.

Uses a precomputed uint16 LUT (~60x faster than np.power at 4K resolution).
"""

from __future__ import annotations

import numpy as np

# sRGB transfer function constants (IEC 61966-2-1)
_LINEAR_THRESHOLD = 0.0031308
_LINEAR_SCALE = 12.92
_GAMMA = 1.0 / 2.4
_SCALE = 1.055
_OFFSET = 0.055

_LUT_SIZE = 65536
_lut_x = np.linspace(0.0, 1.0, _LUT_SIZE, dtype=np.float64)

_linear_to_srgb_lut: np.ndarray = np.where(
    _lut_x <= _LINEAR_THRESHOLD,
    _lut_x * _LINEAR_SCALE,
    _SCALE * np.power(np.maximum(_lut_x, 1e-12), _GAMMA) - _OFFSET,
).astype(np.float32)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """Convert a linear light float32 image to sRGB using a uint16 LUT.

    Uses the IEC 61966-2-1 piecewise transfer function. Values are clamped
    to [0, 1] before conversion.

    Only call this when ``ClipManifest.is_linear`` is True.

    Args:
        image: float32 array [H, W, 3], linear light, range 0.0–1.0.

    Returns:
        float32 array [H, W, 3], sRGB gamma-encoded, range 0.0–1.0.
    """
    indices = np.clip(np.rint(np.clip(image, 0.0, 1.0) * (_LUT_SIZE - 1)).astype(np.uint16), 0, _LUT_SIZE - 1)
    return _linear_to_srgb_lut[indices]
