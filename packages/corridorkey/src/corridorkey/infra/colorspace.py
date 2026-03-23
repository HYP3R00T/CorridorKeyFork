"""Shared sRGB transfer function LUT.

A single 65536-entry lookup table used by both the preprocessor (CPU numpy
path) and the postprocessor. Centralising it here avoids three independent
implementations of the same IEC 61966-2-1 piecewise function.

Usage::

    from corridorkey.infra.colorspace import linear_to_srgb_lut, srgb_to_linear_lut, lut_apply

    srgb = lut_apply(linear_arr, linear_to_srgb_lut)
    linear = lut_apply(srgb_arr, srgb_to_linear_lut)
"""

from __future__ import annotations

import numpy as np

# IEC 61966-2-1 constants
_LINEAR_THRESHOLD = 0.0031308
_ENCODED_THRESHOLD = 0.04045
_LINEAR_SCALE = 12.92
_GAMMA = 1.0 / 2.4
_SCALE = 1.055
_OFFSET = 0.055

_LUT_SIZE = 65536
_lut_x = np.linspace(0.0, 1.0, _LUT_SIZE, dtype=np.float64)

linear_to_srgb_lut: np.ndarray = np.where(
    _lut_x <= _LINEAR_THRESHOLD,
    _lut_x * _LINEAR_SCALE,
    _SCALE * np.power(np.maximum(_lut_x, 1e-12), _GAMMA) - _OFFSET,
).astype(np.float32)

srgb_to_linear_lut: np.ndarray = np.where(
    _lut_x <= _ENCODED_THRESHOLD,
    _lut_x / _LINEAR_SCALE,
    np.power(np.maximum((_lut_x + _OFFSET) / _SCALE, 1e-12), 2.4),
).astype(np.float32)


def lut_apply(x: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a 65536-entry LUT to a float32 array in [0, 1].

    Args:
        x: float32 array, values in [0, 1] (clipped internally).
        lut: 65536-entry float32 LUT (linear_to_srgb_lut or srgb_to_linear_lut).

    Returns:
        float32 array, same shape as x.
    """
    # Clip input to [0, 1] first — this guarantees indices land in
    # [0, _LUT_SIZE-1] exactly, so no second clip on the indices is needed.
    indices = np.rint(np.clip(x, 0.0, 1.0) * (_LUT_SIZE - 1)).astype(np.int32)
    return lut[indices]
