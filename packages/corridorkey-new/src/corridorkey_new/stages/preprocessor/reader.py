"""Preprocessing stage — frame reader (internal).

Reads one frame pair (image + alpha) from disk and returns two float32 arrays
in range 0.0–1.0. This is the only place in the preprocessor that touches
the filesystem.

Not part of the public API — called by stage.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from corridorkey_new.errors import FrameReadError

logger = logging.getLogger(__name__)


def _read_frame_pair(
    image_path: Path,
    alpha_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one image and its corresponding alpha hint from disk.

    Handles both uint8 (PNG, JPEG, TIFF) and float32 (EXR) sources.
    The returned arrays are always float32 in range 0.0–1.0.

    The image is returned as-is (sRGB or linear — the caller knows which
    via ``ClipManifest.is_linear`` and handles conversion downstream).
    The alpha is always treated as linear.

    Args:
        image_path: Path to the input frame file.
        alpha_path: Path to the corresponding alpha hint frame file.

    Returns:
        Tuple of (image [H, W, 3], alpha [H, W, 1]), both float32 0.0–1.0.

    Raises:
        FrameReadError: If either file cannot be read or has an unexpected shape.
    """
    image = _read_image(image_path, channels=3)
    alpha = _read_image(alpha_path, channels=1)

    if image.shape[:2] != alpha.shape[:2]:
        logger.warning(
            "Alpha dimensions %s do not match image dimensions %s — "
            "resizing alpha to match (image=%s, alpha=%s). "
            "Consider regenerating alpha at the correct resolution.",
            alpha.shape[:2],
            image.shape[:2],
            image_path.name,
            alpha_path.name,
        )
        alpha_resized = cv2.resize(alpha[:, :, 0], (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        alpha = alpha_resized[:, :, np.newaxis]

    return image, alpha


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_image(path: Path, channels: int) -> np.ndarray:
    """Read an image file and return a float32 array normalised to 0.0–1.0.

    Args:
        path: Path to the image file.
        channels: Expected number of output channels (1 for alpha, 3 for RGB).

    Returns:
        float32 array shaped [H, W, channels], range 0.0–1.0.

    Raises:
        FrameReadError: If the file cannot be read or decoded.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FrameReadError(f"cv2.imread returned None for '{path}' — file missing or unreadable.")

    arr = _to_float32(raw, path)
    arr = _to_channels(arr, channels, path)
    return arr


def _to_float32(arr: np.ndarray, path: Path) -> np.ndarray:
    """Normalise array to float32 in range 0.0–1.0.

    uint8  -> divide by 255
    uint16 -> divide by 65535
    float  -> clamp to [0, 1] (EXR values outside range are clipped)
    """
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    if np.issubdtype(arr.dtype, np.floating):
        return np.clip(arr.astype(np.float32), 0.0, 1.0)
    raise FrameReadError(f"Unsupported dtype '{arr.dtype}' in '{path}'.")


def _to_channels(arr: np.ndarray, channels: int, path: Path) -> np.ndarray:
    """Reshape and reorder channels to match the expected output shape.

    OpenCV reads images as BGR (or BGRA). This converts to RGB for 3-channel
    images and ensures a trailing channel dim for single-channel images.

    Args:
        arr: float32 array as returned by cv2.imread (H, W) or (H, W, C).
        channels: Desired output channel count (1 or 3).
        path: Source path, used in error messages only.

    Returns:
        float32 array shaped [H, W, channels].

    Raises:
        FrameReadError: If the source channel count is incompatible.
    """
    # Grayscale — add trailing channel dim
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    c = arr.shape[2]

    if channels == 1:
        if c == 1:
            return arr
        if c in (3, 4):
            # Convert multi-channel alpha hint to grayscale luminance
            gray = (
                cv2.cvtColor(
                    (arr[:, :, :3] * 255).astype(np.uint8),
                    cv2.COLOR_BGR2GRAY,
                ).astype(np.float32)
                / 255.0
            )
            return gray[:, :, np.newaxis]
        raise FrameReadError(f"Cannot reduce {c}-channel image to 1 channel: '{path}'.")

    if channels == 3:
        if c == 1:
            # Grayscale image used as RGB — broadcast to 3 channels
            return np.repeat(arr, 3, axis=2)
        if c == 3:
            # BGR -> RGB
            return arr[:, :, ::-1].copy()
        if c == 4:
            # BGRA -> RGB (drop alpha channel)
            return arr[:, :, 2::-1].copy()
        raise FrameReadError(f"Cannot convert {c}-channel image to 3 channels: '{path}'.")

    raise FrameReadError(f"Unsupported channel count requested: {channels}.")
