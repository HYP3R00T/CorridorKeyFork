from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from corridorkey.errors import FrameReadError

logger = logging.getLogger(__name__)

# BGR→GRAY luminance weights (OpenCV convention: B=0.114, G=0.587, R=0.299).
# Module-level constant — avoids re-allocating a new array on every alpha read.
_BGR_TO_GRAY = np.array([0.114, 0.587, 0.299], dtype=np.float32)


def read_frame_pair(
    image_path: Path,
    alpha_path: Path,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Read one image and its corresponding alpha hint from disk.

    Handles both uint8 (PNG, JPEG, TIFF) and float32 (EXR) sources.
    The returned arrays are always float32 in range 0.0–1.0.

    The image channels are in OpenCV's native BGR order — no CPU reorder is
    performed here. The returned ``bgr`` flag tells the caller whether to
    reorder on-device. This avoids a full-resolution CPU memcopy per frame.

    The image is returned as-is (sRGB or linear — the caller knows which
    via ``ClipManifest.is_linear`` and handles conversion downstream).
    The alpha is always treated as linear.

    Args:
        image_path: Path to the input frame file.
        alpha_path: Path to the corresponding alpha hint frame file.

    Returns:
        Tuple of:
          - image [H, W, 3] float32 0.0–1.0, channels in BGR order
          - alpha [H, W, 1] float32 0.0–1.0
          - bgr: True if image channels are BGR (always True for OpenCV reads)

    Raises:
        FrameReadError: If either file cannot be read or has an unexpected shape.
    """
    image, bgr = _read_image(image_path, channels=3)
    alpha, _ = _read_image(alpha_path, channels=1)

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
        alpha_resized = cv2.resize(
            alpha[:, :, 0],
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        alpha = alpha_resized[:, :, np.newaxis]

    return image, alpha, bgr


def _read_image(path: Path, channels: int) -> tuple[np.ndarray, bool]:
    # Read an image file and return a float32 array normalised to 0.0–1.0.
    # Returns (array [H, W, channels] float32, bgr) where bgr=True means BGR order.
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FrameReadError(f"cv2.imread returned None for '{path}' — file missing or unreadable.")

    arr = _to_float32(raw, path)
    arr, bgr = _to_channels(arr, channels, path)
    return arr, bgr


def _to_float32(arr: np.ndarray, path: Path) -> np.ndarray:
    # Normalise to float32 in range 0.0–1.0.
    # uint8  → multiply by 1/255   (single allocation, no intermediate)
    # uint16 → multiply by 1/65535 (single allocation, no intermediate)
    # float  → clamp to [0, 1]     (EXR values outside range are clipped)
    if arr.dtype == np.uint8:
        return np.multiply(arr, 1.0 / 255.0, dtype=np.float32)
    if arr.dtype == np.uint16:
        return np.multiply(arr, 1.0 / 65535.0, dtype=np.float32)
    if np.issubdtype(arr.dtype, np.floating):
        # Avoid a redundant copy when the array is already float32 (e.g. EXR).
        f32 = arr if arr.dtype == np.float32 else arr.astype(np.float32)
        return np.clip(f32, 0.0, 1.0)
    raise FrameReadError(f"Unsupported dtype '{arr.dtype}' in '{path}'.")


def _to_channels(arr: np.ndarray, channels: int, path: Path) -> tuple[np.ndarray, bool]:
    # Reshape to the expected output shape without reordering BGR channels.
    # For 3-channel images: keep native BGR, return bgr=True so the caller
    # can reorder on-device as a zero-copy strided view (no CPU memcopy).
    # For single-channel and grayscale: bgr=False, channel order irrelevant.
    # Grayscale — add trailing channel dim
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]

    c = arr.shape[2]

    if channels == 1:
        if c == 1:
            return arr, False
        if c in (3, 4):
            # Convert multi-channel alpha hint to grayscale luminance.
            # Direct float32 dot product — no uint8 round-trip, preserves
            # full precision from 16-bit or float32 sources.
            # Weights match OpenCV's BGR→GRAY: B=0.114, G=0.587, R=0.299.
            gray = np.dot(arr[:, :, :3], _BGR_TO_GRAY)
            return gray[:, :, np.newaxis], False
        raise FrameReadError(f"Cannot reduce {c}-channel image to 1 channel: '{path}'.")

    if channels == 3:
        if c == 1:
            # Grayscale image used as RGB — broadcast to 3 channels, no BGR issue
            return np.repeat(arr, 3, axis=2), False
        if c == 3:
            # Keep native BGR — caller reorders on-device
            return arr, True
        if c == 4:
            # BGRA — drop alpha channel, keep BGR order
            return np.ascontiguousarray(arr[:, :, :3]), True
        raise FrameReadError(f"Cannot convert {c}-channel image to 3 channels: '{path}'.")

    raise FrameReadError(f"Unsupported channel count requested: {channels}.")
