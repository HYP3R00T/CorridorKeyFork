"""Writer stage — orchestrator.

Public entry point: write_frame(frame, config)

Writes alpha, fg, and comp images to subdirectories under config.output_dir.
Output subdirectories are created on first write.

    alpha/  — alpha matte
    fg/     — foreground colour
    comp/   — checkerboard preview composite
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from corridorkey_new.postprocessor.contracts import PostprocessedFrame
from corridorkey_new.writer.contracts import WriteConfig

logger = logging.getLogger(__name__)

# EXR compression codec IDs (cv2 constant names not available in all builds).
_EXR_COMPRESSION_IDS: dict[str, int] = {
    "none": 0,
    "rle": 1,
    "zips": 2,
    "zip": 3,
    "piz": 4,
    "pxr24": 5,
    "dwaa": 6,
    "dwab": 7,
}


def write_frame(frame: PostprocessedFrame, config: WriteConfig) -> None:
    """Write all enabled outputs for one postprocessed frame to disk.

    Args:
        frame: PostprocessedFrame from the postprocessor stage.
        config: WriteConfig controlling which outputs to write and where.

    Raises:
        OSError: If any cv2.imwrite call fails.
    """
    exr_flags = _exr_flags(config.exr_compression)

    if config.alpha_enabled:
        _write(
            _alpha_to_bgr(frame.alpha),
            config.output_dir / "alpha" / f"{frame.stem}.{config.alpha_format}",
            config.alpha_format,
            exr_flags,
        )

    if config.fg_enabled:
        _write(
            cv2.cvtColor(frame.fg, cv2.COLOR_RGB2BGR),
            config.output_dir / "fg" / f"{frame.stem}.{config.fg_format}",
            config.fg_format,
            exr_flags,
        )

    if config.comp_enabled:
        _write(
            cv2.cvtColor(frame.comp, cv2.COLOR_RGB2BGR),
            config.output_dir / "comp" / f"{frame.stem}.{config.comp_format}",
            config.comp_format,
            exr_flags,
        )

    logger.debug("write_frame: stem=%s", frame.stem)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _exr_flags(compression: str) -> list[int]:
    codec = _EXR_COMPRESSION_IDS.get(compression.lower(), _EXR_COMPRESSION_IDS["dwaa"])
    return [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF, cv2.IMWRITE_EXR_COMPRESSION, codec]


def _alpha_to_bgr(alpha: np.ndarray) -> np.ndarray:
    """Convert [H, W, 1] alpha to [H, W, 3] BGR grayscale for writing."""
    a2d = alpha[:, :, 0] if alpha.ndim == 3 else alpha
    return np.stack([a2d, a2d, a2d], axis=-1)


def _write(img: np.ndarray, path: Path, fmt: str, exr_flags: list[int]) -> None:
    """Write a single image to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "exr":
        arr = img if img.dtype == np.float32 else img.astype(np.float32)
        ok = cv2.imwrite(str(path), arr, exr_flags)
    else:
        arr = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        ok = cv2.imwrite(str(path), arr)
    if not ok:
        raise OSError(f"cv2.imwrite failed: {path}")
