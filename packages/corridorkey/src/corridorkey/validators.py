"""Validation utilities for frame processing.

All validators either return cleaned data or raise typed exceptions
from corridorkey.errors.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from corridorkey.errors import (
    FrameMismatchError,
    FrameReadError,
    MaskChannelError,
    WriteFailureError,
)

if TYPE_CHECKING:
    from corridorkey.clip_state import ClipEntry

logger = logging.getLogger(__name__)


def validate_frame_counts(
    clip_name: str,
    input_count: int,
    alpha_count: int,
    strict: bool = False,
) -> int:
    """Validate that input and alpha frame counts are compatible.

    Args:
        clip_name: Clip name used in error messages.
        input_count: Number of input frames.
        alpha_count: Number of alpha frames.
        strict: Raise on mismatch when True; log a warning and return the
            minimum when False.

    Returns:
        Number of frames to process (minimum of both counts).

    Raises:
        FrameMismatchError: If strict is True and counts differ.
    """
    if input_count != alpha_count:
        if strict:
            raise FrameMismatchError(clip_name, input_count, alpha_count)
        logger.warning(
            "Clip '%s': frame count mismatch - input has %d, alpha has %d. Truncating to %d.",
            clip_name,
            input_count,
            alpha_count,
            min(input_count, alpha_count),
        )
    return min(input_count, alpha_count)


def normalize_mask_channels(
    mask: np.ndarray,
    clip_name: str = "",
    frame_index: int = 0,
) -> np.ndarray:
    """Reduce a mask to a single-channel 2D array.

    Extracts the first channel from multi-channel masks.

    Args:
        mask: Input mask array of shape [H, W] or [H, W, C].
        clip_name: Clip name used in error messages.
        frame_index: Frame index used in error messages.

    Returns:
        2D float32 array of shape [H, W].

    Raises:
        MaskChannelError: If the array has zero channels or unexpected ndim.
    """
    if mask.ndim == 3:
        if mask.shape[2] == 0:
            raise MaskChannelError(clip_name, frame_index, 0)
        mask = mask[:, :, 0]
    elif mask.ndim != 2:
        raise MaskChannelError(clip_name, frame_index, mask.ndim)

    return mask.astype(np.float32) if mask.dtype != np.float32 else mask


def normalize_mask_dtype(mask: np.ndarray) -> np.ndarray:
    """Convert a mask to float32 in [0.0, 1.0] from any common dtype.

    Args:
        mask: Input mask array with dtype uint8, uint16, float32, or float64.

    Returns:
        float32 array with values in [0.0, 1.0].
    """
    if mask.dtype == np.uint8:
        return mask.astype(np.float32) / 255.0
    elif mask.dtype == np.uint16:
        return mask.astype(np.float32) / 65535.0
    elif mask.dtype == np.float64:
        return mask.astype(np.float32)
    elif mask.dtype == np.float32:
        return mask
    else:
        return mask.astype(np.float32)


def validate_frame_read(
    frame: np.ndarray | None,
    clip_name: str,
    frame_index: int,
    path: str,
) -> np.ndarray:
    """Validate that a frame was read successfully.

    Args:
        frame: Result of cv2.imread() - None if the read failed.
        clip_name: Clip name used in error messages.
        frame_index: Frame index used in error messages.
        path: File path that was read.

    Returns:
        The frame array unchanged.

    Raises:
        FrameReadError: If frame is None.
    """
    if frame is None:
        raise FrameReadError(clip_name, frame_index, path)
    return frame


def validate_write(
    success: bool,
    clip_name: str,
    frame_index: int,
    path: str,
) -> None:
    """Validate that a cv2.imwrite() call succeeded.

    Args:
        success: Return value of cv2.imwrite().
        clip_name: Clip name used in error messages.
        frame_index: Frame index used in error messages.
        path: File path that was written.

    Raises:
        WriteFailureError: If success is False.
    """
    if not success:
        raise WriteFailureError(clip_name, frame_index, path)


def ensure_output_dirs(clip_root: str) -> dict[str, str]:
    """Create output subdirectories for a clip and return their paths.

    Args:
        clip_root: Absolute path to the clip folder.

    Returns:
        Dict with keys 'root', 'fg', 'matte', 'comp', 'processed' mapping
        to absolute directory paths.
    """
    out_root = os.path.join(clip_root, "Output")
    dirs = {
        "root": out_root,
        "fg": os.path.join(out_root, "FG"),
        "matte": os.path.join(out_root, "Matte"),
        "comp": os.path.join(out_root, "Comp"),
        "processed": os.path.join(out_root, "Processed"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


@dataclass
class ValidationResult:
    """Outcome of validate_job_inputs.

    Attributes:
        ok: True if all checks passed and the job can proceed.
        errors: Fatal problems that must be resolved before processing.
        warnings: Non-fatal issues the user should be aware of.
    """

    ok: bool
    errors: list[str]
    warnings: list[str]


def validate_job_inputs(
    clip: ClipEntry,
    min_vram_gb: float = 6.0,
    expected_output_gb: float = 2.0,
) -> ValidationResult:
    """Validate a clip's inputs before loading the inference engine.

    Runs two tiers of checks:

    Tier 1 - instant (milliseconds):
        - Input asset path exists and is readable.
        - Alpha asset path exists and is readable (if present).
        - Output directory can be created.
        - Enough free disk space for expected output.
        - GPU VRAM meets the minimum requirement (if CUDA is available).
        - Mask file count matches frame count (directory listing only).

    Tier 2 - sample decode (a few seconds):
        - Decodes first, last, and one random middle frame.
        - Verifies all three have consistent resolution and dtype.

    Args:
        clip: ClipEntry in READY state (has both input and alpha assets).
        min_vram_gb: Minimum free VRAM in GB required to proceed.
            Defaults to 6.0 GB (safe floor for the 2048-resolution model).
        expected_output_gb: Estimated output size in GB used for disk space check.
            Defaults to 2.0 GB as a conservative per-clip estimate.

    Returns:
        ValidationResult with ok=True if all checks passed.
        Errors are fatal; warnings are informational.
    """
    import random

    import cv2

    errors: list[str] = []
    warnings: list[str] = []

    # Tier 1: instant checks

    # 1a. Input asset exists
    if clip.input_asset is None:
        errors.append(f"Clip '{clip.name}': no input asset found.")
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    input_path = clip.input_asset.path
    if not os.path.exists(input_path):
        errors.append(f"Clip '{clip.name}': input path does not exist: {input_path}")

    # 1b. Alpha asset exists (if present)
    if clip.alpha_asset is not None:
        alpha_path = clip.alpha_asset.path
        if not os.path.exists(alpha_path):
            errors.append(f"Clip '{clip.name}': alpha path does not exist: {alpha_path}")

    # 1c. Output directory can be created
    output_root = os.path.join(clip.root_path, "Output")
    try:
        os.makedirs(output_root, exist_ok=True)
    except OSError as exc:
        errors.append(f"Clip '{clip.name}': cannot create output directory '{output_root}': {exc}")

    # 1d. Disk space
    try:
        stat = os.statvfs(clip.root_path) if hasattr(os, "statvfs") else None
        if stat is not None:
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            if free_gb < expected_output_gb:
                errors.append(
                    f"Clip '{clip.name}': insufficient disk space - "
                    f"{free_gb:.1f} GB free, need ~{expected_output_gb:.1f} GB."
                )
        else:
            # Windows fallback via shutil
            import shutil

            free_gb = shutil.disk_usage(clip.root_path).free / (1024**3)
            if free_gb < expected_output_gb:
                errors.append(
                    f"Clip '{clip.name}': insufficient disk space - "
                    f"{free_gb:.1f} GB free, need ~{expected_output_gb:.1f} GB."
                )
    except Exception as exc:
        warnings.append(f"Clip '{clip.name}': disk space check failed (skipped): {exc}")

    # 1e. VRAM check
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            reserved = torch.cuda.memory_reserved(0)
            free_vram_gb = (props.total_mem - reserved) / (1024**3)
            if free_vram_gb < min_vram_gb:
                errors.append(
                    f"Clip '{clip.name}': insufficient VRAM - {free_vram_gb:.1f} GB free, need ~{min_vram_gb:.1f} GB."
                )
    except Exception as exc:
        warnings.append(f"Clip '{clip.name}': VRAM check failed (skipped): {exc}")

    # 1f. Mask count vs frame count (sequence assets only)
    if (
        clip.input_asset is not None
        and clip.alpha_asset is not None
        and clip.input_asset.asset_type == "sequence"
        and clip.alpha_asset.asset_type == "sequence"
    ):
        try:
            input_files = clip.input_asset.get_frame_files()
            alpha_files = clip.alpha_asset.get_frame_files()
            if len(input_files) != len(alpha_files):
                errors.append(
                    f"Clip '{clip.name}': frame count mismatch - "
                    f"{len(input_files)} input frames vs {len(alpha_files)} alpha frames."
                )
        except Exception as exc:
            warnings.append(f"Clip '{clip.name}': frame count check failed (skipped): {exc}")

    # Bail early if Tier 1 already has fatal errors - no point decoding.
    if errors:
        return ValidationResult(ok=False, errors=errors, warnings=warnings)

    # Tier 2: sample decode

    if clip.input_asset.asset_type == "sequence":
        try:
            input_files = clip.input_asset.get_frame_files()
            n = len(input_files)
            if n == 0:
                errors.append(f"Clip '{clip.name}': input sequence is empty.")
                return ValidationResult(ok=False, errors=errors, warnings=warnings)

            indices = sorted({0, n - 1, random.randint(0, n - 1)})
            shapes: list[tuple[int, int]] = []
            dtypes: list[str] = []

            for idx in indices:
                fpath = os.path.join(input_path, input_files[idx])
                frame = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                if frame is None:
                    errors.append(f"Clip '{clip.name}': could not decode frame '{input_files[idx]}'.")
                    continue
                shapes.append((frame.shape[0], frame.shape[1]))
                dtypes.append(str(frame.dtype))

            if len(set(shapes)) > 1:
                errors.append(f"Clip '{clip.name}': inconsistent frame resolution across sample frames: {set(shapes)}.")
            if len(set(dtypes)) > 1:
                warnings.append(f"Clip '{clip.name}': mixed dtypes across sample frames: {set(dtypes)}.")
        except Exception as exc:
            warnings.append(f"Clip '{clip.name}': sample decode check failed (skipped): {exc}")

    ok = len(errors) == 0
    return ValidationResult(ok=ok, errors=errors, warnings=warnings)
