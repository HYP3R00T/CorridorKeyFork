"""Preprocessing stage — orchestrator.

Runs steps 1–10 in order. Owns no transformation logic itself — each step
is delegated to its own module.

    Step 1  — validate inputs            (here)
    Step 2  — read from disk             → reader.py
    Step 3  — capture original dims      (here)
    Step 4  — color space conversion     → colorspace.py
    Step 5  — resize                     → resize.py
    Step 6  — ImageNet normalisation     → normalise.py
    Steps 7–9 — tensor + device transfer → tensor.py
    Step 10 — return PreprocessedFrame   (here)

Public entry point: preprocess_frame(manifest, i, config)
Each stage in the pipeline has a corresponding orchestrator.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from corridorkey_new.loader.contracts import ClipManifest
from corridorkey_new.loader.validator import get_frame_files
from corridorkey_new.preprocessor.colorspace import linear_to_srgb
from corridorkey_new.preprocessor.normalise import normalise_image
from corridorkey_new.preprocessor.reader import _read_frame_pair
from corridorkey_new.preprocessor.resize import resize_frame
from corridorkey_new.preprocessor.tensor import to_tensor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for the preprocessing stage.

    Attributes:
        img_size: Square resolution the model runs at. 2048 is the native
            training resolution — do not change unless retraining.
        device: PyTorch device string ("cuda", "mps", "cpu").
        resize_strategy: How to fit the frame into img_size × img_size.
            "squish" stretches to square (fast, mild distortion).
            "letterbox" pads the shorter dimension with black (preserves
            aspect ratio — not yet implemented, falls back to squish).
    """

    img_size: int = 2048
    device: str = "cpu"
    resize_strategy: Literal["squish", "letterbox"] = "squish"


@dataclass(frozen=True)
class FrameMeta:
    """Metadata carried alongside the tensor for use by postprocessing.

    Attributes:
        frame_index: Index of this frame within the clip's frame_range.
        original_h: Frame height before resizing, in pixels.
        original_w: Frame width before resizing, in pixels.
    """

    frame_index: int
    original_h: int
    original_w: int


@dataclass(frozen=True)
class PreprocessedFrame:
    """Output contract of the preprocessing stage.

    Attributes:
        tensor: Model input tensor [1, 4, img_size, img_size] on device.
            Channels: [R_norm, G_norm, B_norm, alpha_hint].
        meta: Original frame dimensions and index for postprocessing.
    """

    tensor: torch.Tensor
    meta: FrameMeta


def preprocess_frame(
    manifest: ClipManifest,
    i: int,
    config: PreprocessConfig,
    *,
    image_files: list[Path] | None = None,
    alpha_files: list[Path] | None = None,
) -> PreprocessedFrame:
    """Preprocess one frame from a clip for model inference.

    Args:
        manifest: ClipManifest from stage 1. Must have needs_alpha=False.
        i: Frame index within manifest.frame_range.
        config: img_size, device, resize_strategy.
        image_files: Pre-sorted image paths (build once per clip, pass every frame).
        alpha_files: Pre-sorted alpha paths (build once per clip, pass every frame).

    Returns:
        PreprocessedFrame — tensor on device + FrameMeta for postprocessing.

    Raises:
        ValueError: If manifest still needs alpha or i is out of range.
        FrameReadError: If a frame file cannot be read.
    """
    # Step 1 — validate
    if manifest.needs_alpha:
        raise ValueError(f"Clip '{manifest.clip_name}' still needs alpha — call resolve_alpha() before preprocessing.")
    start, end = manifest.frame_range
    if not (start <= i < end):
        raise ValueError(f"Frame index {i} is out of range [{start}, {end}) for clip '{manifest.clip_name}'.")

    # Step 2 — read from disk
    image_path, alpha_path = _resolve_paths(manifest, i, image_files, alpha_files)
    image, alpha = _read_frame_pair(image_path, alpha_path)

    # Step 3 — capture original dimensions before any resizing
    original_h, original_w = image.shape[:2]

    # Step 4 — color space: linear → sRGB (on original resolution data)
    if manifest.is_linear:
        image = linear_to_srgb(image)

    # Step 5 — resize image and alpha to model resolution
    image, alpha = resize_frame(image, alpha, config.img_size, config.resize_strategy)

    # Step 6 — ImageNet normalisation (image only)
    image = normalise_image(image)

    # Steps 7–9 — concat, transpose, move to device
    tensor = to_tensor(image, alpha, config.device)

    logger.debug(
        "preprocess_frame clip=%s i=%d original=(%d,%d) img_size=%d device=%s",
        manifest.clip_name,
        i,
        original_h,
        original_w,
        config.img_size,
        config.device,
    )

    # Step 10 — return
    return PreprocessedFrame(
        tensor=tensor,
        meta=FrameMeta(frame_index=i, original_h=original_h, original_w=original_w),
    )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _resolve_paths(
    manifest: ClipManifest,
    i: int,
    image_files: list[Path] | None,
    alpha_files: list[Path] | None,
) -> tuple[Path, Path]:
    """Resolve image and alpha paths for frame index i."""
    imgs = image_files if image_files is not None else get_frame_files(manifest.frames_dir)
    alps = alpha_files if alpha_files is not None else get_frame_files(manifest.alpha_frames_dir)  # type: ignore[arg-type]
    offset = manifest.frame_range[0]
    return imgs[i - offset], alps[i - offset]
