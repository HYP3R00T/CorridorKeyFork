"""Preprocessing stage — orchestrator.

Runs steps 1–10 in order. Owns no transformation logic itself — each step
is delegated to its own module.

    Step 1  — validate inputs            (here)
    Step 2  — read from disk             → reader.py
    Step 3  — capture original dims      (here)
    Step 4  — move to device tensor      → tensor.py
    Step 5  — color space conversion     → colorspace.py
    Step 6  — resize                     → resize.py
    Step 7  — ImageNet normalisation     → normalise.py
    Steps 8–9 — concat + return          (here)
    Step 10 — return PreprocessedFrame   (here)

All transforms from step 5 onward run on the configured device (CUDA, MPS,
or CPU) — no separate fallback is needed; PyTorch handles device dispatch.

Public entry point: preprocess_frame(manifest, i, config)
Each stage in the pipeline has a corresponding orchestrator.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from corridorkey_new.loader.contracts import ClipManifest
from corridorkey_new.loader.validator import get_frame_files
from corridorkey_new.preprocessor.colorspace import linear_to_srgb
from corridorkey_new.preprocessor.normalise import normalise_image
from corridorkey_new.preprocessor.reader import _read_frame_pair
from corridorkey_new.preprocessor.resize import resize_frame
from corridorkey_new.preprocessor.tensor import to_tensors

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for the preprocessing stage.

    Attributes:
        img_size: Square resolution the model runs at. 2048 is the native
            training resolution — do not change unless retraining.
        device: PyTorch device string ("cuda", "mps", "cpu"). All transforms
            after the disk read run on this device. PyTorch handles CPU
            fallback transparently when no GPU is available.
        resize_strategy: How to fit the frame into img_size × img_size.
            "squish" stretches to square (fast, mild distortion).
            "letterbox" pads the shorter dimension with black (preserves
            aspect ratio — not yet implemented, falls back to squish).
        source_passthrough: If True, carry the original sRGB source image in
            FrameMeta so the postprocessor can replace model FG in opaque
            interior regions with original source pixels. Eliminates dark
            fringing caused by background contamination in the model FG output.
    """

    img_size: int = 2048
    device: str = "cpu"
    resize_strategy: Literal["squish", "letterbox"] = "squish"
    source_passthrough: bool = True


@dataclass(frozen=True)
class FrameMeta:
    """Metadata carried alongside the tensor for use by postprocessing.

    Attributes:
        frame_index: Index of this frame within the clip's frame_range.
        original_h: Frame height before resizing, in pixels.
        original_w: Frame width before resizing, in pixels.
        source_image: Original sRGB image [H, W, 3] float32 at source resolution,
            used by postprocessor source_passthrough to replace model FG in
            opaque interior regions. None if source passthrough is disabled.
    """

    frame_index: int
    original_h: int
    original_w: int
    source_image: np.ndarray | None = None


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

    All transforms after the disk read run on ``config.device``. PyTorch
    handles CPU/GPU dispatch transparently — no separate fallback is needed.

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

    # Step 2 — read from disk (CPU NumPy — I/O boundary)
    image_path, alpha_path = _resolve_paths(manifest, i, image_files, alpha_files)
    image, alpha = _read_frame_pair(image_path, alpha_path)

    # Step 3 — capture original dimensions before any resizing
    original_h, original_w = image.shape[:2]

    # Step 4 — move to device; all subsequent ops run on-device
    img_t, alp_t = to_tensors(image, alpha, config.device)  # [1,3,H,W], [1,1,H,W]

    # Step 5 — color space: linear → sRGB (on device)
    if manifest.is_linear:
        img_t = linear_to_srgb(img_t)

    # Carry the source sRGB image at original resolution for source_passthrough.
    # Pulled back to CPU numpy so the postprocessor can blend at full resolution
    # without needing to know the inference device.
    source_image: np.ndarray | None = (
        img_t.squeeze(0).permute(1, 2, 0).cpu().numpy() if config.source_passthrough else None
    )

    # Step 6 — resize image and alpha to model resolution (on device)
    img_t, alp_t = resize_frame(img_t, alp_t, config.img_size, config.resize_strategy)

    # Step 7 — ImageNet normalisation (image only, on device)
    img_t = normalise_image(img_t)

    # Steps 8–9 — concat channels → [1, 4, H, W]
    tensor = torch.cat([img_t, alp_t], dim=1)

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
        meta=FrameMeta(
            frame_index=i,
            original_h=original_h,
            original_w=original_w,
            source_image=source_image,
        ),
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
