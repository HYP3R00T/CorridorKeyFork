# Preprocessing pipeline — runs steps 1–9 in order.
# Owns no transformation logic itself — each step is delegated to its own module.
#
#   Step 1  — validate inputs            (here)
#   Step 2  — read from disk             → reader.py
#   Step 3  — capture original dims      (here)
#   Step 4  — capture source_image       (here, CPU — no GPU→CPU transfer)
#   Step 5  — move to device tensor      → tensor.py  (single PCIe transfer)
#   Step 6  — color space conversion     → colorspace.py
#   Step 7  — resize to model resolution → resize.py  (simple square stretch)
#   Step 8  — ImageNet normalisation     → normalise.py
#   Step 9  — concat + return            (here)
#
# All transforms from step 5 onward run on the configured device (CUDA, MPS,
# or CPU) — no separate fallback is needed; PyTorch handles device dispatch.

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from corridorkey.stages.loader.contracts import ClipManifest
from corridorkey.stages.loader.validator import list_frames
from corridorkey.stages.preprocessor.colorspace import linear_to_srgb, linear_to_srgb_numpy
from corridorkey.stages.preprocessor.contracts import FrameMeta, PreprocessedFrame
from corridorkey.stages.preprocessor.normalise import normalise_image
from corridorkey.stages.preprocessor.reader import read_frame_pair
from corridorkey.stages.preprocessor.resize import resize_frame
from corridorkey.stages.preprocessor.tensor import to_tensors

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for the preprocessing stage.

    Attributes:
        img_size: Square resolution the model runs at. 2048 is the native
            training resolution — do not change unless retraining.
        device: PyTorch device string ("cuda", "mps", "cpu").
        half_precision: If True, cast tensors to float16 before inference.
        source_passthrough: If True, carry the original sRGB source image in
            FrameMeta so the postprocessor can replace model FG in opaque
            interior regions with original source pixels.
    """

    img_size: int = 2048
    device: str = "cpu"
    half_precision: bool = False
    source_passthrough: bool = True

    def __post_init__(self) -> None:
        if self.img_size <= 0:
            raise ValueError(f"PreprocessConfig.img_size must be > 0, got {self.img_size}.")


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
        config: Preprocessing configuration.
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
        raise ValueError(f"Clip '{manifest.clip_name}' still needs alpha — call attach_alpha() before preprocessing.")
    start, end = manifest.frame_range
    if not (start <= i < end):
        raise ValueError(f"Frame index {i} is out of range [{start}, {end}) for clip '{manifest.clip_name}'.")

    # Step 2 — read from disk (CPU NumPy — I/O boundary)
    image_path, alpha_path = _resolve_paths(manifest, i, image_files, alpha_files)
    image, alpha, bgr = read_frame_pair(image_path, alpha_path)

    # Step 3 — capture original dimensions before any resizing
    original_h, original_w = image.shape[:2]

    # Step 4 — capture source_image and alpha_hint on CPU before going to device.
    # Avoids a GPU→CPU transfer on every frame.
    source_image: np.ndarray | None = None
    if config.source_passthrough:
        rgb = image[:, :, ::-1].copy() if bgr else image.copy()
        source_image = linear_to_srgb_numpy(rgb) if manifest.is_linear else rgb

    # alpha_hint: raw alpha at source resolution [H, W, 1] float32 0-1.
    # Stored before resize so the postprocessor can build a hard binary mask
    # at native resolution, avoiding soft edges from upscaling.
    alpha_hint: np.ndarray | None = None
    if alpha is not None:
        # alpha from read_frame_pair is [H, W] or [H, W, 1] float32 0-1
        arr = alpha if alpha.ndim == 3 else alpha[:, :, np.newaxis]
        alpha_hint = arr.astype(np.float32, copy=False)

    # Step 5 — move to device; single PCIe transfer for image+alpha combined.
    img_t, alp_t = to_tensors(image, alpha, config.device, bgr=bgr)  # [1,3,H,W], [1,1,H,W]

    # Step 6 — color space: linear → sRGB (on device)
    if manifest.is_linear:
        img_t = linear_to_srgb(img_t)

    # Step 7 — resize to model resolution (simple square stretch, on device)
    img_t, alp_t = resize_frame(
        img_t,
        alp_t,
        config.img_size,
    )

    # Step 8 — ImageNet normalisation (image only, on device)
    img_t = normalise_image(img_t)

    # Step 9 — concat channels → [1, 4, H, W] and optional half precision cast
    tensor = torch.cat([img_t, alp_t], dim=1)
    if config.half_precision:
        tensor = tensor.half()

    logger.debug(
        "preprocess_frame clip=%s i=%d original=(%d,%d) img_size=%d device=%s half=%s",
        manifest.clip_name,
        i,
        original_h,
        original_w,
        config.img_size,
        config.device,
        config.half_precision,
    )

    return PreprocessedFrame(
        tensor=tensor,
        meta=FrameMeta(
            frame_index=i,
            original_h=original_h,
            original_w=original_w,
            source_image=source_image,
            alpha_hint=alpha_hint,
        ),
    )


def _resolve_paths(
    manifest: ClipManifest,
    i: int,
    image_files: list[Path] | None,
    alpha_files: list[Path] | None,
) -> tuple[Path, Path]:
    imgs = image_files if image_files is not None else list_frames(manifest.frames_dir)
    assert manifest.alpha_frames_dir is not None  # noqa: S101 — guaranteed by needs_alpha=False check in preprocess_frame
    alps = alpha_files if alpha_files is not None else list_frames(manifest.alpha_frames_dir)
    offset = manifest.frame_range[0]
    return imgs[i - offset], alps[i - offset]
