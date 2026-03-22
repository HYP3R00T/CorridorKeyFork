"""Preprocessing stage — orchestrator.

Runs steps 1–10 in order. Owns no transformation logic itself — each step
is delegated to its own module.

    Step 1  — validate inputs            (here)
    Step 2  — read from disk             → reader.py
    Step 3  — capture original dims      (here)
    Step 4  — capture source_image       (here, CPU — no GPU→CPU transfer)
    Step 5  — move to device tensor      → tensor.py  (single PCIe transfer)
    Step 6  — color space conversion     → colorspace.py
    Step 7  — letterbox to model resolution → resize.py
    Step 8  — ImageNet normalisation     → normalise.py
    Steps 9–10 — concat + return         (here)

All transforms from step 5 onward run on the configured device (CUDA, MPS,
or CPU) — no separate fallback is needed; PyTorch handles device dispatch.

Public entry point: preprocess_frame(manifest, i, config)
Each stage in the pipeline has a corresponding orchestrator.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from corridorkey_new.stages.loader.contracts import ClipManifest
from corridorkey_new.stages.loader.validator import get_frame_files
from corridorkey_new.stages.preprocessor.colorspace import linear_to_srgb, linear_to_srgb_numpy
from corridorkey_new.stages.preprocessor.normalise import normalise_image
from corridorkey_new.stages.preprocessor.reader import _read_frame_pair
from corridorkey_new.stages.preprocessor.resize import (
    DEFAULT_ALPHA_UPSAMPLE_MODE,
    DEFAULT_UPSAMPLE_MODE,
    LetterboxPad,
    UpsampleMode,
    letterbox_frame,
)
from corridorkey_new.stages.preprocessor.tensor import to_tensors

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
        upsample_mode: Interpolation mode used when the source image is smaller
            than img_size. "bicubic" (default) gives the sharpest result.
            "bilinear" is faster but slightly softer. Has no effect when
            downscaling — area mode is always used then.
        alpha_upsample_mode: Interpolation mode for upscaling the alpha matte.
            Defaults to "bilinear" to avoid bicubic ringing on matte edges,
            which can produce negative alpha values at sharp transitions.
        half_precision: If True, cast tensors to float16 before inference.
            Halves VRAM usage and PCIe bandwidth. Requires the model to
            support float16 (most modern inference runtimes do). Default False.
        source_passthrough: If True, carry the original sRGB source image in
            FrameMeta so the postprocessor can replace model FG in opaque
            interior regions with original source pixels. Eliminates dark
            fringing caused by background contamination in the model FG output.
            Disable for a small speed gain if fringing is not a concern.
        sharpen_strength: Unsharp mask strength applied after upscaling.
            0.0 (default) disables sharpening. Typical range 0.1–0.5.
            Has no effect when downscaling. Enable in the quality profile
            to recover softness introduced by the antialias filter.
    """

    img_size: int = 2048
    device: str = "cpu"
    upsample_mode: UpsampleMode = DEFAULT_UPSAMPLE_MODE
    alpha_upsample_mode: UpsampleMode = DEFAULT_ALPHA_UPSAMPLE_MODE
    half_precision: bool = False
    source_passthrough: bool = True
    sharpen_strength: float = 0.0

    # ------------------------------------------------------------------
    # Named profiles — factory constructors for common use cases.
    # These produce a PreprocessConfig tuned for a specific quality/speed
    # tradeoff. The pipeline itself has no conditional logic — it just
    # reads the config fields.
    # ------------------------------------------------------------------

    @classmethod
    def quality(cls, device: str = "cpu", img_size: int = 2048) -> PreprocessConfig:
        """Highest quality — bicubic image upscale, bilinear alpha, float32, sharpening on.

        Best for final renders where quality is the priority.
        """
        return cls(
            img_size=img_size,
            device=device,
            upsample_mode="bicubic",
            alpha_upsample_mode="bilinear",
            half_precision=False,
            source_passthrough=True,
            sharpen_strength=0.3,
        )

    @classmethod
    def balanced(cls, device: str = "cpu", img_size: int = 2048) -> PreprocessConfig:
        """Balanced quality and speed — bilinear for both, float32, no sharpening.

        Good default for most production workflows.
        """
        return cls(
            img_size=img_size,
            device=device,
            upsample_mode="bilinear",
            alpha_upsample_mode="bilinear",
            half_precision=False,
            source_passthrough=True,
            sharpen_strength=0.0,
        )

    @classmethod
    def speed(cls, device: str = "cpu", img_size: int = 2048) -> PreprocessConfig:
        """Maximum speed — bilinear for both, float16, no source passthrough, no sharpening.

        For previews, proxies, or hardware-constrained environments.
        Requires the model and device to support float16.
        """
        return cls(
            img_size=img_size,
            device=device,
            upsample_mode="bilinear",
            alpha_upsample_mode="bilinear",
            half_precision=True,
            source_passthrough=False,
            sharpen_strength=0.0,
        )

    def __post_init__(self) -> None:
        if self.img_size <= 0:
            raise ValueError(f"PreprocessConfig.img_size must be > 0, got {self.img_size}.")


@dataclass(frozen=True)
class FrameMeta:
    """Metadata carried alongside the tensor for use by postprocessing.

    Attributes:
        frame_index: Index of this frame within the clip's frame_range.
        original_h: Frame height before resizing, in pixels.
        original_w: Frame width before resizing, in pixels.
        pad: Letterbox padding offsets added during preprocessing. The
            postprocessor uses these to crop the model output back to the
            original aspect ratio before scaling to source resolution.
        source_image: Original sRGB image [H, W, 3] float32, RGB channel order,
            at source resolution, used by postprocessor source_passthrough to
            replace model FG in opaque interior regions. None if source
            passthrough is disabled.
    """

    frame_index: int
    original_h: int
    original_w: int
    pad: LetterboxPad = None  # type: ignore[assignment]
    source_image: np.ndarray | None = None

    def __post_init__(self) -> None:
        # Default pad to a no-op LetterboxPad if not provided
        if self.pad is None:
            object.__setattr__(self, "pad", LetterboxPad(0, 0, 0, 0, self.original_h, self.original_w))


@dataclass(frozen=True)
class PreprocessedFrame:
    """Output contract of the preprocessing stage.

    Attributes:
        tensor: Model input tensor [1, 4, img_size, img_size] on device.
            Channels: [R_norm, G_norm, B_norm, alpha_hint].
            dtype is float32 unless PreprocessConfig.half_precision=True.
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
        config: Preprocessing configuration (img_size, device, modes, etc.).
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
    image, alpha, bgr = _read_frame_pair(image_path, alpha_path)

    # Step 3 — capture original dimensions before any resizing
    original_h, original_w = image.shape[:2]

    # Step 4 — capture source_image on CPU before it ever goes to the device.
    # Doing this here avoids a GPU→CPU transfer on every frame — the array
    # never leaves CPU, so there is no PCIe round-trip.
    # For linear inputs we apply the sRGB conversion on CPU so the
    # source_image stays consistent with what the model sees.
    # Note: source_image is always in RGB order for the postprocessor.
    source_image: np.ndarray | None = None
    if config.source_passthrough:
        rgb = image[:, :, ::-1].copy() if bgr else image.copy()
        source_image = linear_to_srgb_numpy(rgb) if manifest.is_linear else rgb

    # Step 5 — move to device; single PCIe transfer for image+alpha combined.
    # BGR→RGB reorder happens on-device (zero-copy strided index on CUDA).
    img_t, alp_t = to_tensors(image, alpha, config.device, bgr=bgr)  # [1,3,H,W], [1,1,H,W]

    # Step 6 — color space: linear → sRGB (on device)
    if manifest.is_linear:
        img_t = linear_to_srgb(img_t)

    # Step 7 — letterbox to model resolution (on device)
    # Preserves aspect ratio; pads remainder with mean pixel value.
    # Returns pad offsets so the postprocessor can crop back.
    img_t, alp_t, pad = letterbox_frame(
        img_t,
        alp_t,
        config.img_size,
        upsample_mode=config.upsample_mode,
        alpha_upsample_mode=config.alpha_upsample_mode,
        sharpen_strength=config.sharpen_strength,
        is_srgb=not manifest.is_linear,
    )

    # Step 8 — ImageNet normalisation (image only, on device, in-place)
    img_t = normalise_image(img_t)

    # Step 9 — concat channels → [1, 4, H, W]
    tensor = torch.cat([img_t, alp_t], dim=1)

    # Step 10 — optional half precision cast
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
            pad=pad,
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
