"""Preprocessing stage — output contracts.

Defines the data types produced by this stage and consumed by downstream
stages (inference, postprocessor).

Consistent with the other stages:
    scanner   → contracts.py  (Clip, SkippedPath, ScanResult)
    loader    → contracts.py  (ClipManifest)
    preprocessor → contracts.py  (FrameMeta, PreprocessedFrame)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from corridorkey.stages.preprocessor.resize import LetterboxPad


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
            Defaults to a no-op LetterboxPad (all zeros) when not provided.
        source_image: Original sRGB image [H, W, 3] float32, RGB channel order,
            at source resolution, used by postprocessor source_passthrough to
            replace model FG in opaque interior regions. None if source
            passthrough is disabled.
    """

    frame_index: int
    original_h: int
    original_w: int
    pad: LetterboxPad | None = None
    source_image: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.pad is None:
            object.__setattr__(self, "pad", LetterboxPad(0, 0, 0, 0, self.original_h, self.original_w))

    @property
    def resolved_pad(self) -> LetterboxPad:
        """Always returns a LetterboxPad — never None after __post_init__."""
        assert self.pad is not None  # guaranteed by __post_init__
        return self.pad


@dataclass(frozen=True)
class PreprocessedFrame:
    """Output contract of the preprocessing stage.

    Attributes:
        tensor: Model input tensor [1, 4, img_size, img_size] on device.
            Channels: [R_norm, G_norm, B_norm, alpha_hint].
            dtype is float32 unless PreprocessConfig.half_precision=True.
        meta: Original frame dimensions, letterbox pad offsets, and optional
            source image for postprocessing.
    """

    tensor: torch.Tensor
    meta: FrameMeta
