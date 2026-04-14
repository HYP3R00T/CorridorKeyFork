from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class FrameMeta:
    """Metadata carried alongside the tensor between pipeline stages.

    .. note::
        **Pass-through type.** Downstream consumers (GUI, CLI, plugin) do not
        construct or inspect ``FrameMeta`` directly. It is created by
        :func:`~corridorkey.preprocess_frame`, embedded in
        :class:`~corridorkey.PreprocessedFrame` and
        :class:`~corridorkey.InferenceResult`, and consumed internally by
        :func:`~corridorkey.postprocess_frame`. It is exported so that custom
        :class:`~corridorkey.ModelBackend` implementations can satisfy the
        protocol without importing from submodules.

    Attributes:
        frame_index: Index of this frame within the clip's frame_range.
        original_h: Frame height before resizing, in pixels.
        original_w: Frame width before resizing, in pixels.
        source_image: Original sRGB image [H, W, 3] float32, RGB channel order,
            at source resolution, used by postprocessor source_passthrough to
            replace model FG in opaque interior regions. None if source
            passthrough is disabled.
        alpha_hint: Raw alpha hint [H, W, 1] float32 0-1 at source resolution,
            used by postprocessor hint_sharpen to produce a hard binary mask
            that eliminates soft edge tails introduced by upscaling. None if
            no alpha hint was provided.
    """

    frame_index: int
    original_h: int
    original_w: int
    source_image: np.ndarray | None = None
    alpha_hint: np.ndarray | None = None


@dataclass(frozen=True)
class PreprocessedFrame:
    """Output contract of the preprocessing stage.

    Attributes:
        tensor: Model input tensor [1, 4, img_size, img_size] on device.
            Channels: [R_norm, G_norm, B_norm, alpha_hint].
            dtype is float32 unless PreprocessConfig.half_precision=True.
        meta: Original frame dimensions and optional source image for postprocessing.
    """

    tensor: torch.Tensor
    meta: FrameMeta
