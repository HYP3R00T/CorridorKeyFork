"""Inference stage — output contract.

InferenceResult is the single output type of the inference stage.
It carries raw model predictions (still on device, still at model resolution)
and the FrameMeta needed by postprocessing to resize back to source resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from corridorkey.stages.preprocessor.contracts import FrameMeta


@dataclass(frozen=True)
class InferenceResult:
    """Output contract of the inference stage. Input to postprocessing.

    Attributes:
        alpha: Predicted alpha matte [1, 1, img_size, img_size], sigmoid-activated,
            range 0-1, on device.
        fg: Predicted foreground colour [1, 3, img_size, img_size], sigmoid-activated
            sRGB range 0-1, on device.
        meta: Original frame dimensions and index, carried through from preprocessing
            so postprocessing can resize outputs back to source resolution.
    """

    alpha: torch.Tensor
    fg: torch.Tensor
    meta: FrameMeta

    def __post_init__(self) -> None:
        if self.alpha.ndim != 4 or self.alpha.shape[1] != 1:
            raise ValueError(f"InferenceResult.alpha must be [B, 1, H, W], got {tuple(self.alpha.shape)}")
        if self.fg.ndim != 4 or self.fg.shape[1] != 3:
            raise ValueError(f"InferenceResult.fg must be [B, 3, H, W], got {tuple(self.fg.shape)}")
        if self.alpha.shape[0] != self.fg.shape[0]:
            raise ValueError(f"InferenceResult batch size mismatch: alpha={self.alpha.shape[0]}, fg={self.fg.shape[0]}")
        if self.alpha.shape[2:] != self.fg.shape[2:]:
            raise ValueError(
                f"InferenceResult spatial size mismatch: alpha={self.alpha.shape[2:]}, fg={self.fg.shape[2:]}"
            )
