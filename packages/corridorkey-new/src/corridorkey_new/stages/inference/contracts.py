"""Inference stage — output contract.

InferenceResult is the single output type of the inference stage.
It carries raw model predictions (still on device, still at model resolution)
and the FrameMeta needed by postprocessing to resize back to source resolution.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from corridorkey_new.stages.preprocessor.orchestrator import FrameMeta


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
