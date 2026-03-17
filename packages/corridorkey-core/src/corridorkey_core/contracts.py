"""Data contracts flowing between pipeline stages 3 -> 4 -> 5.

These dataclasses are the typed boundaries between each compute stage.
Nothing outside corridorkey-core needs to import these directly - the
public API is create_engine() in corridorkey_core.__init__.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PreprocessedTensor:
    """Output of stage_3_preprocess. Input to stage_4_infer.

    Attributes:
        tensor: Float32 tensor [1, 4, img_size, img_size] on device.
            Channels 0-2 = ImageNet-normalised image, channel 3 = mask.
        img_size: Square resolution the tensor was prepared at.
        device: Torch device string the tensor lives on.
        source_h: Original frame height - carried through for upsampling in stage 5.
        source_w: Original frame width - carried through for upsampling in stage 5.
    """

    tensor: Any  # torch.Tensor - Any avoids a hard torch import at module level
    img_size: int
    device: str
    source_h: int
    source_w: int


@dataclass
class RawPrediction:
    """Output of stage_4_infer. Input to stage_5_postprocess.

    All arrays are at model resolution (img_size x img_size), float32.

    Attributes:
        alpha: Predicted alpha matte [img_size, img_size, 1], linear, 0-1.
        fg: Predicted foreground RGB [img_size, img_size, 3], sRGB straight, 0-1.
        img_size: Resolution of the prediction.
        source_h: Original frame height (needed for upsampling in stage 5).
        source_w: Original frame width (needed for upsampling in stage 5).
    """

    alpha: np.ndarray
    fg: np.ndarray
    img_size: int
    source_h: int
    source_w: int


@dataclass
class PostprocessParams:
    """Parameters controlling stage_5_postprocess behaviour.

    Attributes:
        despill_strength: Green spill suppression strength (0.0-1.0).
        auto_despeckle: Remove small disconnected alpha islands.
        despeckle_size: Minimum connected region area to keep (pixels).
        source_passthrough: Use original source pixels in opaque interior regions.
        edge_erode_px: Interior mask erosion radius for source passthrough.
        edge_blur_px: Transition seam blur radius for source passthrough.
        fg_is_straight: Whether the fg prediction is straight (unpremultiplied).
    """

    despill_strength: float = 1.0
    auto_despeckle: bool = True
    despeckle_size: int = 400
    source_passthrough: bool = False
    edge_erode_px: int = 3
    edge_blur_px: int = 7
    fg_is_straight: bool = True


@dataclass
class ProcessedFrame:
    """Output of stage_5_postprocess. Input to write_outputs.

    All arrays are at original source resolution, float32.

    Attributes:
        alpha: Alpha matte [H, W, 1], linear, 0-1.
        fg: Foreground RGB [H, W, 3], sRGB straight, 0-1.
        comp: Preview composite over checkerboard [H, W, 3], sRGB, 0-1.
        processed: Linear premultiplied RGBA [H, W, 4].
        source_h: Frame height.
        source_w: Frame width.
        stem: Filename stem carried through from the input frame.
    """

    alpha: np.ndarray
    fg: np.ndarray
    comp: np.ndarray
    processed: np.ndarray
    source_h: int
    source_w: int
    stem: str = ""
