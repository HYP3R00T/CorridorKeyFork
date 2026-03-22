"""Postprocessor stage — configuration contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FgUpsampleMode = Literal["bilinear", "bicubic", "lanczos4"]
AlphaUpsampleMode = Literal["bilinear", "lanczos4"]


@dataclass(frozen=True)
class PostprocessConfig:
    """Configuration for the postprocessor stage.

    Attributes:
        fg_upsample_mode: Interpolation mode for upscaling the foreground when
            the model resolution is smaller than the source. "bicubic" (default)
            is sharp and accurate. "lanczos4" is slightly sharper but slower.
            "bilinear" is fastest. Downscaling always uses INTER_AREA.
        alpha_upsample_mode: Interpolation mode for upscaling the alpha matte.
            "lanczos4" (default) gives the sharpest matte edges. "bilinear" is
            faster. Downscaling always uses INTER_AREA.
        despill_strength: Green spill suppression strength (0.0 = off, 1.0 = full).
        auto_despeckle: Remove small disconnected alpha islands.
        despeckle_size: Minimum connected region area in pixels to keep.
        checkerboard_size: Tile size in pixels for the preview composite background.
        source_passthrough: Replace model FG in opaque interior regions with the
            original source pixels. Eliminates dark fringing caused by background
            contamination in the model FG prediction. Requires source_image in
            FrameMeta (set PreprocessConfig.source_passthrough=True).
        edge_erode_px: Erosion radius (pixels) for the interior mask used by
            source_passthrough. Shrinks the interior region inward so the blend
            seam sits inside the subject rather than at the raw alpha edge.
        edge_blur_px: Gaussian blur radius for the source_passthrough blend seam.
            Higher values produce a softer transition between model FG and source.
    """

    fg_upsample_mode: FgUpsampleMode = "bicubic"
    alpha_upsample_mode: AlphaUpsampleMode = "lanczos4"
    despill_strength: float = 1.0
    auto_despeckle: bool = True
    despeckle_size: int = 400
    checkerboard_size: int = 64
    source_passthrough: bool = True
    edge_erode_px: int = 3
    edge_blur_px: int = 7
