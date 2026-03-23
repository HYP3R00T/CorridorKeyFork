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
            the model resolution is smaller than the source. "lanczos4" (default)
            gives the sharpest result. "bicubic" is slightly faster. "bilinear"
            is fastest. Downscaling always uses INTER_AREA.
        alpha_upsample_mode: Interpolation mode for upscaling the alpha matte.
            "lanczos4" (default) gives the sharpest matte edges. "bilinear" is
            faster. Downscaling always uses INTER_AREA.
        despill_strength: Green spill suppression strength (0.0 = off, 1.0 = full).
        auto_despeckle: Remove small disconnected alpha islands.
        despeckle_size: Minimum connected region area in pixels to keep.
        despeckle_dilation: Dilation radius in pixels applied after component removal
            to recover edges lost during binarisation. Default 25.
        despeckle_blur: Gaussian blur radius applied after dilation to soften the
            hard mask edge. Default 5.
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
        hint_sharpen: Apply a hard binary mask derived from the alpha hint to
            eliminate soft edge tails introduced by upscaling. Requires an alpha
            hint in FrameMeta. Default True.
        hint_sharpen_dilation: Dilation radius in pixels applied to the binarised
            hint before masking. Gives breathing room so fine model edge detail
            is not clipped. Default 3.
        debug_dump: Save raw inference output (before any postprocessing) to a
            ``debug/`` subfolder alongside the normal outputs. Writes four PNGs
            per frame: raw_alpha, raw_fg, post_hint_alpha, post_hint_fg.
            Useful for diagnosing whether quality issues originate in the model
            or in postprocessing. Default False.
    """

    fg_upsample_mode: FgUpsampleMode = "lanczos4"
    alpha_upsample_mode: AlphaUpsampleMode = "lanczos4"
    despill_strength: float = 0.5
    auto_despeckle: bool = True
    despeckle_size: int = 400
    despeckle_dilation: int = 25
    despeckle_blur: int = 5
    checkerboard_size: int = 128
    source_passthrough: bool = True
    edge_erode_px: int = 3
    edge_blur_px: int = 7
    hint_sharpen: bool = True
    hint_sharpen_dilation: int = 3
    debug_dump: bool = False
