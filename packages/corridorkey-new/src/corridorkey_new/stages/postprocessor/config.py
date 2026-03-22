"""Postprocessor stage — configuration contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PostprocessConfig:
    """Configuration for the postprocessor stage.

    Attributes:
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

    despill_strength: float = 1.0
    auto_despeckle: bool = True
    despeckle_size: int = 400
    checkerboard_size: int = 64
    source_passthrough: bool = True
    edge_erode_px: int = 3
    edge_blur_px: int = 7
