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
    """

    despill_strength: float = 1.0
    auto_despeckle: bool = True
    despeckle_size: int = 400
    checkerboard_size: int = 64
