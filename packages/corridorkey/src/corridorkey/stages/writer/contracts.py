"""Writer stage — configuration contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ImageFormat = Literal["png", "exr"]


@dataclass(frozen=True)
class WriteConfig:
    """Configuration for the writer stage.

    Controls which outputs are written and in what format.
    All output subdirectories are created under ``output_dir``.

    Attributes:
        output_dir: Root directory for all outputs.
        alpha_enabled: Write the alpha matte.
        alpha_format: File format for alpha output ("png" or "exr").
        fg_enabled: Write the straight sRGB foreground colour image.
        fg_format: File format for fg output ("png" or "exr").
        processed_enabled: Write the premultiplied linear RGBA output.
            This is the primary compositor output — transparent regions are
            correctly zeroed out. Saved as EXR (float32) by default.
        processed_format: File format for processed output ("png" or "exr").
        comp_enabled: Write the checkerboard preview composite.
        comp_format: File format for comp output (always "png").
        exr_compression: EXR compression codec name.
            One of: "none", "rle", "zips", "zip", "piz", "pxr24", "dwaa", "dwab".
    """

    output_dir: Path
    alpha_enabled: bool = True
    alpha_format: ImageFormat = "png"
    fg_enabled: bool = True
    fg_format: ImageFormat = "png"
    processed_enabled: bool = True
    processed_format: ImageFormat = "png"
    comp_enabled: bool = True
    comp_format: Literal["png"] = "png"
    exr_compression: str = "dwaa"
