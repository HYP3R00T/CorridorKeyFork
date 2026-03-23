"""Preprocessing stage — user-facing configuration settings."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class PreprocessSettings(BaseModel):
    """User-facing preprocessing settings.

    Mirrors :class:`~corridorkey.stages.preprocessor.PreprocessConfig` but
    lives in the config layer so it can be loaded from TOML / env vars.

    In ``corridorkey.toml``::

        [preprocess]
        img_size = 0  # 0 = auto-select based on VRAM
        image_upsample_mode = "bicubic"
        sharpen_strength = 0.3
        half_precision = false
        source_passthrough = true
    """

    img_size: Annotated[
        Literal[0, 512, 1024, 1536, 2048],
        Field(
            default=0,
            description=(
                "Square resolution the model runs at. "
                "0 (default) = auto-select based on available VRAM: "
                "<6 GB → 1024, 6–12 GB → 1536, 12+ GB → 2048. "
                "2048 is the native training resolution and produces the best output."
            ),
        ),
    ] = 0

    image_upsample_mode: Annotated[
        Literal["bicubic", "bilinear"],
        Field(
            default="bicubic",
            description=(
                "Interpolation mode when the source frame is smaller than img_size. "
                "'bicubic' (default) gives the sharpest result. "
                "'bilinear' is faster but slightly softer. "
                "Has no effect when downscaling — area mode is always used then."
            ),
        ),
    ] = "bicubic"

    sharpen_strength: Annotated[
        float,
        Field(
            default=0.3,
            ge=0.0,
            le=1.0,
            description=(
                "Unsharp mask strength applied after upscaling. "
                "0.3 (default) recovers softness from the antialias filter. "
                "0.0 disables. Has no effect when downscaling."
            ),
        ),
    ] = 0.3

    half_precision: Annotated[
        bool,
        Field(
            default=False,
            description=("Cast tensors to float16 before inference. Halves VRAM usage and PCIe bandwidth."),
        ),
    ] = False

    source_passthrough: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Carry the original sRGB source image through to the postprocessor. "
                "Enables replacing model FG in opaque interior regions with original "
                "source pixels, eliminating dark fringing."
            ),
        ),
    ] = True
