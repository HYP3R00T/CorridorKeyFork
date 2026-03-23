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
        img_size = 2048
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
                "2048 is the native training resolution and produces the best output. "
                "Smaller values reduce VRAM usage at the cost of output quality."
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
                "Has no effect when downscaling — area mode is always used then. "
                "Alpha upscale is always bilinear internally (bicubic rings on matte edges)."
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
                "0.3 (default) recovers softness introduced by the antialias filter. "
                "0.0 disables sharpening. Typical range 0.1–0.5. "
                "Has no effect when downscaling."
            ),
        ),
    ] = 0.3

    half_precision: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "Cast tensors to float16 before inference. "
                "Halves VRAM usage and PCIe bandwidth. "
                "Requires the model and device to support float16."
            ),
        ),
    ] = False

    source_passthrough: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Carry the original sRGB source image through to the postprocessor. "
                "Enables replacing model FG in opaque interior regions with original "
                "source pixels, eliminating dark fringing. "
                "Disable for a small speed gain if fringing is not a concern."
            ),
        ),
    ] = True
