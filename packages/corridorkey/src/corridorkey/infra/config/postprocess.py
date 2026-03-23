"""Postprocessing stage — user-facing configuration settings."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class PostprocessSettings(BaseModel):
    """User-facing postprocessing settings.

    Mirrors :class:`~corridorkey.stages.postprocessor.PostprocessConfig` but
    lives in the config layer so it can be loaded from TOML / env vars.

    In ``corridorkey.toml``::

        [postprocess]
        fg_upsample_mode = "bicubic"
        alpha_upsample_mode = "lanczos4"
        despill_strength = 1.0
        auto_despeckle = true
        despeckle_size = 400
        source_passthrough = true
        edge_erode_px = 3
        edge_blur_px = 7
    """

    fg_upsample_mode: Annotated[
        Literal["bilinear", "bicubic", "lanczos4"],
        Field(
            default="bicubic",
            description=(
                "Interpolation mode for upscaling the foreground when the model resolution "
                "is smaller than the source. 'bicubic' (default) is sharp and accurate. "
                "'lanczos4' is slightly sharper but slower. 'bilinear' is fastest. "
                "Downscaling always uses INTER_AREA regardless of this setting."
            ),
        ),
    ] = "bicubic"

    alpha_upsample_mode: Annotated[
        Literal["bilinear", "lanczos4"],
        Field(
            default="lanczos4",
            description=(
                "Interpolation mode for upscaling the alpha matte. "
                "'lanczos4' (default) gives the sharpest matte edges. "
                "'bilinear' is faster. "
                "Downscaling always uses INTER_AREA regardless of this setting."
            ),
        ),
    ] = "lanczos4"

    despill_strength: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.0,
            le=1.0,
            description="Green spill suppression strength. 0.0 = off, 1.0 = full suppression.",
        ),
    ] = 1.0

    auto_despeckle: Annotated[
        bool,
        Field(
            default=True,
            description="Remove small disconnected alpha islands automatically.",
        ),
    ] = True

    despeckle_size: Annotated[
        int,
        Field(
            default=400,
            ge=1,
            description="Minimum connected region area in pixels to keep when auto_despeckle is enabled.",
        ),
    ] = 400

    source_passthrough: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Replace model FG in opaque interior regions with the original source pixels. "
                "Eliminates dark fringing caused by background contamination in the model FG prediction. "
                "Requires preprocess.source_passthrough = true."
            ),
        ),
    ] = True

    edge_erode_px: Annotated[
        int,
        Field(
            default=3,
            ge=0,
            description=(
                "Erosion radius in pixels for the interior mask used by source_passthrough. "
                "Shrinks the interior region inward so the blend seam sits inside the subject."
            ),
        ),
    ] = 3

    edge_blur_px: Annotated[
        int,
        Field(
            default=7,
            ge=1,
            description=(
                "Gaussian blur radius for the source_passthrough blend seam. "
                "Higher values produce a softer transition between model FG and source."
            ),
        ),
    ] = 7
