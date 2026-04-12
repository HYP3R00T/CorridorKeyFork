"""Postprocessing stage — user-facing configuration settings."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class PostprocessSettings(BaseModel):
    """User-facing postprocessing settings.

    Mirrors :class:`~corridorkey.stages.postprocessor.PostprocessConfig` but
    lives in the config layer so it can be loaded from TOML.

    In ``corridorkey.toml``::

        [postprocess]
        fg_upsample_mode = "lanczos4"
        alpha_upsample_mode = "lanczos4"
        despill_strength = 0.5
        auto_despeckle = true
        despeckle_size = 400
        despeckle_dilation = 25
        despeckle_blur = 5
        source_passthrough = true
        edge_erode_px = 3
        edge_blur_px = 7
    """

    fg_upsample_mode: Annotated[
        Literal["bilinear", "bicubic", "lanczos4"],
        Field(
            default="lanczos4",
            description=(
                "Interpolation mode for upscaling the foreground when the model resolution "
                "is smaller than the source. 'lanczos4' (default) gives the sharpest result. "
                "'bicubic' is slightly faster. 'bilinear' is fastest. "
                "Downscaling always uses INTER_AREA regardless of this setting."
            ),
        ),
    ] = "lanczos4"

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
            default=0.5,
            ge=0.0,
            le=1.0,
            description="Green spill suppression strength. 0.0 = off, 1.0 = full suppression.",
        ),
    ] = 0.5

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

    despeckle_dilation: Annotated[
        int,
        Field(
            default=25,
            ge=0,
            description=(
                "Dilation radius in pixels applied after component removal to recover edges lost during binarisation."
            ),
        ),
    ] = 25

    despeckle_blur: Annotated[
        int,
        Field(
            default=5,
            ge=0,
            description="Gaussian blur radius applied after dilation to soften the hard mask edge.",
        ),
    ] = 5

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
            ge=0,
            description=(
                "Gaussian blur radius for the source_passthrough blend seam. "
                "0 = disabled. Higher values produce a softer transition between model FG and source."
            ),
        ),
    ] = 7

    hint_sharpen: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Apply a hard binary mask derived from the alpha hint to eliminate soft edge "
                "tails introduced by upscaling. Also zeros FG white bleed in the background zone. "
                "Requires an alpha hint (clip must have an alpha source)."
            ),
        ),
    ] = True

    hint_sharpen_dilation: Annotated[
        int,
        Field(
            default=3,
            ge=0,
            description=(
                "Dilation radius in pixels applied to the binarised hint before masking. "
                "Gives breathing room so fine model edge detail is not clipped."
            ),
        ),
    ] = 3

    debug_dump: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "Save intermediate debug images after each postprocessing step to a debug/ "
                "subfolder. Writes PNG snapshots of alpha and FG at: raw inference output, "
                "after hint_sharpen, after source_passthrough, after despeckle, after despill. "
                "Use this to diagnose whether quality issues come from the model or postprocessing."
            ),
        ),
    ] = False
