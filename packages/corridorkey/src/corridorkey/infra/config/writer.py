"""Writer stage — user-facing configuration settings."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

ImageFormat = Literal["png", "exr"]


class WriterSettings(BaseModel):
    """User-facing writer settings.

    Mirrors :class:`~corridorkey.stages.writer.WriteConfig` but lives in
    the config layer so it can be loaded from TOML / env vars.

    In ``corridorkey.toml``::

        [writer]
        alpha_enabled = true
        alpha_format = "png"
        fg_enabled = true
        fg_format = "png"
        processed_enabled = true
        processed_format = "png"
        comp_enabled = true
        exr_compression = "dwaa"
    """

    alpha_enabled: Annotated[
        bool,
        Field(default=True, description="Write the alpha matte output."),
    ] = True

    alpha_format: Annotated[
        ImageFormat,
        Field(default="png", description="File format for alpha output. 'png' or 'exr'."),
    ] = "png"

    fg_enabled: Annotated[
        bool,
        Field(default=True, description="Write the straight sRGB foreground colour image."),
    ] = True

    fg_format: Annotated[
        ImageFormat,
        Field(default="png", description="File format for foreground output. 'png' or 'exr'."),
    ] = "png"

    processed_enabled: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Write the premultiplied RGBA output. "
                "This is the primary compositor output — transparent regions are correctly zeroed out."
            ),
        ),
    ] = True

    processed_format: Annotated[
        ImageFormat,
        Field(default="png", description="File format for processed RGBA output. 'png' or 'exr'."),
    ] = "png"

    comp_enabled: Annotated[
        bool,
        Field(default=True, description="Write the checkerboard preview composite."),
    ] = True

    exr_compression: Annotated[
        Literal["none", "rle", "zips", "zip", "piz", "pxr24", "dwaa", "dwab"],
        Field(
            default="dwaa",
            description=(
                "EXR compression codec. 'dwaa' (default) gives good compression with fast decode. "
                "'zip' is lossless. 'none' is uncompressed."
            ),
        ),
    ] = "dwaa"
