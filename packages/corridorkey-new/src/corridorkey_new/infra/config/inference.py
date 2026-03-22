"""Inference stage — user-facing configuration settings."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class InferenceSettings(BaseModel):
    """User-facing inference settings.

    Mirrors :class:`~corridorkey_new.stages.inference.InferenceConfig` but
    lives in the config layer so it can be loaded from TOML / env vars.

    In ``corridorkey.toml``::

        [inference]
        checkpoint_path = "~/models/greenformer.pth"
        use_refiner = true
        mixed_precision = true
        model_precision = "auto"
        optimization_mode = "auto"
        refiner_scale = 1.0
    """

    checkpoint_path: Annotated[
        Path | None,
        Field(
            default=None,
            description=(
                "Path to the .pth model checkpoint file. "
                "Defaults to ~/.config/corridorkey/models/CorridorKey_v1.0.pth. "
                "The model is downloaded automatically on first run if not present."
            ),
        ),
    ] = None

    use_refiner: Annotated[
        bool,
        Field(
            default=True,
            description="Enable the CNN refiner module for sharper alpha edges.",
        ),
    ] = True

    mixed_precision: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Run the forward pass under fp16 autocast. "
                "Ignored on CPU. Reduces VRAM usage with minimal quality impact."
            ),
        ),
    ] = True

    model_precision: Annotated[
        Literal["auto", "float32", "float16", "bfloat16"],
        Field(
            default="auto",
            description=(
                "Weight dtype for the model forward pass. "
                "'auto' selects bfloat16 on Ampere+/Apple Silicon, float16 on older GPUs, float32 on CPU. "
                "'float32' is the safe choice for debugging or maximum numerical stability."
            ),
        ),
    ] = "auto"

    optimization_mode: Annotated[
        Literal["auto", "speed", "lowvram"],
        Field(
            default="auto",
            description=(
                "'auto' probes VRAM and selects the best mode automatically. "
                "'speed' uses a full-frame refiner pass. "
                "'lowvram' tiles the refiner (512×512, 128px overlap) to fit in less VRAM."
            ),
        ),
    ] = "auto"

    refiner_scale: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.0,
            le=1.0,
            description=(
                "Scale factor for the CNN edge refiner's delta corrections. "
                "1.0 applies full refinement. 0.0 skips the refiner output entirely. "
                "Reducing toward 0.0 speeds up processing at the cost of edge quality."
            ),
        ),
    ] = 1.0
