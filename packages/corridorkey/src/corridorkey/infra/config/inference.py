"""Inference stage — user-facing configuration settings."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from corridorkey.infra.model_hub import default_checkpoint_path


class InferenceSettings(BaseModel):
    """User-facing inference settings.

    Mirrors :class:`~corridorkey.stages.inference.InferenceConfig` but
    lives in the config layer so it can be loaded from TOML / env vars.

    In ``corridorkey.toml``::

        [inference]
        checkpoint_path = "~/models/greenformer.pth"
        use_refiner = true
        mixed_precision = true
        model_precision = "auto"
        refiner_mode = "auto"
        refiner_scale = 1.0
    """

    checkpoint_path: Annotated[
        Path,
        Field(
            default_factory=default_checkpoint_path,
            description=(
                "Path to the .pth model checkpoint file. "
                "Defaults to ~/.config/corridorkey/models/CorridorKey_v1.0.pth. "
                "The model is downloaded automatically on first run if not present."
            ),
        ),
    ] = Field(default_factory=default_checkpoint_path)

    use_refiner: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Enable the CNN refiner module. "
                "The refiner corrects transformer macroblocking artifacts at subject edges, "
                "producing sharp, production-quality alpha mattes. "
                "Disabling it is faster but results in visibly coarser edges."
            ),
        ),
    ] = True

    mixed_precision: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Run the model forward pass under autocast (fp16/bf16). "
                "Ignored on CPU. Reduces VRAM usage with minimal quality impact. "
                "Disable only if you observe numerical instability."
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
                "'float32' is the safe fallback for debugging or maximum numerical stability. "
                "'float16' / 'bfloat16' reduce VRAM usage on supported hardware."
            ),
        ),
    ] = "auto"

    refiner_mode: Annotated[
        Literal["auto", "full_frame", "tiled"],
        Field(
            default="auto",
            description=(
                "Controls how the CNN refiner executes. Output quality is identical for all modes. "
                "'auto' probes available VRAM and selects the best mode automatically "
                "(<12 GB → tiled, 12+ GB → full_frame). "
                "'full_frame' runs the refiner on the entire image at once — "
                "best throughput on GPUs with 12+ GB VRAM. "
                "'tiled' runs the refiner in 512×512 overlapping tiles — "
                "keeps peak VRAM flat, required on low-VRAM GPUs."
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
                "Scale factor applied to the CNN refiner's edge correction output. "
                "1.0 (default) applies full refinement. "
                "0.0 disables the refiner corrections entirely. "
                "Values between 0 and 1 blend between no refinement and full refinement."
            ),
        ),
    ] = 1.0
