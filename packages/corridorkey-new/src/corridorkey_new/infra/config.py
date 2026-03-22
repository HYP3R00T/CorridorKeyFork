"""Centralised configuration for CorridorKey.

Settings can be overridden via:

- A config file at ``~/.config/corridorkey/corridorkey.toml``  (global)
- A ``corridorkey.toml`` in the current working directory  (project)
- Environment variables prefixed with ``CK_``
- Runtime overrides passed to ``load_config()``

Precedence (lowest to highest):
    defaults < global config < project config < env vars < overrides

Example ``corridorkey.toml``::

    [preprocess]
    img_size = 2048
    upsample_mode = "bicubic"
    alpha_upsample_mode = "bilinear"
    half_precision = false
    source_passthrough = true
    sharpen_strength = 0.0

    [inference]
    checkpoint_path = "~/models/greenformer.pth"
    use_refiner = true
    mixed_precision = true
    optimization_mode = "auto"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from utilityhub_config import load_settings

logger = logging.getLogger(__name__)

APP_NAME = "corridorkey"


class PreprocessSettings(BaseModel):
    """User-facing preprocessing settings.

    Mirrors :class:`~corridorkey_new.stages.preprocessor.PreprocessConfig` but lives
    in the config layer so it can be loaded from TOML / env vars.
    """

    img_size: Annotated[
        int,
        Field(
            default=0,
            ge=0,
            description=(
                "Square resolution the model runs at. "
                "0 (default) = auto-select based on available VRAM: "
                "<6 GB -> 1024, 6-12 GB -> 1536, 12+ GB -> 2048 (native training resolution). "
                "Set explicitly to override (e.g. 2048 for maximum quality regardless of VRAM)."
            ),
        ),
    ] = 0

    upsample_mode: Annotated[
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

    alpha_upsample_mode: Annotated[
        Literal["bicubic", "bilinear"],
        Field(
            default="bilinear",
            description=(
                "Interpolation mode for upscaling the alpha matte. "
                "Defaults to 'bilinear' to avoid bicubic ringing on matte edges. "
                "Has no effect when downscaling — area mode is always used then."
            ),
        ),
    ] = "bilinear"

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

    sharpen_strength: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            description=(
                "Unsharp mask strength applied after upscaling. "
                "0.0 (default) disables sharpening. Typical range 0.1–0.5. "
                "Has no effect when downscaling. "
                "Set to 0.3 for the quality profile to recover softness from the antialias filter."
            ),
        ),
    ] = 0.0


class InferenceSettings(BaseModel):
    """User-facing inference settings.

    Mirrors :class:`~corridorkey_new.stages.inference.InferenceConfig` but lives in
    the config layer so it can be loaded from TOML / env vars.

    ``checkpoint_path`` is the only required field — all others have sensible
    defaults.
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


class CorridorKeyConfig(BaseModel):
    """Validated configuration for the CorridorKey pipeline.

    All Path fields support tilde (``~``) and environment variable
    expansion (e.g. ``$STUDIO_ROOT/corridorkey``).

    Load with :func:`load_config`.
    """

    log_dir: Annotated[
        Path,
        Field(
            default=Path("~/.config/corridorkey/logs"),
            description=(
                "Directory where rotating log files are written. Share the latest log file when reporting bugs."
            ),
        ),
    ] = Path("~/.config/corridorkey/logs")

    log_level: Annotated[
        Literal["DEBUG", "INFO", "WARNING", "ERROR"],
        Field(
            default="INFO",
            description=(
                "'DEBUG' adds verbose internal details. "
                "'INFO' captures all normal processing events (recommended). "
                "'WARNING' logs only problems."
            ),
        ),
    ] = "INFO"

    device: Annotated[
        Literal["auto", "cuda", "rocm", "mps", "cpu"],
        Field(
            default="auto",
            description=(
                "Compute device for inference. "
                "'auto' detects the best available device at runtime (ROCm > CUDA > MPS > CPU). "
                "'cuda' forces NVIDIA GPU. 'rocm' forces AMD GPU. "
                "'mps' forces Apple Silicon. 'cpu' forces CPU."
            ),
        ),
    ] = "auto"

    preprocess: Annotated[
        PreprocessSettings,
        Field(default_factory=PreprocessSettings, description="Preprocessing stage settings."),
    ] = Field(default_factory=PreprocessSettings)

    inference: Annotated[
        InferenceSettings,
        Field(default_factory=InferenceSettings, description="Inference stage settings."),
    ] = Field(default_factory=InferenceSettings)

    def to_preprocess_config(
        self, device: str | None = None, resolved_img_size: int | None = None
    ):  # -> PreprocessConfig
        """Build a :class:`~corridorkey_new.stages.preprocessor.PreprocessConfig` from this config.

        Args:
            device: Override the device string. If None, uses ``self.device``
                (after ``resolve_device`` has been called by the interface).
            resolved_img_size: Pre-resolved img_size (from to_inference_config).
                If None, uses self.preprocess.img_size (or 2048 if 0).

        Returns:
            PreprocessConfig ready to pass to ``preprocess_frame``.
        """
        from corridorkey_new.stages.preprocessor import PreprocessConfig

        img_size = resolved_img_size or self.preprocess.img_size or 2048

        return PreprocessConfig(
            img_size=img_size,
            device=device or self.device,
            upsample_mode=self.preprocess.upsample_mode,
            alpha_upsample_mode=self.preprocess.alpha_upsample_mode,
            half_precision=self.preprocess.half_precision,
            source_passthrough=self.preprocess.source_passthrough,
            sharpen_strength=self.preprocess.sharpen_strength,
        )

    def to_inference_config(self, device: str | None = None):  # -> InferenceConfig
        """Build an :class:`~corridorkey_new.stages.inference.InferenceConfig` from this config.

        Args:
            device: Override the device string. If None, uses ``self.device``.

        Returns:
            InferenceConfig ready to pass to ``load_model`` / ``run_inference``.
        """
        import torch

        from corridorkey_new.infra.model_hub import default_checkpoint_path
        from corridorkey_new.stages.inference import InferenceConfig
        from corridorkey_new.stages.inference.config import adaptive_img_size
        from corridorkey_new.stages.inference.orchestrator import _probe_vram_gb

        checkpoint = self.inference.checkpoint_path or default_checkpoint_path()
        resolved_device = device or self.device

        _precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        if self.inference.model_precision == "auto":
            dev = torch.device(resolved_device)
            if dev.type == "cpu":
                logger.info("Precision auto -> float32 (CPU)")
                model_dtype = torch.float32
            elif dev.type == "mps":
                logger.info("Precision auto -> bfloat16 (Apple Silicon MPS)")
                model_dtype = torch.bfloat16
            elif dev.type == "cuda" and torch.cuda.is_available():
                props = torch.cuda.get_device_properties(dev)
                if props.major >= 8:
                    logger.info("Precision auto -> bfloat16 (Ampere+ GPU: %s)", props.name)
                    model_dtype = torch.bfloat16
                else:
                    logger.info("Precision auto -> float16 (pre-Ampere GPU: %s)", props.name)
                    model_dtype = torch.float16
            else:
                logger.info("Precision auto -> float32 (fallback)")
                model_dtype = torch.float32
        else:
            model_dtype = _precision_map[self.inference.model_precision]

        # Resolve img_size: 0 means auto-select based on VRAM.
        if self.preprocess.img_size == 0:
            vram_gb = _probe_vram_gb(resolved_device)
            img_size = adaptive_img_size(vram_gb)
            logger.info(
                "img_size auto: %.1f GB VRAM detected → img_size=%d",
                vram_gb,
                img_size,
            )
        else:
            img_size = self.preprocess.img_size

        return InferenceConfig(
            checkpoint_path=checkpoint,
            device=resolved_device,
            img_size=img_size,
            use_refiner=self.inference.use_refiner,
            mixed_precision=self.inference.mixed_precision,
            model_precision=model_dtype,
            optimization_mode=self.inference.optimization_mode,
            refiner_scale=self.inference.refiner_scale,
        )


def _load(overrides: dict | None) -> tuple[CorridorKeyConfig, Any]:
    config, metadata = load_settings(
        CorridorKeyConfig,
        app_name=APP_NAME,
        env_prefix="CK",
        overrides=overrides,
    )
    config.log_dir.expanduser().mkdir(parents=True, exist_ok=True)
    logger.debug("Config loaded: %s", config.model_dump())
    return config, metadata


def load_config(overrides: dict | None = None) -> CorridorKeyConfig:
    """Load and validate CorridorKey configuration from all sources.

    Resolution order (lowest to highest priority):
        defaults < global config < project config < env vars < overrides
    """
    config, _ = _load(overrides)
    return config


def load_config_with_metadata(overrides: dict | None = None) -> tuple[CorridorKeyConfig, Any]:
    """Like :func:`load_config` but also returns per-field source metadata."""
    return _load(overrides)
