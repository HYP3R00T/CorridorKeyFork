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
    resize_strategy = "squish"

    [inference]
    checkpoint_path = "~/models/greenformer.pth"
    use_refiner = true
    mixed_precision = true
    optimization_mode = "auto"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from utilityhub_config import load_settings
from utilityhub_config.utils import expand_path

logger = logging.getLogger(__name__)

_APP_NAME = "corridorkey"


class PreprocessSettings(BaseModel):
    """User-facing preprocessing settings.

    Mirrors :class:`~corridorkey_new.preprocessor.PreprocessConfig` but lives
    in the config layer so it can be loaded from TOML / env vars.
    """

    img_size: Annotated[
        int,
        Field(
            default=2048,
            ge=64,
            description=(
                "Square resolution the model runs at. "
                "2048 is the native training resolution — do not change unless retraining."
            ),
        ),
    ] = 2048

    resize_strategy: Annotated[
        Literal["squish", "letterbox"],
        Field(
            default="squish",
            description=(
                "'squish' stretches the frame to a square (fast, mild distortion). "
                "'letterbox' pads the shorter dimension with black (preserves aspect ratio)."
            ),
        ),
    ] = "squish"


class InferenceSettings(BaseModel):
    """User-facing inference settings.

    Mirrors :class:`~corridorkey_new.inference.InferenceConfig` but lives in
    the config layer so it can be loaded from TOML / env vars.

    ``checkpoint_path`` is the only required field — all others have sensible
    defaults.
    """

    checkpoint_path: Annotated[
        Path | None,
        Field(
            default=None,
            description="Path to the .pth model checkpoint file. Required to run inference.",
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
        Literal["float32", "float16"],
        Field(
            default="float32",
            description=(
                "Weight dtype. 'float32' is safe everywhere; 'float16' saves VRAM on CUDA but may reduce accuracy."
            ),
        ),
    ] = "float32"

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

    @classmethod
    def _expand_paths(cls, v: Path | str) -> Path:
        return expand_path(str(v) if isinstance(v, Path) else v)

    def to_preprocess_config(self, device: str | None = None):  # -> PreprocessConfig
        """Build a :class:`~corridorkey_new.preprocessor.PreprocessConfig` from this config.

        Args:
            device: Override the device string. If None, uses ``self.device``
                (after ``resolve_device`` has been called by the interface).

        Returns:
            PreprocessConfig ready to pass to ``preprocess_frame``.
        """
        from corridorkey_new.preprocessor import PreprocessConfig

        return PreprocessConfig(
            img_size=self.preprocess.img_size,
            device=device or self.device,
            resize_strategy=self.preprocess.resize_strategy,
        )

    def to_inference_config(self, device: str | None = None):  # -> InferenceConfig
        """Build an :class:`~corridorkey_new.inference.InferenceConfig` from this config.

        Args:
            device: Override the device string. If None, uses ``self.device``.

        Returns:
            InferenceConfig ready to pass to ``load_model`` / ``run_inference``.

        Raises:
            ValueError: If ``inference.checkpoint_path`` is not set.
        """
        import torch

        from corridorkey_new.inference import InferenceConfig

        if self.inference.checkpoint_path is None:
            raise ValueError(
                "inference.checkpoint_path is not set. Add it to corridorkey.toml or set CK_INFERENCE__CHECKPOINT_PATH."
            )

        _precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
        }

        return InferenceConfig(
            checkpoint_path=self.inference.checkpoint_path,
            device=device or self.device,
            img_size=self.preprocess.img_size,
            use_refiner=self.inference.use_refiner,
            mixed_precision=self.inference.mixed_precision,
            model_precision=_precision_map[self.inference.model_precision],
            optimization_mode=self.inference.optimization_mode,
        )


def load_config(overrides: dict | None = None) -> CorridorKeyConfig:
    """Load and validate CorridorKey configuration from all sources.

    Resolution order (lowest to highest priority):
        1. Model field defaults
        2. ``~/.config/corridorkey/corridorkey.toml`` (global user config)
        3. ``./corridorkey.toml`` in the current working directory
        4. Environment variables prefixed with ``CK_``
        5. ``overrides`` dict passed to this function

    Args:
        overrides: Optional dict of field values to apply at highest priority.

    Returns:
        Validated ``CorridorKeyConfig`` instance.
    """
    config, _ = load_settings(
        CorridorKeyConfig,
        app_name=_APP_NAME,
        env_prefix="CK",
        overrides=overrides,
    )

    config.log_dir.expanduser().mkdir(parents=True, exist_ok=True)

    logger.debug("Config loaded: %s", config.model_dump())
    return config
