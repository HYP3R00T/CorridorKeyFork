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


def get_config_path() -> Path:
    """Return the global user config file path (~/.config/corridorkey/corridorkey.toml).

    Derived from the same logic utilityhub_config uses for global config lookup,
    so it always stays in sync with what load_settings() reads.
    """
    return Path.home() / ".config" / _APP_NAME / f"{_APP_NAME}.toml"


class PreprocessSettings(BaseModel):
    """User-facing preprocessing settings.

    Mirrors :class:`~corridorkey_new.preprocessor.PreprocessConfig` but lives
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
                "<6 GB → 1024, 6-12 GB → 1536, 12+ GB → 2048 (native training resolution). "
                "Set explicitly to override (e.g. 2048 for maximum quality regardless of VRAM)."
            ),
        ),
    ] = 0

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

    @classmethod
    def _expand_paths(cls, v: Path | str) -> Path:
        return expand_path(str(v) if isinstance(v, Path) else v)

    def to_preprocess_config(
        self, device: str | None = None, resolved_img_size: int | None = None
    ):  # -> PreprocessConfig
        """Build a :class:`~corridorkey_new.preprocessor.PreprocessConfig` from this config.

        Args:
            device: Override the device string. If None, uses ``self.device``
                (after ``resolve_device`` has been called by the interface).
            resolved_img_size: Pre-resolved img_size (from to_inference_config).
                If None, uses self.preprocess.img_size (or 2048 if 0).

        Returns:
            PreprocessConfig ready to pass to ``preprocess_frame``.
        """
        from corridorkey_new.preprocessor import PreprocessConfig

        img_size = resolved_img_size or self.preprocess.img_size or 2048

        return PreprocessConfig(
            img_size=img_size,
            device=device or self.device,
            resize_strategy=self.preprocess.resize_strategy,
        )

    def to_inference_config(self, device: str | None = None):  # -> InferenceConfig
        """Build an :class:`~corridorkey_new.inference.InferenceConfig` from this config.

        Args:
            device: Override the device string. If None, uses ``self.device``.

        Returns:
            InferenceConfig ready to pass to ``load_model`` / ``run_inference``.
        """
        import torch

        from corridorkey_new.inference import InferenceConfig
        from corridorkey_new.inference.config import adaptive_img_size
        from corridorkey_new.inference.orchestrator import _probe_vram_gb
        from corridorkey_new.infra.model_hub import default_checkpoint_path

        checkpoint = self.inference.checkpoint_path or default_checkpoint_path()
        resolved_device = device or self.device

        _precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        if self.inference.model_precision == "auto":
            model_dtype = _resolve_precision_auto(resolved_device)
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


def _resolve_precision_auto(device: str):  # returns torch.dtype, imported lazily
    """Auto-select model weight dtype based on device capabilities.

    - CPU            → float32
    - MPS            → bfloat16
    - CUDA Ampere+   → bfloat16  (compute capability >= 8.0)
    - CUDA pre-Ampere → float16
    - fallback       → float32
    """
    import torch

    dev = torch.device(device)
    if dev.type == "cpu":
        logger.info("Precision auto -> float32 (CPU)")
        return torch.float32
    if dev.type == "mps":
        logger.info("Precision auto -> bfloat16 (Apple Silicon MPS)")
        return torch.bfloat16
    if dev.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(dev)
        if props.major >= 8:
            logger.info("Precision auto -> bfloat16 (Ampere+ GPU: %s)", props.name)
            return torch.bfloat16
        logger.info("Precision auto -> float16 (pre-Ampere GPU: %s)", props.name)
        return torch.float16
    logger.info("Precision auto -> float32 (fallback)")
    return torch.float32


def export_config(config: CorridorKeyConfig, path: Path | None = None) -> Path:
    """Write the current configuration to a commented TOML file.

    Mirrors the ``ck config init`` pattern from corridorkey-cli.
    If no path is given, writes to ``~/.config/corridorkey/corridorkey.toml``.

    Args:
        config: Validated ``CorridorKeyConfig`` instance to export.
        path: Destination file path. Defaults to the global user config file.

    Returns:
        Absolute path to the written file.
    """
    dest = (path or get_config_path()).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _p(v: Path) -> str:
        return str(v.expanduser()).replace("\\", "/")

    checkpoint_line = (
        f'checkpoint_path = "{_p(config.inference.checkpoint_path)}"'
        if config.inference.checkpoint_path
        else '# checkpoint_path = "~/.config/corridorkey/models/CorridorKey_v1.0.pth"'
    )

    content = (
        "# CorridorKey configuration\n"
        "# Generated by `ck config init` — edit freely.\n"
        "# Changes take effect on the next run (no restart needed).\n"
        "# Override any value with an env var: CK_<KEY>=value\n"
        "#   e.g.  CK_DEVICE=cuda   or   CK_INFERENCE__OPTIMIZATION_MODE=lowvram\n"
        "\n"
        "# ---------------------------------------------------------------------------\n"
        "# Top-level\n"
        "# ---------------------------------------------------------------------------\n"
        "\n"
        "# Compute device for inference.\n"
        "# Options: auto | cuda | rocm | mps | cpu\n"
        "#   auto — picks the best available device at runtime (CUDA > MPS > CPU).\n"
        f'device = "{config.device}"\n'
        "\n"
        "# Minimum log level written to the log file.\n"
        "# Options: DEBUG | INFO | WARNING | ERROR\n"
        f'log_level = "{config.log_level}"\n'
        "\n"
        "# Directory where rotating log files are written.\n"
        f'log_dir = "{_p(config.log_dir)}"\n'
        "\n"
        "# ---------------------------------------------------------------------------\n"
        "# [preprocess]\n"
        "# ---------------------------------------------------------------------------\n"
        "\n"
        "[preprocess]\n"
        "\n"
        "# Internal model resolution (square). 0 = VRAM-adaptive auto-select:\n"
        "#   < 6 GB  -> 1024  (fast, fits 4 GB GPUs, e.g. RTX 3050 Laptop)\n"
        "#   6-12 GB -> 1536  (balanced quality / speed)\n"
        "#   12+ GB  -> 2048  (native training resolution, best quality)\n"
        "# Set explicitly to override regardless of VRAM, e.g. img_size = 2048\n"
        f"img_size = {config.preprocess.img_size}\n"
        "\n"
        "# How to fit non-square frames into the square model resolution.\n"
        "# Options: squish | letterbox\n"
        "#   squish    -- stretch to square (fast, mild distortion on wide footage)\n"
        "#   letterbox -- pad shorter dimension with black (preserves aspect ratio)\n"
        f'resize_strategy = "{config.preprocess.resize_strategy}"\n'
        "\n"
        "# ---------------------------------------------------------------------------\n"
        "# [inference]\n"
        "# ---------------------------------------------------------------------------\n"
        "\n"
        "[inference]\n"
        "\n"
        "# Path to the .pth model checkpoint.\n"
        "# Leave commented to use the auto-downloaded default location.\n"
        f"{checkpoint_line}\n"
        "\n"
        "# Enable the CNN refiner for sharper alpha edges.\n"
        "# Set to false to skip the refiner entirely (faster, softer edges).\n"
        f"use_refiner = {str(config.inference.use_refiner).lower()}\n"
        "\n"
        "# Run the forward pass under reduced-precision autocast.\n"
        "# Cuts VRAM usage with minimal quality impact. Ignored on CPU.\n"
        f"mixed_precision = {str(config.inference.mixed_precision).lower()}\n"
        "\n"
        "# Weight dtype for the model forward pass.\n"
        "# Options: auto | float32 | float16 | bfloat16\n"
        "#   auto      -- bfloat16 on Ampere+ (RTX 30xx+) / Apple Silicon,\n"
        "#                float16 on older CUDA GPUs, float32 on CPU\n"
        "#   float16   -- saves VRAM on older CUDA GPUs (pre-Ampere)\n"
        "#   bfloat16  -- preferred on Ampere+ and Apple Silicon\n"
        "#   float32   -- maximum numerical stability, highest VRAM usage\n"
        f'model_precision = "{config.inference.model_precision}"\n'
        "\n"
        "# Refiner tiling strategy.\n"
        "# Options: auto | speed | lowvram\n"
        "#   auto    -- probes VRAM; < 12 GB -> lowvram, else -> speed\n"
        "#   speed   -- full-frame refiner pass (best quality, needs ~12+ GB at 2048)\n"
        "#   lowvram -- tiled refiner 512x512 with 128px overlap (works on 4 GB)\n"
        f'optimization_mode = "{config.inference.optimization_mode}"\n'
        "\n"
        "# CNN refiner delta scale.\n"
        "# 1.0 = full refinement, 0.0 = skip refiner output entirely.\n"
        "# Reduce toward 0.5 to soften over-sharpened edges on noisy footage.\n"
        f"refiner_scale = {config.inference.refiner_scale}\n"
    )

    dest.write_text(content, encoding="utf-8")
    logger.info("Config exported to: %s", dest)
    return dest


def ensure_config_file() -> Path:
    """Write the default config file if it does not already exist.

    Mirrors the auto-create behaviour in corridorkey-cli's ``init`` command.

    Returns:
        Path to the config file (newly created or pre-existing).
    """
    dest = get_config_path()
    if not dest.exists():
        export_config(CorridorKeyConfig(), path=dest)
        logger.info("Created default config at %s", dest)
    return dest


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
