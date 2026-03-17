"""Centralised configuration for CorridorKey.

All tool-managed files (models, logs, cache) live under ``app_dir``,
which defaults to ``~/.config/corridorkey``. Users can override any
field via:

- A config file at ``~/.config/corridorkey/corridorkey.yaml``  (global)
- A ``corridorkey.yaml`` dropped in the current working directory  (project)
- Environment variables prefixed with ``CORRIDORKEY_``
- Runtime overrides passed to ``load_config()``

Precedence (lowest to highest):
    defaults < global config < project config < env vars < overrides

Example config file (``~/.config/corridorkey/corridorkey.yaml``):

    checkpoint_dir: ~/studio/shared/corridorkey/models
    device: cuda
    despill_strength: 0.85
    fg_format: exr
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator
from utilityhub_config import load_settings
from utilityhub_config.utils import expand_path

logger = logging.getLogger(__name__)

_APP_NAME = "corridorkey"
_DEFAULT_APP_DIR = Path("~/.config/corridorkey")
_DEFAULT_CHECKPOINT_DIR = Path("~/.config/corridorkey/models")


class CorridorKeyConfig(BaseModel):
    """Validated configuration for the CorridorKey pipeline.

    All Path fields support tilde (``~``) and environment variable
    expansion (e.g. ``$STUDIO_ROOT/models``).

    Load with :func:`load_config`. Export with :func:`export_config`.
    See ``docs/knowledge/configuration/index.md`` for the full reference.
    """

    # ------------------------------------------------------------------ #
    # Paths                                                                #
    # ------------------------------------------------------------------ #

    app_dir: Annotated[
        Path,
        Field(
            default=_DEFAULT_APP_DIR,
            description=(
                "Root directory for all tool-managed files (config, models, cache). "
                "Created on first use. Override to relocate the entire tool data directory."
            ),
        ),
    ] = _DEFAULT_APP_DIR

    checkpoint_dir: Annotated[
        Path,
        Field(
            default=_DEFAULT_CHECKPOINT_DIR,
            description=(
                "Directory where model checkpoints are stored. "
                "Override to a shared network path in studio environments so all "
                "workstations share a single model download."
            ),
        ),
    ] = _DEFAULT_CHECKPOINT_DIR

    model_download_url: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "URL to download the inference model checkpoint. "
                "Defaults to the official release URL when None. "
                "Override to point at a mirror or a local file server in air-gapped studios."
            ),
        ),
    ] = None

    model_filename: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Expected filename of the downloaded checkpoint. "
                "Defaults to the built-in constant in model_manager when None. "
                "Only change this if you are hosting a renamed build."
            ),
        ),
    ] = None

    log_dir: Annotated[
        Path,
        Field(
            default=Path("~/.config/corridorkey/logs"),
            description=(
                "Directory where rotating log files are written. "
                "Each CLI session appends to corridorkey.log (max 5 MB, 5 rotations kept). "
                "Share the latest log file when reporting bugs."
            ),
        ),
    ] = Path("~/.config/corridorkey/logs")

    log_level: Annotated[
        Literal["DEBUG", "INFO", "WARNING", "ERROR"],
        Field(
            default="INFO",
            description=(
                "Minimum log level written to the log file. "
                "'INFO' captures all normal processing events (recommended). "
                "'DEBUG' adds per-frame timing and internal engine details. "
                "'WARNING' logs only problems. "
                "The console always shows WARNING+ unless --verbose is passed."
            ),
        ),
    ] = "INFO"

    # ------------------------------------------------------------------ #
    # Device and engine                                                    #
    # ------------------------------------------------------------------ #

    device: Annotated[
        Literal["auto", "cuda", "mps", "cpu"],
        Field(
            default="auto",
            description=(
                "Compute device for inference. "
                "'auto' selects the best available device at runtime (CUDA > MPS > CPU). "
                "See docs/knowledge/configuration/index.md."
            ),
        ),
    ] = "auto"

    optimization_mode: Annotated[
        Literal["auto", "speed", "lowvram"],
        Field(
            default="auto",
            description=(
                "CNN refiner tiling strategy. "
                "'auto' probes available VRAM and picks 'speed' (>=12 GB free) or 'lowvram' (<12 GB). "
                "'speed' runs the full 2048x2048 refiner pass in one shot. "
                "'lowvram' tiles the refiner in 512x512 patches with 128px overlap. "
                "See docs/knowledge/improvements/optimization-modes.md."
            ),
        ),
    ] = "auto"

    precision: Annotated[
        Literal["auto", "fp16", "bf16", "fp32"],
        Field(
            default="auto",
            description=(
                "Floating point format for the model forward pass. "
                "'auto' selects bf16 on Ampere+/Apple Silicon, fp16 on older GPUs, fp32 on CPU. "
                "'fp32' is the safe choice for debugging or maximum numerical stability. "
                "See docs/knowledge/improvements/precision-modes.md."
            ),
        ),
    ] = "auto"

    # ------------------------------------------------------------------ #
    # Inference parameters                                                 #
    # ------------------------------------------------------------------ #

    input_is_linear: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "Treat input frames as linear light (e.g. EXR from a VFX pipeline). "
                "Leave False for standard sRGB footage (PNG, JPEG from a camera). "
                "Incorrect setting causes colour shifts in the output."
            ),
        ),
    ] = False

    despill_strength: Annotated[
        float,
        Field(
            default=1.0,
            ge=0.0,
            le=1.0,
            description=(
                "Green spill suppression strength. "
                "0.0 disables despill entirely. 1.0 applies full correction. "
                "Reduce toward 0.5 if skin tones look too magenta after keying."
            ),
        ),
    ] = 1.0

    auto_despeckle: Annotated[
        bool,
        Field(
            default=True,
            description=(
                "Remove small disconnected alpha islands from the matte automatically. "
                "Eliminates single-pixel noise and floating fragments. "
                "Disable if fine detail like hair strands is being incorrectly removed."
            ),
        ),
    ] = True

    despeckle_size: Annotated[
        int,
        Field(
            default=400,
            ge=0,
            description=(
                "Maximum connected region area in pixels to treat as a speckle and remove. "
                "Regions smaller than this threshold are deleted from the matte. "
                "Increase if small holes remain; decrease if fine detail like hair is being removed."
            ),
        ),
    ] = 400

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

    source_passthrough: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "Use original source pixels in fully opaque interior regions instead of "
                "the model's foreground prediction. Preserves full colour fidelity in "
                "faces, clothing, and fine patterns. Only the edge transition band uses "
                "the model's fg prediction. "
                "See docs/knowledge/improvements/source-passthrough.md."
            ),
        ),
    ] = False

    edge_erode_px: Annotated[
        int,
        Field(
            default=3,
            ge=0,
            description=(
                "Pixels to erode the interior mask inward before blending source and model fg. "
                "Acts as a safety buffer to avoid using raw source pixels near green-spill edges. "
                "Only applies when source_passthrough is True."
            ),
        ),
    ] = 3

    edge_blur_px: Annotated[
        int,
        Field(
            default=7,
            ge=0,
            description=(
                "Gaussian blur radius for the transition seam between source pixels and model fg. "
                "Larger values produce a softer, less visible boundary. "
                "Only applies when source_passthrough is True."
            ),
        ),
    ] = 7

    # ------------------------------------------------------------------ #
    # Output formats                                                       #
    # ------------------------------------------------------------------ #

    fg_format: Annotated[
        Literal["exr", "png"],
        Field(
            default="exr",
            description=(
                "File format for foreground (FG) output frames. "
                "'exr' preserves full float32 precision for compositing pipelines. "
                "'png' is 8-bit sRGB, suitable for quick review or web delivery."
            ),
        ),
    ] = "exr"

    matte_format: Annotated[
        Literal["exr", "png"],
        Field(
            default="exr",
            description=(
                "File format for alpha matte output frames. "
                "'exr' preserves full float32 precision. "
                "'png' is 8-bit greyscale, sufficient for most compositing workflows."
            ),
        ),
    ] = "exr"

    comp_format: Annotated[
        Literal["exr", "png"],
        Field(
            default="png",
            description=(
                "File format for composite preview frames (subject over checkerboard). "
                "PNG is the default - these frames are for quick review, not final delivery."
            ),
        ),
    ] = "png"

    processed_format: Annotated[
        Literal["exr", "png"],
        Field(
            default="png",
            description=(
                "File format for processed RGBA output frames (linear premultiplied). "
                "'exr' is recommended for compositing delivery. "
                "'png' is 8-bit and loses precision in the alpha channel."
            ),
        ),
    ] = "png"

    exr_compression: Annotated[
        Literal["dwaa", "piz", "zip", "none"],
        Field(
            default="dwaa",
            description=(
                "EXR compression codec applied to all EXR outputs. "
                "'dwaa' - lossy DCT, visually lossless, ~5x faster writes (recommended). "
                "'piz' - lossless wavelet, good for noisy images. "
                "'zip' - lossless ZIP deflate, widely compatible. "
                "'none' - uncompressed, maximum compatibility, largest files."
            ),
        ),
    ] = "dwaa"

    @field_validator("app_dir", "checkpoint_dir", "log_dir", mode="before")
    @classmethod
    def _expand_paths(cls, v: Path | str) -> Path:
        """Expand tilde and environment variables in path fields without requiring existence."""
        return expand_path(str(v) if isinstance(v, Path) else v)


def export_config(config: CorridorKeyConfig, path: str | Path | None = None) -> Path:
    """Write the current configuration to a YAML file.

    If no path is given, writes to ``~/.config/corridorkey/corridorkey.yaml``
    (the global user config location that ``load_config`` reads from).

    Args:
        config: Validated ``CorridorKeyConfig`` instance to export.
        path: Destination file path. Defaults to the global user config file.

    Returns:
        Absolute path to the written file.
    """
    dest = Path(path).expanduser() if path else config.app_dir / "corridorkey.yaml"
    dest.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# CorridorKey configuration",
        "# Generated by corridorkey.config.export_config()",
        "# See docs/knowledge/configuration/index.md for the full reference.",
        "",
    ]

    for field_name, field_info in CorridorKeyConfig.model_fields.items():
        # Write the field description as a comment above each entry.
        description = field_info.description
        if description:
            lines.extend(f"# {line}" for line in textwrap.wrap(description, width=78))

        value = getattr(config, field_name)
        if isinstance(value, Path):
            # Use forward slashes — safe on all platforms and avoids YAML
            # escape issues with Windows backslashes inside quoted strings.
            value = value.as_posix()
        if value is None:
            lines.append(f"{field_name}: null")
        elif isinstance(value, bool):
            lines.append(f"{field_name}: {'true' if value else 'false'}")
        elif isinstance(value, str):
            # Only quote strings that contain YAML-special characters.
            # Plain unquoted strings are safer on Windows (no backslash escaping).
            needs_quotes = any(
                c in value
                for c in (": ", "#", "[", "]", "{", "}", ",", "&", "*", "?", "|", "<", ">", "=", "!", "%", "@", "`")
            )
            if needs_quotes:
                lines.append(f"{field_name}: '{value}'")
            else:
                lines.append(f"{field_name}: {value}")
        else:
            lines.append(f"{field_name}: {value}")
        lines.append("")

    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Config exported to: %s", dest)
    return dest


def load_config(overrides: dict | None = None) -> CorridorKeyConfig:
    """Load and validate CorridorKey configuration from all sources.

    Resolution order (lowest to highest priority):
        1. Model field defaults
        2. ``~/.config/corridorkey/corridorkey.yaml`` (global user config)
        3. ``./corridorkey.yaml`` in the current working directory
        4. Environment variables prefixed with ``CORRIDORKEY_``
        5. ``overrides`` dict passed to this function

    After loading, ensures ``app_dir`` and ``checkpoint_dir`` exist on disk.

    Args:
        overrides: Optional dict of field values to apply at highest priority.
            Useful for CLI flags and programmatic configuration.

    Returns:
        Validated ``CorridorKeyConfig`` instance.

    Raises:
        utilityhub_config.errors.ConfigValidationError: If any field fails
            Pydantic validation (e.g. wrong type, out-of-range value).
    """
    try:
        config, _ = load_settings(
            CorridorKeyConfig,
            app_name=_APP_NAME,
            env_prefix="CORRIDORKEY",
            overrides=overrides,
        )
    except RuntimeError as exc:
        if "Failed to parse YAML" in str(exc):
            config_file = Path(_DEFAULT_APP_DIR).expanduser() / "corridorkey.yaml"
            if config_file.exists():
                backup = config_file.with_suffix(".yaml.bak")
                config_file.rename(backup)
                logger.warning(
                    "Corrupt config file moved to %s - using defaults. Run `corridorkey config init` to regenerate.",
                    backup,
                )
            config, _ = load_settings(
                CorridorKeyConfig,
                app_name=_APP_NAME,
                env_prefix="CORRIDORKEY",
                overrides=overrides,
            )
        else:
            raise

    config.app_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)

    logger.debug(
        "Config loaded: device=%s, checkpoint_dir=%s",
        config.device,
        config.checkpoint_dir,
    )
    return config
